# 关键设计决策记录

## ADR-001: Shell 入口层 + Python 编排层分离

**问题**: 需要同时支持多运行模式、超时控制、环境激活 —— 放在 Python 中会使 Hydra 配置过于复杂。

**决策**: 双层入口 —— `scripts/workflow.sh` 负责 Shell 级操作（conda、mamba、timeout、参数模式解析），`src/workflow.py` 负责业务编排。

**后果**:
- ✅ Shell 层天然支持 `timeout` 命令、conda 激活
- ✅ Python 层专注 Hydra 配置和业务逻辑
- ⚠️ 命令构建在两层重复: Shell 拼接 CLI → Python 解析 → `CommandBuilder` 再生成 shell 命令

**相关**: [`scripts/workflow.sh`](scripts/workflow.sh:1), [`src/workflow.py`](src/workflow.py:1)

---

## ADR-002: Code Snapshot 隔离实验（历史方案）

**问题**: 实验过程中代码变更会污染后续阶段结果，需要可复现的代码版本冻结。git worktree 逻辑复杂（snapshot → branch → worktree add → registry → cleanup script）。

**原方案**: git worktree + 自毁清理脚本。

**历史方案**: 利用 W&B `save_code` 元数据重建代码:
1. Sweep 阶段 run 自动记录 `run.metadata.git.commit` + `diff.patch`
2. eval/ablation/sensitivity 时，从 W&B API 查询 best_run 的 commit + diff
3. `CodeSnapshotService` 在姊妹目录中 `git init + fetch --depth=1 origin <commit> + checkout + apply diff`
4. 复制 `.env` / `data/` 到姊妹目录
5. `CommandBuilder` 在命令前插入 `cd '{sibling_dir}' &&`

**后果**:
- ✅ 无需 git worktree，无需 branch 管理，无需 registry
- ✅ 代码版本与 W&B run 绑定，跨机器可复现
- ✅ 姊妹目录可复用（`force_rebuild=false`）
- ⚠️ 依赖 git remote server 支持 shallow fetch
- ⚠️ 未提交修改通过 diff.patch 还原，冲突时回退到 base commit

**相关**: [`src/services/code_snapshot_service.py`](src/services/code_snapshot_service.py:1), [`src/services/command_builder.py`](src/services/command_builder.py:157), [`configs/workflow/mnist.yaml`](configs/workflow/mnist.yaml:137)

> 当前实现已由 ADR-012 的 immutable sandbox 取代。`workflow.snapshot.enabled=false` 仍作为调试开关保留，用于旧 sweep 缺少 source artifact 时直接在当前工作区执行。

---

## ADR-003: W&B 凭据零全局污染

**问题**: 在 `/proc/PID/cmdline` 中暴露 `WANDB_API_KEY` 是安全风险；Docker 无 DNS 环境需代理。

**决策**:
1. **命令行安全**: `CommandBuilder` 构建命令时不内联凭据，而是通过临时脚本 `export`
2. **代理安全**: `email_templates.py` 中 `send_email_proxy_aware()` 使用 PySocks 劫持 socket → HTTP CONNECT 隧道，try/finally 恢复全局状态

**后果**:
- ✅ `ps aux` 看不到 API key
- ✅ 邮件发送在代理环境零全局污染
- ⚠️ 临时脚本依赖 `/tmp` 自动清理

**相关**: [`src/services/command_builder.py`](src/services/command_builder.py:49), [`src/utils/email_templates.py`](src/utils/email_templates.py)

---

## ADR-004: 类型化异常替代 sys.exit(1)

**问题**: 裸 `sys.exit(1)` 无法被捕获，导致 tmux session 残留、W&B run 成为孤儿。

**决策**: 定义异常层次结构:
- `WorkflowError` — 基类
- `ConfigError`, `SweepError`, `EvaluationError`, `SessionError` — 子类

`@task_wrapper` 装饰器统一捕获 `WorkflowError`，在 `finally` 中执行 `wandb.finish()` 和清理。

**后果**:
- ✅ 单 trial 失败不终止整个 Hydra multirun
- ✅ 异常可被 `except` 捕获，实现优雅降级
- ✅ 孤儿 run 防护: tmux kill 前 SIGINT → wandb flush

**相关**: [`src/utils/exceptions.py`](src/utils/exceptions.py:1), [`src/utils/__init__.py`](src/utils/__init__.py) (task_wrapper)

---

## ADR-005: 评估策略 Strategy Pattern

**问题**: Evaluate/Override/Ablation/Sensitivity 都需要"重跑最优参数"，但行为不同（rerun vs resume）。

**决策**: 提取 [`EvaluationStrategy`](src/tasks/shared.py:234) 基类，`RerunStrategy` 和 `ResumeStrategy` 实现。`BaseTask.execute_strategy()` 根据配置实例化。

**后果**:
- ✅ 策略切换仅需改配置 `mode=rerun/resume`
- ✅ 新策略可扩展而不修改任务类
- ✅ `RerunStrategy` 支持多 GPU worker 并行

**相关**: [`src/tasks/shared.py`](src/tasks/shared.py:234), [`src/tasks/base.py`](src/tasks/base.py:122)

---

## ADR-006: Pipeline 断点恢复

**问题**: pipeline 多阶段运行中途失败，重新运行需要从断点续跑。

**决策**: 基于 sweep_id 的进度文件:
- 文件: `logs/workflow/.pipeline_progress_{sweep_id}`
- 每行记录已完成 stage name
- 显式传入 `target_sweep_id` 时强制重跑全部（清空 completed_stages）

**后果**:
- ✅ 失败后可 `bash scripts/workflow.sh pipeline <sweep_id>` 续跑
- ✅ `target_sweep_id` 显式时强制完整重跑，避免误用旧结果
- ⚠️ 进度文件为文本行格式，无版本控制

**相关**: [`src/tasks/shared.py`](src/tasks/shared.py:208), [`src/workflow.py`](src/workflow.py:136)

---

## ADR-007: top_n 评估 + eval_rank

**问题**: 消融/敏感性实验需要基于第 N 优参数（而非总是 best），且 Evaluate 阶段可能需要评估多个候选。

**决策**:
- `evaluate_task.top_n`: Evaluate 阶段评估前 N 个最优参数（默认 2，最大 3）
- `evaluate_task.eval_rank`: Ablation/Sensitivity 使用第 N 优参数（1=best, 2=2nd best）
- per-rank 时间窗口隔离 checkpoint 收集

**后果**:
- ✅ Evaluate 输出多 rank 报告（top-1, top-2...）
- ✅ 消融可对比不同基线的效果
- ✅ checkpoint 按时间窗口关联正确 rank

**相关**: [`src/tasks/evaluate.py`](src/tasks/evaluate.py:62), [`src/tasks/ablation.py`](src/tasks/ablation.py:71), [`src/tasks/sensitivity.py`](src/tasks/sensitivity.py:83)

---

## ADR-008: Grid 组合数上限

**问题**: 用户配置错误导致 grid 搜索组合爆炸（如 3 个参数各 10 值 = 1000 组合）。

**决策**: 配置化上限 `max_grid_combinations`（默认 100），workflow.py 和 SensitivityTask 均校验，超限抛出 `ConfigError`。

**后果**:
- ✅ 防止误操作导致的资源浪费
- ✅ 可 CLI 覆盖: `workflow.sweep_task.max_grid_combinations=200`

**相关**: [`src/workflow.py`](src/workflow.py:168), [`src/tasks/sensitivity.py`](src/tasks/sensitivity.py:133), [`configs/workflow/mnist.yaml`](configs/workflow/mnist.yaml:45)

---

## ADR-009: 配置 key 校验

**问题**: Hydra override key 易写错（如 `optimizer.lr` 应为 `model.optimizer.lr`），导致静默失败。

**决策**: `helpers.validate_config_keys()` 校验 key 前缀白名单，无效 key 打印警告，SensitivityTask 中升级为 `ConfigError` 阻断。

**后果**:
- ✅ 提前发现配置错误
- ⚠️ 白名单需随项目结构同步维护

**相关**: [`src/utils/helpers.py`](src/utils/helpers.py:46), [`src/tasks/sensitivity.py`](src/tasks/sensitivity.py:99)

---

## ADR-010: Email Markdown 存档 + 失败兜底

**问题**: SMTP 发送可能失败，实验结果不能丢失。

**决策**: 所有邮件内容同时保存 Markdown:
- 成功: `logs/mail/{mode}/{subject}_{ts}.md`
- 失败: 额外保存到 `logs/mail_fail/{mode}/`

**后果**:
- ✅ SMTP 故障不丢结果
- ✅ 本地可检索历史邮件内容

**相关**: [`src/utils/helpers.py`](src/utils/helpers.py:83)

---

## ADR-011: 论文风格绘图

**问题**: 默认 matplotlib 样式不适合论文投稿。

**决策**: `visualization.py` 封装纯函数，固定 serif 字体、3.5in 宽度、300 DPI、顶部/右侧 spine 移除。输出 PNG（嵌入邮件）+ PDF（附件）。

**后果**:
- ✅ 1D 折线图 + 2D 热力图直接可用
- ✅ 与 workflow 解耦，可独立使用

**相关**: [`src/utils/visualization.py`](src/utils/visualization.py:1)

---

## ADR-012: Immutable Sandbox 替代 Worktree 与 Git Patch

**问题**: 既有的 Worktree 方案导致项目根目录被大量隔离文件夹污染；而 ADR-002 中尝试的基于 Git diff 的 Code Snapshot 方案因依赖 Git Server 设置和极易发生 Patch 冲突而过于脆弱。同时，在跨机器和 Docker 容器下执行多 Agent Sweep 时，代码同步和环境隔离困难。

**决策**: 全面采用基于物理打包的“不可变沙盒 (Immutable Sandbox)”机制。在实验初始化阶段将代码打包成 tarball 存入 W&B Artifacts。各节点独立 Agent 执行时，在 `/tmp` 下创建沙盒，解压代码并用软链接（Symlink）接入重资源（如 `data/`, `.env`）。

**后果**:
- ✅ 彻底解决主目录污染问题，实验沙盒被限制在临时目录。
- ✅ 零 Patch 冲突，支持任意 Untracked 和未 Commit 修改的完美还原。
- ✅ 强力支持多机器和容器化 Agent 并行（结合已有的基础环境 `rsync`）。
- ✅ 天然适配巨型权重或数据集的零拷贝挂载需求。
- ✅ `snapshot.enabled=false` 可显式跳过沙盒，便于调试旧 sweep。
- ⚠️ 旧 sweep 若没有 `code_sweep_<id>` artifact，必须关闭 snapshot 或重新跑 sweep。

**相关**: [`memory-bank/sandbox_migration_plan.md`](memory-bank/sandbox_migration_plan.md)

---

## ADR-013: Orchestrator 与 Payload 的进程分离与沙盒化

**问题**: 在多机器、分布式评估和超参搜索时，如何保证任务调度的简易性与代码执行的一致性？

**决策**: 明确界定 Orchestrator（编排器）与 Payload（执行负载）的边界。
1. **Orchestrator (`workflow.py`)** 始终运行在宿主机原始目录下，利用当前最新代码负责查询 W&B、生成超参组合（Overrides）、以及分发沙盒。
2. **Payload (`train.py`, `eval.py`, `wandb agent`)** 必须通过子进程启动，并强制将其工作目录 (`cwd`) 设置为由 `SandboxService` 根据 `SWEEP_ID` 动态生成的 `/tmp` 沙盒中。

**后果**:
- ✅ Orchestrator 和 Payload 分离，用户可以在不影响当前运行中的 Sweep/Eval 的情况下继续本地开发。
- ✅ 彻底杜绝了并发执行时的环境污染与文件锁竞争。


---

## ADR-014: 真实邮件与 W&B Local 一致性修复

**问题**: 2026-05-21 真实跑 `evaluate / ablation / sensitivity` 时发现:
1. W&B Local 在删除并重建 group 后可能短时间返回 stale/deleted runs。
2. 单 seed 报告中的 pandas `std()` 为 `NaN`，会污染 JSON/CSV/邮件。
3. Ablation/Sensitivity 邮件复用了 Eval 的 JSON/CSV 字段，报告入口不准确。
4. HTML → Markdown 转换把 `<body>` 误识别为 `<b>`，邮件存档出现孤立 `**`。

**决策**:
- `WandbService.get_runs_by_group()` 跳过 `load(force=True)` 失败的 stale/deleted run。
- Evaluate 聚合强制刷新；Sensitivity 收集结果强制刷新并短重试。
- 单 seed `std` 归一为 `0.0`。
- 邮件模板支持 `report_label`，CSV 为 `N/A` 时隐藏该行。
- Markdown 转换正则只匹配完整 `b/strong/em` 标签。

**后果**:
- ✅ 真实邮件内容与实际报告路径一致。
- ✅ `num_seeds=1` 的冒烟测试不会出现 `NaN`。
- ✅ W&B Local 删除/重建延迟不会导致 Sensitivity 误判无结果。
