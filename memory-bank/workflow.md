# 工作流规范与阶段定义

## 1. 运行模式

| 模式 | 命令示例 | 说明 |
|------|----------|------|
| `pipeline` | `bash scripts/workflow.sh [pipeline] [SWEEP_ID]` | 全流程: sweep → evaluate → pipeline_tasks |
| `sweep` | `bash scripts/workflow.sh sweep` | 仅超参搜索 (Stage 1) |
| `evaluate` | `bash scripts/workflow.sh evaluate SWEEP_ID` | 仅多随机种子评估 (Stage 2) |
| `ablation` | `bash scripts/workflow.sh ablation SWEEP_ID` | 消融实验 (独立模式) |
| `sensitivity` | `bash scripts/workflow.sh sensitivity SWEEP_ID` | 参数敏感性 (独立模式) |
| `dry-run` | `bash scripts/workflow.sh dry-run [SWEEP_ID]` | 预览命令，不执行 |
| `notify` | `bash scripts/workflow.sh notify [SWEEP_ID]` | 启用邮件通知 |

## 2. 6 阶段工作流

```mermaid
flowchart LR
    S1["Stage 1: Sweep"] --> S2["Stage 2: Evaluate"]
    S2 --> S3["Stage 3: Override"]
    S3 --> S4["Stage 4: Grid"]
    S4 --> S5["Stage 5: Ablation"]
    S5 --> S6["Stage 6: Sensitivity"]
```

### Stage 1: Sweep — 超参数搜索

- **触发**: `task_name=sweep` 或 `pipeline` 且无 `target_sweep_id`
- **文件**: [`src/tasks/sweep.py`](src/tasks/sweep.py:1)
- **流程**:
  1. `CommandBuilder.build_wandb_sweep_command()` → 创建 W&B sweep
  2. `TmuxService.create_workers_session()` → 每个 GPU 一个 `wandb agent`
  3. 等待 sweep FINISHED
  4. 保存 `status_{sweep_id}.yaml`
  5. 发布 W&B source artifact: `code_sweep_{sweep_id}:latest`
- **输出**: sweep_id, tmux session, status YAML, source artifact

### Stage 2: Evaluate — 多随机种子评估

- **触发**: `pipeline_tasks` 中 `type=evaluate` 或 `task_name=evaluate`
- **文件**: [`src/tasks/evaluate.py`](src/tasks/evaluate.py:1)
- **流程**:
  1. 等待 sweep FINISHED
  2. `WandbService.find_top_n_runs(sweep, metric, top_n)` → 获取 top-N 最优 runs
  3. 对每个 rank:
     - `SandboxService.setup_sandbox()` → 从 W&B source artifact 重建代码到 `/tmp/lightning-runs/...`
     - 提取 config overrides
     - `RerunStrategy.execute(snapshot_dir=sandbox_dir)` → 在沙盒中多 GPU 并行重跑
     - `wait_for_session_with_timeout()` → 等待完成（默认 1h 超时）
  4. `_aggregate_and_report()`:
     - 收集每个 rank 的 metrics（mean/std/count/values）
     - 按时间窗口收集 per-rank checkpoints
     - 保存 `optimized_results_{sweep_id}.json`
     - 发送 per-rank 邮件
- **配置项**:
  - `evaluate_task.top_n`: 评估前 N 个参数（默认 2，最大 3）
  - `evaluate_task.num_seeds`: 每个参数重跑种子数（默认 5）
  - `evaluate_task.timeout_secs`: 超时秒数（默认 3600，0=无限制）
  - `evaluate_task.mode`: `rerun` | `resume`
  - `snapshot.enabled`: 是否启用代码快照（默认 true）
  - `snapshot.sibling_base_dir`: 姊妹目录基路径（默认项目父目录）

### Stage 3: Override — 强制覆盖评估

- **触发**: `pipeline_tasks` 中 `type=override`
- **文件**: [`src/tasks/override.py`](src/tasks/override.py:1)
- **流程**:
  1. 获取最优参数（`find_best_run`）
  2. 合并 `override_task.overrides`（强制覆盖变量）
  3. `RerunStrategy.execute()` 执行多种子评估
  4. 聚合报告 + 邮件
- **用途**: 验证某个特定参数变化的影响（如 batch_size=256）

### Stage 4: Grid — 笛卡尔积参数搜索

- **触发**: `pipeline_tasks` 中 `type=grid`
- **文件**: [`src/workflow.py`](src/workflow.py:162)
- **流程**:
  1. `itertools.product()` 展开所有参数组合
  2. `max_grid_combinations` 上限检查（默认 100）
  3. 每个组合作为一个 `OverrideTask` 子任务执行
- **用途**: 多参数敏感性分析（如 lin1_size × lin2_size）

### Stage 5: Ablation — 消融实验

- **触发**: `task_name=ablation` 或 `pipeline_tasks` 中 `type=ablation`
- **文件**: [`src/tasks/ablation.py`](src/tasks/ablation.py:1)
- **流程**:
  1. 前置检查: sweep 必须存在
  2. `ensure_evaluate_results()` — 无 evaluate 报告则自动运行（skip_email）
  3. 获取 rank-N 最优参数（`eval_rank`）
  4. 解析 `ablation_task.components` 列表
  5. **并行模式** (`parallel=true`): 所有消融组一次 launch，单 tmux session
  6. **串行模式**: 逐组执行
  7. 收集结果 → 对比表（Full Model vs 各变体 + Relative Drop%），单 seed `std=0.0`
  8. 发送消融对比邮件（含 Group URLs 与 ablation report directory）
- **配置项**:
  - `ablation_task.parallel`: 是否并行（默认 true）
  - `ablation_task.timeout_secs`: 超时（默认 600s）
  - `ablation_task.components`: 消融组件列表

### Stage 6: Sensitivity — 参数敏感性分析

- **触发**: `task_name=sensitivity` 或 `pipeline_tasks` 中 `type=sensitivity`
- **文件**: [`src/tasks/sensitivity.py`](src/tasks/sensitivity.py:1)
- **流程**:
  1. 前置检查: sweep 存在
  2. `ensure_evaluate_results()` — 自动获取 evaluate 基线
  3. 获取 rank-N 最优参数
  4. 解析 `sensitivity_task.sensitivities` 列表（或向后兼容 `param_grid`）
  5. `validate_config_keys()` 校验参数 key
  6. `max_grid_combinations` 上限检查
  7. 所有 study 的所有 combo 并行执行（单 tmux session，按 device round-robin）
  8. 按 study 收集结果 → force refresh + 短重试，按 config 匹配 combo runs
  9. **1D**: `plot_sensitivity_1d()` → 折线图（PNG+PDF）
  10. **2D**: `plot_sensitivity_2d()` → 热力图（PNG+PDF）
  11. 保存汇总 JSON 报告
  12. 发送敏感性邮件（嵌入图 + PDF 附件 + Group URLs + sensitivity JSON path）
- **配置项**:
  - `sensitivity_task.sensitivities`: 多 study 列表（新格式）
  - `sensitivity_task.param_grid`: 单 study（向后兼容）
  - `sensitivity_task.primary_metric`: 主指标（默认 test/acc）
  - `sensitivity_task.figure_width`: 图宽（默认 3.5in）
  - `sensitivity_task.figure_dpi`: DPI（默认 300）

## 3. W&B Group 命名规范

| 任务 | 格式 | 示例 |
|------|------|------|
| evaluate | `eval/{sweep_id}/top-{rank}` | `eval/abc123/top-1` |
| ablation | `ablation/{sweep_id}/[r{rank}/]{component}` | `ablation/abc123/r2/no_lin1_bn` |
| sensitivity | `sensitivity/{sweep_id}/[r{rank}/]{study}` | `sensitivity/abc123/r2/width_sensitivity` |
| override | `override/{sweep_id}/{name}` | `override/abc123/batch_size_256` |

## 4. 断点恢复规则

1. **自动恢复**: pipeline 运行时，`.pipeline_progress_{sweep_id}` 记录已完成阶段，重新运行自动跳过
2. **强制重跑**: 显式传入 `target_sweep_id` 时，`completed_stages = set()`，全部重新执行
3. **独立模式**（ablation/sensitivity）不受 pipeline 进度影响

## 5. 超时与清理规范

### 5.1 超时层级

| 层级 | 控制者 | 默认值 | 行为 |
|------|--------|--------|------|
| Shell | `timeout` 命令 | 300s | 整个进程 SIGTERM |
| Evaluate Python | `wait_for_session_with_timeout()` | 3600s | SIGINT tmux → graceful kill |
| Ablation Python | `wait_for_session_with_timeout()` | 600s | SIGINT tmux → graceful kill |
| Sensitivity Python | `wait_for_session_with_timeout()` | 600s | SIGINT tmux → graceful kill |

### 5.2 优雅停止

[`TmuxService.kill_session()`](src/services/tmux_service.py:69):
1. `tmux send-keys C-c` — SIGINT 给所有 pane，触发 wandb.finish()
2. 等待 `graceful_timeout`（默认 10s，上限 10s）
3. `tmux kill-session` — 强制终止

### 5.3 孤儿 run 防护

- SIGINT 触发 Lightning `on_exception` → `wandb.finish()`
- run 状态标记为 `failed` 而非永久 `running`

## 6. 邮件规范

### 6.1 通用内容

`evaluate` / `ablation` / `sensitivity` 会发送邮件；`sweep` 只发布 source artifact，不发送邮件。所有邮件均包含:
- Sweep ID、Sweep URL
- 最优配置（`rank N best`）
- 测试指标（mean ± std）
- 复现脚本（`seed=[42,43,44]` 格式）
- 复现脚本 Group URL
- Checkpoint 路径列表

### 6.2 模式专属内容

| 模式 | 专属内容 |
|------|----------|
| Evaluate | per-rank 分段展示（top-1, top-2...）、Baseline 对比 |
| Ablation | Full Model vs 各消融变体对比表、Relative Drop%、各 Group URL |
| Sensitivity | 参数网格结果表、嵌入图（PNG）、PDF 附件、各 Study Group URL |

### 6.3 邮件存档

- 成功: `logs/mail/{mode}/{subject}_{ts}.md`
- 失败: 额外保存 `logs/mail_fail/{mode}/{subject}_{ts}.md`


## 7. 四模式真实调试结论（2026-05-21）

- `sweep`: 真实创建 sweep `bs40ufzp`，tmux session 正常结束，`code_sweep_bs40ufzp` artifact 上传成功。
- `evaluate`: 使用真实 sweep `vi5gum5n`，邮件发送成功，报告 `optimized_results_vi5gum5n.json` 无 `NaN`。
- `ablation`: 默认组件 `no_lin1_bn` / `no_lin2_bn` 均完成，邮件展示 ablation report directory。
- `sensitivity`: `width_sensitivity` 2D grid + `lr_sensitivity` 1D grid 共 12 组完成，邮件展示 JSON/PNG/PDF 与复现脚本。

回归命令：

```bash
bash tests/integration/test_four_modes.sh
pytest tests/unit/test_email_templates.py tests/unit/test_workflow_task.py -q
```
