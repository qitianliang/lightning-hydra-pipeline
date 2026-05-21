# Lightning-Hydra-Workflow

基于 [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) 的自动化实验工作流框架。

## 核心特性

- **6 阶段工作流**: Sweep → Evaluate → Override → Grid → Ablation → Sensitivity
- **典型运行模式**: `pipeline`, `sweep`, `evaluate`, `ablation`, `sensitivity`
- **邮件通知**: 自动汇报实验结果、Run URL、复现脚本、W&B 链接；代理环境自动走 HTTP CONNECT 隧道（含 `getaddrinfo` 劫持），零全局污染
- **论文风格绘图**: 1D 折线图 + 2D 热力图 (serif, 3.5in, dpi=300 PDF)
- **配置 key 校验**: 自动检测 `optimizer.lr` → `model.optimizer.lr` 等错误
- **断点恢复**: pipeline 进度文件自动跳过已完成阶段；显式传入 `target_sweep_id` 时强制重跑全部阶段
- **不可变沙盒隔离**: sweep 结束后发布 W&B source artifact，evaluate/ablation/sensitivity 在 `/tmp/lightning-runs/...` 沙盒中执行 payload
- **CUDA_VISIBLE_DEVICES 自动推断**: `.env` 中设置 `CUDA_VISIBLE_DEVICES=0,1,2` 自动转为 `devices=[0,1,2]`
- **Evaluate 超时保护**: 默认 1h 超时, SIGINT 安全停, 防训练挂起
- **Grid 上限配置化**: `max_grid_combinations`, 默认 100, 可 CLI 覆盖
- **W&B 凭据安全**: 凭据 export 在脚本顶部, 不暴露于 `/proc/PID/cmdline`
- **Builder 安全**: `prepare_data()` 替代 `setup()`, 避免 Trainer 生命周期冲突
- **类型化异常**: `WorkflowError` 系列异常替代 `sys.exit(1)`, 支持 except 捕获
- **孤儿 run 防护**: tmux kill 前 SIGINT → wandb flush → 标记 failed

## 快速开始

```bash
# 1. 环境配置
cp .env.example .env  # 填入 WANDB_API_KEY, SMTP_* 等

# 2. Debug 运行 (experiment=example, limit_*=3)
bash scripts/workflow.sh pipeline                        # 全流程 (单卡，默认 devices=[0])
bash scripts/workflow.sh pipeline <SWEEP_ID>             # 复用 sweep
bash scripts/workflow.sh evaluate <SWEEP_ID>             # 只跑最优参数评估
bash scripts/workflow.sh ablation <SWEEP_ID>             # 消融实验
bash scripts/workflow.sh sensitivity <SWEEP_ID>          # 参数敏感性

# 3. 正式运行 (experiment=mnist_full, num_seeds=5)
TIMEOUT_SECS=0 bash scripts/workflow.sh pipeline "" \
  "workflow.evaluate_task.run_command.base_args=[python,src/train.py,experiment=mnist_full,trainer=gpu,logger.wandb.project=mnist]" \
  "workflow.evaluate_task.num_seeds=5"
```

## 运行模式

| 模式 | 命令 | 说明 |
|------|------|------|
| `pipeline` | `bash scripts/workflow.sh [pipeline] [SWEEP_ID]` | 全流程: sweep → evaluate → pipeline_tasks |
| `sweep` | `bash scripts/workflow.sh sweep` | 仅超参搜索 |
| `evaluate` | `bash scripts/workflow.sh evaluate SWEEP_ID` | 仅多种子评估并发送 Eval 邮件 |
| `ablation` | `bash scripts/workflow.sh ablation SWEEP_ID` | 消融实验 (对比表 + 邮件) |
| `sensitivity` | `bash scripts/workflow.sh sensitivity SWEEP_ID` | 参数敏感性 (1D/2D 绘图 + 邮件) |
| `dry-run` | `bash scripts/workflow.sh dry-run [SWEEP_ID]` | 预览命令 |

## 邮件内容

`evaluate` / `ablation` / `sensitivity` 会发送邮件；`sweep` 只创建 sweep、等待 agent 并发布 source artifact。邮件均包含 Sweep ID、Run URL、最优配置 (`rank N best`)、测试指标、复现脚本 (`seed=[42,43,44]` 格式)、Sweep URL。

- **Ablation**: Full Model vs 各消融变体对比表 + Relative Drop% + Group URLs
- **Sensitivity**: 参数网格结果表 + 嵌入图 (PNG) + PDF 附件 + Group URLs

## 测试与真实调试

```bash
# 离线四模式入口测试：不访问真实 W&B，不创建 tmux
bash tests/integration/test_four_modes.sh

# 单元覆盖：邮件内容、ablation components、sensitivity studies/grid 展开
pytest tests/unit/test_email_templates.py tests/unit/test_workflow_task.py -q

# 可选真实链路：需要有效 .env / W&B / SMTP 配置
RUN_REAL_WANDB=1 REAL_SWEEP_ID=<id> bash tests/integration/test_four_modes.sh
```

2026-05-21 真实调试覆盖了 `sweep / evaluate / ablation / sensitivity`。`evaluate`、`ablation`、`sensitivity` 的邮件均确认能发送并落盘到 `logs/mail/{mode}/`；单 seed 报告统一输出 `std=0.0`，不会在 JSON/CSV/邮件中出现 `NaN`。

## 项目结构

```
src/
├── workflow.py              # 主编排器
├── tasks/                   # 解耦任务模块
│   ├── base.py              # BaseTask (公共逻辑)
│   ├── evaluate.py          # EvaluateTask
│   ├── ablation.py          # AblationTask
│   └── sensitivity.py       # SensitivityTask
├── services/                # 服务层
│   ├── wandb_service.py
│   ├── tmux_service.py
│   └── command_builder.py
└── utils/
    ├── exceptions.py        # 类型化异常 (WorkflowError/SweepError/…)
    ├── helpers.py           # 邮件发送 + 配置校验
    ├── visualization.py     # 论文风格绘图
    └── email_templates.py   # 邮件模板
```

## 文档

- [USAGE.md](USAGE.md) — 详细使用指南 + 4 种正式 CLI 脚本
- [memory-bank/](memory-bank/) — 架构、决策、工作流规范

## W&B Group Naming

| 任务 | 格式 | 示例 |
|------|------|------|
| evaluate | `eval/{sweep_id}/top-{rank}` | `eval/abc123/top-1` |
| ablation | `ablation/{sweep_id}/[r{rank}/]{component}` | `ablation/abc123/r2/no_lin1_bn` |
| sensitivity | `sensitivity/{sweep_id}/[r{rank}/]{study}` | `sensitivity/abc123/r2/width_sensitivity` |
