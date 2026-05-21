# Lightning-Hydra Workflow Usage Guide

## Entry Points

```bash
python src/train.py experiment=example       # Single train + test
python src/eval.py ckpt_path=path/to/ckpt    # Evaluate an existing checkpoint
python src/workflow.py                       # Full pipeline (default)
```

## Scenario 1: Full Pipeline (Default)

Sweep → evaluate best params → email notification.

```bash
# All-in-one
python src/workflow.py
# or via script (defaults to pipeline mode)
bash scripts/workflow.sh
```

What happens:
1. Create W&B sweep (bayesian, 4 runs max)
2. Wait for sweep agents to finish
3. Publish immutable source artifact `code_sweep_<sweep_id>` to W&B
4. Find best run by `val/acc_best`
5. Retrain with different seeds using best params in `/tmp/lightning-runs/...` sandbox
6. Aggregate `test/acc` and `test/loss` → `logs/final_reports/`
7. Send email notification (config: `notification.enabled: true`)

**实测输出:**
```
[INFO] Workflow Orchestrator Initialized
[INFO] ▶️ Starting a new Sweep...
[INFO] ✅ Sweep created successfully! ID: rd2mmn38
[INFO] Monitor Sweep: tmux attach -t mnist_workflow_rd2mmn38
[INFO] ✅ Report saved to logs/final_reports/optimized_results_rd2mmn38.json
[INFO] 🎉 All pipeline tasks finished successfully!
✅ 完成! 耗时 377s
```

## Scenario 2: Reuse Existing Sweep

Skip hyperparameter search, go straight to multi-seed evaluation.

```bash
python src/workflow.py workflow.target_sweep_id=wxsxa50o
# or
bash scripts/workflow.sh pipeline wxsxa50o
# only evaluate
bash scripts/workflow.sh evaluate wxsxa50o
```

Useful when:
- Sweep already finished, only need best-params evaluation
- Resuming after pipeline crash
- Different evaluation settings (more seeds, different metrics)

Variations:
```bash
# More seeds for better statistics
python src/workflow.py workflow.target_sweep_id=wxsxa50o workflow.evaluate_task.num_seeds=5

# Top-2 best runs, each evaluated independently
python src/workflow.py workflow.target_sweep_id=wxsxa50o workflow.evaluate_task.top_n=2

# Resume mode: skip retraining, aggregate existing wandb runs only
python src/workflow.py workflow.target_sweep_id=wxsxa50o workflow.evaluate_task.mode=resume

# Dry-run: preview all commands without executing
python src/workflow.py workflow.target_sweep_id=wxsxa50o workflow.dry_run=true
```

**evaluate 实测输出 (rerun 模式, top_n=1, num_seeds=2):**
```json
{
    "sweep_id": "uppb79dk",
    "top_n": 1,
    "evaluation_summary": {
        "top-1": {
            "best_run_name": "zany-sweep-2",
            "best_run_config": "{\"data.batch_size\": \"128\", \"model.optimizer.lr\": \"0.005\"}",
            "metrics": {
                "test/acc": {"mean": 0.71, "std": 0.02, "count": 2, "values": [0.71, 0.69]},
                "test/loss": {"mean": 1.84, "std": 0.03, "count": 2, "values": [1.84, 1.87]}
            },
            "checkpoints": ["/abs/path/epoch_000.ckpt"]
        }
    }
}
```

> **💡 resume 模式说明**: `workflow.evaluate_task.mode=resume` 不重训，直接从 wandb 聚合已有 runs 的 metrics。适合多次分析同一 sweep 结果。

## Scenario 3: Ablation Mode (Standalone)

Dedicated ablation experiment mode with automatic evaluate result retrieval and comparison email.

```bash
# Via script
bash scripts/workflow.sh ablation wxsxa50o

# Via python (使用 workflow=ablation 加载消融专用配置)
python src/workflow.py workflow=ablation workflow.target_sweep_id=wxsxa50o
```

What happens:
1. **Check sweep exists** → not found → error, exit
2. **Get evaluate results** → if none, run evaluate first (**no email sent**)
3. For each component in `ablation_task.components`:
   - Inherit best params + override component → train+evaluate (3 seeds) in sandbox when `snapshot.enabled=true`
4. **Send ablation email**: full model vs each ablation variant comparison table + relative drop%, plus the ablation report directory

Configuration (`configs/workflow/ablation.yaml`):
```yaml
# 继承 mnist 基础配置, 添加消融组件
defaults:
  - mnist

task_name: ablation
target_sweep_id: null

ablation_task:
  num_seeds: 3
  components:
    - name: "no_lin1_bn"
      overrides:
        model.net.lin1_size: 32
    - name: "no_lin2_bn"
      overrides:
        model.net.lin2_size: 32
```

**实测输出 (num_seeds=1, sweep_id=wxsxa50o):**
```
🧪 ▶️ Launching Ablation Task...
✅ Found existing evaluate report: logs/final_reports/optimized_results_wxsxa50o.json
========== 🧪 Ablation: [no_lin1_bn] ==========
🚀 Launching 1 workers to handle 1 evaluation runs.
========== 🧪 Ablation: [no_lin2_bn] ==========
🚀 Launching 1 workers to handle 1 evaluation runs.
```

消融报告 `logs/final_reports/ablation_wxsxa50o/no_lin1_bn.json`:
```json
{
    "name": "no_lin1_bn",
    "overrides": { "model.net.lin1_size": 32 },
    "metrics": {
        "test/acc": { "mean": 0.557, "std": 0.0, "values": [0.557] },
        "test/loss": { "mean": 1.449, "std": 0.0, "values": [1.449] }
    }
}
```

> **💡 注意**: `num_seeds=1` 时报告中的 `std` 会归一为 `0.0`，避免邮件/JSON/CSV 出现 `NaN`。正式实验仍建议 `num_seeds >= 3`。

## Scenario 4: Sensitivity Mode (Standalone)

Parameter sensitivity analysis with 1D/2D grid, paper-style plots, and PDF-attached email.

```bash
# Via script
bash scripts/workflow.sh sensitivity wxsxa50o

# Via python (使用 workflow=sensitivity 加载敏感性专用配置)
python src/workflow.py workflow=sensitivity workflow.target_sweep_id=wxsxa50o
```

What happens:
1. **Check sweep exists** → not found → error, exit
2. **Get evaluate results** → if none, run evaluate first (**no email sent**)
3. For each parameter combination in `sensitivity_task.param_grid`:
   - Inherit best params + override → train+evaluate (3 seeds) in sandbox when `snapshot.enabled=true`
4. **Plot**: 1D (line + error bar) or 2D (heatmap), paper-style (serif, 3.5in wide)
5. **Save**: PNG + PDF (dpi=300) in `logs/final_reports/`
6. **Send email**: best params metrics table + embedded plot + PDF attachment
7. W&B Local 有最终一致性延迟时，结果收集会强制刷新并短重试，避免刚删除/重建 group 后误判无结果

Configuration (`configs/workflow/sensitivity.yaml`):
```yaml
# 继承 mnist 基础配置, 添加敏感性参数网格
defaults:
  - mnist

task_name: sensitivity
target_sweep_id: null

sensitivity_task:
  primary_metric: "test/acc"
  # 1D example:
  # param_grid:
  #   optimizer.lr: [0.0001, 0.001, 0.01]
  # 2D example:
  param_grid:
    model.net.lin1_size: [32, 64, 128]
    model.net.lin2_size: [32, 64, 128]
  axis_labels:
    model.net.lin1_size: "First Layer Size"
    model.net.lin2_size: "Second Layer Size"
    optimizer.lr: "Learning Rate"
  figure_width: 3.5           # Single-column width for 2-column papers
  figure_dpi: 300
```

**实测输出 (num_seeds=1, sweep_id=wxsxa50o, 2D grid):**
```
📐 ▶️ Launching Sensitivity Task...
📐 Sensitivity Grid: 9 combinations (2D)
  -> [Grid 1/9] sensitivity_lin1_size_32_lin2_size_32
  -> [Grid 2/9] sensitivity_lin1_size_32_lin2_size_64
  ...
  -> [Grid 9/9] sensitivity_lin1_size_128_lin2_size_128
📊 2D sensitivity plot saved: logs/final_reports/sensitivity_wxsxa50o/sensitivity_2d_...png
```

生成文件:
```
logs/final_reports/sensitivity_wxsxa50o/
├── sensitivity_2d_model_net_lin1_size_model_net_lin2_size.png   (65KB)
└── sensitivity_2d_model_net_lin1_size_model_net_lin2_size.pdf   (15KB)
logs/final_reports/sensitivity_wxsxa50o.json                      # 汇总报告
```

## Scenario 5: Direct Checkpoint Evaluation

```bash
python src/eval.py ckpt_path=logs/mnist-demo/runs/2026-05-16_00-38-41/checkpoints/epoch_000.ckpt
```

## Scenario 6: Notify Mode

Run pipeline with email notification enabled.

```bash
# Via script
bash scripts/workflow.sh notify wxsxa50o

# Via python
python src/workflow.py workflow.target_sweep_id=wxsxa50o workflow.notification.enabled=true
```

**实测输出:**
```
Connecting to SMTP server smtp.163.com:465 to send email...
✅ Email notification sent successfully to 634984192@qq.com
```

## Log Files

| Path | Contents |
|------|----------|
| `logs/workflow/pipeline/runs/<ts>/run.log` | Full pipeline run log |
| `logs/sweeps/status_<sweep_id>.yaml` | Sweep metadata |
| `logs/final_reports/optimized_results_<sweep_id>.json` | Aggregated metrics report (per-rank) |
| `logs/final_reports/optimized_results_<sweep_id>.csv` | Metrics table (mean, std) |
| `logs/final_reports/eval_checkpoints_<sweep_id>.json` | Per-rank checkpoint path mapping |
| `logs/final_reports/ablation_<sweep_id>[_r{rank}]/<component>.json` | Ablation per-component report |
| `logs/final_reports/sensitivity_<sweep_id>[_r{rank}].json` | Sensitivity summary report |
| `logs/final_reports/sensitivity_<sweep_id>[_r{rank}]/*.png` | Sensitivity plots |
| `logs/final_reports/sensitivity_<sweep_id>[_r{rank}]/*.pdf` | Sensitivity plots (300dpi) |
| `logs/mail/{mode}/<project>_<sid>_<ts>.md` | Email markdown (mode=eval/ablation/sensitivity/override) |
| `logs/workflow/evaluate/<session>/gpu_0.log` | Per-GPU evaluation training log |
| `logs/workflow/.pipeline_progress_<sweep_id>` | Resume checkpoint |

## Config Reference

### evaluate_task.timeout_secs (new in v1.1)

```yaml
evaluate_task:
  timeout_secs: 3600  # default: 1h, 0=disable
```

Controls max wall-clock time for multi-seed evaluation. When exceeded:
1. tmux session receives Ctrl+C (SIGINT) → wandb flushes → run marked `failed`
2. tmux `kill-session` force cleanup
3. Workflow proceeds to aggregation with whatever runs completed

Previously evaluate had **no timeout** — a hung training run blocked pipeline forever.
Ablation/sensitivity had `timeout_secs` already; evaluate now consistent.

### sweep_task.max_grid_combinations (new in v1.1)

```yaml
sweep_task:
  max_grid_combinations: 100  # default
```

Controls max Cartesian product size for `type: grid` tasks.
Previously hardcoded at 100. Overridable via CLI:
```bash
python src/workflow.py workflow.sweep_task.max_grid_combinations=200
```

### Typed Exceptions (new in v1.1)

`sys.exit(1)` calls replaced with typed exceptions for better error handling:

| Exception | Raised When |
|-----------|-------------|
| `WorkflowError` | Generic pipeline failure |
| `ConfigError` | Invalid/missing config |
| `SweepError` | Sweep creation/lookup failure |
| `EvaluationError` | Aggregation fails |
| `SessionError` | tmux session conflict |

Benefits:
- `task_wrapper` finally block runs cleanup (wandb.finish)
- Callers can catch specific exception types
- Hydra multirun: single trial failure doesn't kill process tree

## Resume After Crash

Re-run the same command. Completed stages detected from
`logs/workflow/.pipeline_progress_<sweep_id>` and skipped.
To force a full re-run:
```bash
rm logs/workflow/.pipeline_progress_<SWEEP_ID>
```

## Workflow Script (`scripts/workflow.sh`)

The workflow script provides 7 run modes with mamba environment activation, `.env` loading, and timeout support.

### Quick Start

```bash
# 完整流程 (sweep + evaluate + ablation + sensitivity)
bash scripts/workflow.sh pipeline

# 仅超参搜索
bash scripts/workflow.sh sweep

# 仅多种子评估 (需指定 sweep_id)
bash scripts/workflow.sh evaluate <sweep_id>

# 消融实验模式
bash scripts/workflow.sh ablation <sweep_id>

# 参数敏感性模式
bash scripts/workflow.sh sensitivity <sweep_id>

# 预览模式 (打印命令不执行)
bash scripts/workflow.sh dry-run

# 邮件通知模式
bash scripts/workflow.sh notify
```

### Immutable Sandbox Isolation

默认启用 `workflow.snapshot.enabled=true`。新 sweep 完成后会发布 `code_sweep_<sweep_id>` W&B artifact；`evaluate` / `ablation` / `sensitivity` 会从该 artifact 在 `/tmp/lightning-runs/<project>/sweep_<id>/...` 重建代码沙盒，并在沙盒内执行训练 payload。

```bash
# 正常沙盒运行
bash scripts/workflow.sh pipeline
bash scripts/workflow.sh evaluate <sweep_id>

# 旧 sweep 没有 source artifact 时，可显式关闭沙盒，直接在当前工作区执行
bash scripts/workflow.sh evaluate <sweep_id> "workflow.snapshot.enabled=false"

# 超时设置 (默认 300s, 0=无限制)
TIMEOUT_SECS=600 bash scripts/workflow.sh sensitivity <sweep_id>
```

**Sandbox 流程**:
1. sweep → W&B 创建 sweep → tmux agent 完成 → 发布 `code_sweep_<sweep_id>` artifact
2. eval/ablation/sensitivity → 下载 source artifact → 解压到 `/tmp/lightning-runs/...`
3. 沙盒软链接 `.env`、`data/` 等运行资源，任务结束后自动清理

### All Modes

| Mode | Description | Requires sweep_id? |
|------|-------------|-------------------|
| `pipeline` (default) | Full pipeline: auto-sweep if no ID + all pipeline stages | No |
| `sweep` | Stage 1 only: hyperparameter search | No |
| `evaluate` | Stage 2 only: multi-seed evaluation | **Yes** |
| `ablation` | Ablation mode: auto-get evaluate results, comparison email | **Yes** |
| `sensitivity` | Sensitivity mode: 1D/2D grid + paper plots + PDF email | **Yes** |
| `dry-run` | Preview mode: print commands without executing | Optional |
| `notify` | Run with email notification enabled | Optional |

### Timeout Control

```bash
# Default: 300s (5min) timeout
bash scripts/workflow.sh sweep

# Custom timeout
TIMEOUT_SECS=600 bash scripts/workflow.sh sensitivity abc123xyz

# No timeout (run until completion) — recommended for long-running tasks
TIMEOUT_SECS=0 bash scripts/workflow.sh sweep
TIMEOUT_SECS=0 bash scripts/workflow.sh evaluate abc123xyz
```

> **⚠️ 重要**: 默认 300s 超时对 evaluate/ablation/sensitivity 模式可能不够! 这些模式需要等待 tmux 训练完成。建议用 `TIMEOUT_SECS=0` 或设置足够大的值。
>
> **v1.1 改进**: evaluate 内置 `timeout_secs: 3600` (1h), 超时自动 tmux kill + SIGINT 让 wandb 标记 failed。
> 设 `timeout_secs=0` 禁用超时。`max_grid_combinations` 配置化 (默认100), 可 CLI 覆盖。

### Usage Examples

```bash
# Evaluate an existing sweep
bash scripts/workflow.sh evaluate abc123xyz

# Ablation with existing sweep (auto-loads ablation config)
bash scripts/workflow.sh ablation abc123xyz

# Sensitivity with existing sweep (auto-loads sensitivity config)
bash scripts/workflow.sh sensitivity abc123xyz

# Enable email notification
bash scripts/workflow.sh notify abc123xyz

# Custom GPU devices (默认从 CUDA_VISIBLE_DEVICES 自动推断)
DEVICES="[0,1]" bash scripts/workflow.sh pipeline   # 显式指定
# 或仅在 .env 中设置:
#   CUDA_VISIBLE_DEVICES=0,1,2
# 脚本自动推断为 devices=[0,1,2]

# Custom conda environment (default myenv)
PYENV=myenv bash scripts/workflow.sh pipeline

# No timeout for long runs
TIMEOUT_SECS=0 bash scripts/workflow.sh evaluate abc123xyz

# Override config via extra args
bash scripts/workflow.sh evaluate abc123xyz 'workflow.evaluate_task.num_seeds=5'
bash scripts/workflow.sh ablation abc123xyz 'workflow.ablation_task.num_seeds=2'
```

### Email Content

When email notification is enabled, the email includes:
- **Sweep ID** — for tracking and resuming
- **Sweep URL** — clickable link to W&B sweep page
- **Run URL** — clickable link to W&B run overview page
- **Run Config (rank N best)** — optimal hyperparameters in JSON format
- **Test Metrics** — mean ± std across seeds
- **Baseline Comparison** — improvement % (if baseline configured)
- **Report Files** — absolute paths to JSON/CSV results
- **Log Directory** — absolute path to run logs

For **ablation mode**, additionally:
- **Comparison Table** — full model vs each ablation variant
- **Relative Drop %** — per-ablation performance drop
- **Reproduction Scripts** — per-component commands with merged seeds (e.g. `seed=[42,43,44]`)
- **Sweep URL** — clickable link to W&B sweep page
- **Group URLs** — clickable links to each ablation variant's W&B group page

For **sensitivity mode**, additionally:
- **Best Params Metrics** — table of optimal parameter results
- **Parameter Grid Results** — table of all grid combinations
- **Embedded Plot** — 1D line chart or 2D heatmap
- **PDF Attachment** — publication-quality figure (dpi=300)
- **Reproduction Scripts** — per-study commands with merged seeds
- **Sweep URL** — clickable link to W&B sweep page
- **Group URLs** — clickable links to each sensitivity study's W&B group page

### Equivalent Python Commands

```bash
# pipeline (default)
python src/workflow.py workflow.sweep_task.conda_env=myenv

# pipeline with existing sweep
python src/workflow.py workflow.target_sweep_id=abc123xyz workflow.sweep_task.conda_env=myenv

# sweep
python src/workflow.py workflow.task_name=sweep workflow.sweep_task.conda_env=myenv

# ablation — 使用 workflow=ablation 加载消融专用配置
python src/workflow.py workflow=ablation workflow.target_sweep_id=abc123xyz

# sensitivity — 使用 workflow=sensitivity 加载敏感性专用配置
python src/workflow.py workflow=sensitivity workflow.target_sweep_id=abc123xyz

# evaluate with resume mode (aggregate existing runs, no retraining)
python src/workflow.py workflow.target_sweep_id=abc123xyz workflow.evaluate_task.mode=resume

# dry-run
python src/workflow.py workflow.dry_run=true workflow.sweep_task.conda_env=myenv
```

## Environment

Set in `.env`:
```bash
WANDB_API_KEY=local-xxx
WANDB_BASE_URL=http://your.server:port
PYENV=myenv       # conda/mamba environment name
CUDA_VISIBLE_DEVICES=0,1,2   # GPU 设备列表，自动转为 workflow.devices
SMTP_HOST=smtp.example.com  # (optional, for email notifications)
SMTP_PASSWORD=xxx           # (optional)
USER_EMAIL_ADDRESS=user@example.com
```

> **💡 CUDA_VISIBLE_DEVICES 说明**: `workflow.sh` 在加载 `.env` 后自动将 `CUDA_VISIBLE_DEVICES=0,1,2` 解析为 `DEVICES=[0,1,2]` 传递给 Python。无需再手动写 `DEVICES="[0,1]"`

## Troubleshooting

### `evaluation_summary: {}` (empty metrics)

wandb.Api() 缓存导致。`WandbService.get_runs_by_group` 使用专用 `_runs_api` 实例避开 sweep.runs 缓存。
如果仍出现:
```bash
# 重新运行 evaluate (resume 模式, 不重训)
bash scripts/workflow.sh evaluate <sweep_id> 'workflow.evaluate_task.mode=resume'
# 或强制刷新 API 实例
python -c "from src.services.wandb_service import WandbService; s=WandbService('e','p'); s.get_runs_by_group('g', force_refresh=True)"
```

### `Key 'workflow' is not in struct`

Hydra struct mode 不允许新增 key。ablation/sensitivity 配置已添加 `target_sweep_id: null` 占位。如果自定义配置遇到此问题:
```bash
# 用 + 前缀添加新 key
python src/workflow.py +workflow.new_key=value
```

### `PosixPath object has no attribute 'format'`

OmegaConf 将路径字符串解析为 PosixPath。已在代码中统一用 `str(...)` 转换后再 `.format()`。

## Debug vs Full 配置

| 配置 | 实验 | Sweep | 说明 |
|------|------|-------|------|
| `workflow=mnist` (默认) | `experiment=example` | `mnist_sweep.yaml` | debug: limit_*=3, max_epochs=1, run_cap=4 |
| `workflow=mnist_full` | `experiment=mnist_full` | `mnist_sweep_full.yaml` | 正式: 无limit, max_epochs=10, run_cap=20 |

**_full 配置文件**:
- `configs/experiment/mnist_full.yaml` — 完整训练 (max_epochs=10, 无 limit_*)
- `configs/sweep/mnist_sweep_full.yaml` — 充分探索 (run_cap=20, experiment=mnist_full)
- `configs/workflow/mnist_full.yaml` — 工作流 full (继承 mnist, 覆盖 sweep/eval/ablation/sensitivity)

> **💡 切换方式**: 在任何命令后加 `"workflow=mnist_full"` 即可从 debug 切换到正式实验。ablation/sensitivity 同理。

## Production CLI Scripts

> **正式实验配置**: `workflow=mnist_full` → `experiment=mnist_full`, `run_cap=20`, `num_seeds=5`

以下 4 种命令可直接复制到终端运行正式实验:

### 1. New Sweep + Evaluate (全流程)

```bash
# 新 sweep → top-2 evaluate (5 seeds each) → email
TIMEOUT_SECS=0 bash scripts/workflow.sh pipeline "" "workflow=mnist_full"
```

### 2. Existing Sweep + Evaluate (复用 sweep)

```bash
# 已有 sweep_id → top-2 evaluate (5 seeds each) → email (per-rank 分段展示)
TIMEOUT_SECS=0 bash scripts/workflow.sh pipeline <SWEEP_ID> "workflow=mnist_full"
```

### 3. Ablation (消融实验)

```bash
# 已有 sweep_id → eval_rank=2 (2nd best) → 消融实验 (5 seeds) → 对比邮件
TIMEOUT_SECS=0 bash scripts/workflow.sh ablation <SWEEP_ID> \
  "workflow=mnist_full" "workflow.evaluate_task.eval_rank=2"
```

### 4. Sensitivity (参数敏感性)

```bash
# 已有 sweep_id → eval_rank=2 (2nd best) → 2D width + 1D lr 敏感性 (5 seeds) → 绘图 + 邮件
TIMEOUT_SECS=0 bash scripts/workflow.sh sensitivity <SWEEP_ID> \
  "workflow=mnist_full" "workflow.evaluate_task.eval_rank=2"
```

> **💡 说明**:
> - `workflow=mnist_full` 自动切换: `experiment=mnist_full` + `run_cap=20` + `num_seeds=5`
> - `TIMEOUT_SECS=0` 禁止超时 kill, 正式实验建议使用
> - `num_seeds=5` 确保统计显著性 (seed=[42,43,44,45,46])
> - `eval_rank=2` 使用第2优参数 (默认 eval_rank=1 即 best), group 加 `r2/` 前缀, 报告加 `_r2` 后缀
> - 替换 `<SWEEP_ID>` 为实际 sweep ID (如 `en1zkicr`)
> - checkpoint 只保留 epoch_*.ckpt (early stopping), 排除 last.ckpt
> - eval checkpoint 映射自动保存到 `logs/final_reports/eval_checkpoints_<sweep_id>.json`

## Real Debug Notes (2026-05-21)

真实运行已覆盖 `scripts/workflow.sh` 的四种典型模式：

| Mode | Result | Email |
|------|--------|-------|
| `sweep` | 创建 sweep、等待 tmux agent、发布 `code_sweep_<id>` artifact | 不发送邮件（设计如此） |
| `evaluate` | top-N best run 多 seed 重跑、保存 JSON/CSV | `logs/mail/eval/` |
| `ablation` | 展开 `no_lin1_bn` / `no_lin2_bn`，保存每组件报告 | `logs/mail/ablation/` |
| `sensitivity` | 展开 `width_sensitivity` 2D grid 与 `lr_sensitivity` 1D grid，保存 JSON/PNG/PDF | `logs/mail/sensitivity/` |

修复过的真实问题：

- `workflow.snapshot.enabled=false` 现在会被任务层尊重，旧 sweep 缺少 source artifact 时可显式关闭沙盒。
- W&B stale/deleted runs 会被跳过，避免 Eval 统计旧 run 或 Sensitivity 误判无结果。
- Sensitivity 结果收集会 force refresh 并重试，适配 W&B Local group 删除/重建后的短暂延迟。
- Ablation/Sensitivity 邮件分别展示真实报告目录/JSON，不再误用 Eval CSV 字段。
- 单 seed `std` 归一为 `0.0`，邮件 markdown 不再出现孤立的 `**`。
