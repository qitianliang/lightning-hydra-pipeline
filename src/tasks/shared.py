# src/tasks/shared.py
"""共享工具函数与评估策略 — 从 workflow.py 抽取, 供各 Task 复用。"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.services.command_builder import CommandBuilder
from src.services.tmux_service import TmuxService
from src.services.wandb_service import WandbService
from src.utils import EvaluationError, RankedLogger, SessionError, send_email_with_dataframe

log = RankedLogger(__name__, rank_zero_only=True)


# ── 指标获取 (带重试) ──────────────────────────────────────────────────────
def _refresh_run(run) -> object:
    """从 wandb API 重新获取 run 以刷新 summary。"""
    try:
        import wandb
        api = wandb.Api()
        # W&B local server: run.path may be list ['entity','project','id']
        # W&B cloud: run.path is string 'entity/project/id'
        path = run.path
        if isinstance(path, list):
            path = "/".join(str(p) for p in path)
        fresh_run = api.run(path)
        return fresh_run
    except Exception as e:
        log.warning(f"Failed to refresh run {run.id}: {e}")
        return run


def safe_get_metric_scores(
    final_runs: list, metric: str, retries: int = 5, delay: float = 3.0
) -> list:
    """从 wandb runs 获取指标值, 带重试。每次重试刷新 run 对象。"""
    scores = []
    for run in final_runs:
        current_run = run
        for attempt in range(retries):
            try:
                val = current_run.summary.get(metric)
                if val is not None:
                    scores.append(val)
                    break
                if attempt < retries - 1:
                    log.warning(f"Retry {attempt+1}/{retries} for run {current_run.id} metric '{metric}': got None, refreshing run...")
                    time.sleep(delay * (attempt + 1))
                    current_run = _refresh_run(current_run)
                else:
                    log.warning(f"Giving up on run {current_run.id} metric '{metric}' after {retries} retries")
            except (ValueError, AttributeError) as e:
                if attempt < retries - 1:
                    log.warning(f"Retry {attempt+1}/{retries} for run {current_run.id}: {e}")
                    time.sleep(delay * (attempt + 1))
                    current_run = _refresh_run(current_run)
                else:
                    log.warning(f"Skipping run {current_run.id}: failed after {retries} retries: {e}")
    return scores


# ── 聚合报告构建 ────────────────────────────────────────────────────────────
def build_aggregation_report(
    final_runs: list,
    report_path: Path,
    test_metrics: list,
    extra_fields: dict,
    notification_cfg: dict,
    task_desc: str,
    best_run_config: str,
    baseline_cfg: Optional[dict] = None,
    sweep_description: str = "N/A",
    skip_email: bool = False,
    reproduction_scripts: list = None,
    sweep_url: str = None,
    mode_label: str = "Eval",
    best_run_metadata: dict = None,
    checkpoint_paths: list = None,
) -> dict:
    """通用报告构建器 + 邮件通知。

    Args:
        skip_email: True 时跳过邮件 (消融/敏感性前置 evaluate 用)
        reproduction_scripts: [{"label": str, "command": str}, ...] 复现脚本列表
        sweep_url: wandb sweep URL
        mode_label: 运行模式标签 (Sweep+Eval / Eval / Ablation / Sensitivity)
        best_run_metadata: {"host": str, "created_at": str, "duration": str}
        checkpoint_paths: model checkpoint 绝对路径列表
    """
    if not final_runs:
        raise EvaluationError("No runs found for aggregation. Nothing to report.")

    final_report = {
        "evaluation_summary": {},
        **extra_fields,
    }
    data = []
    for metric in test_metrics:
        scores = safe_get_metric_scores(final_runs, metric)
        if scores:
            series = pd.Series(scores)
            final_report["evaluation_summary"][metric] = {
                "mean": series.mean(),
                "std": series.std(),
                "count": len(scores),
                "values": series.tolist(),
            }
            data.append([metric, series.mean(), series.std(), series.tolist()])

    # Baseline comparison
    if baseline_cfg and baseline_cfg.get("enabled", False):
        baseline_metric = baseline_cfg.get("metric", "")
        baseline_value = baseline_cfg.get("value", None)
        if baseline_metric and baseline_value is not None:
            eval_summary = final_report.get("evaluation_summary", {})
            if baseline_metric in eval_summary:
                current_mean = eval_summary[baseline_metric]["mean"]
                improvement = current_mean - baseline_value
                improvement_pct = (improvement / baseline_value) * 100 if baseline_value != 0 else 0.0
                final_report["baseline_comparison"] = {
                    "metric": baseline_metric,
                    "baseline_value": baseline_value,
                    "current_mean": current_mean,
                    "improvement": round(improvement, 6),
                    "improvement_pct": round(improvement_pct, 2),
                }

    metrics_data = pd.DataFrame(data, columns=["metric", "mean", "std", "values"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=4)

    log.info("--- Final Report ---")
    log.info(f"\n{json.dumps(final_report, indent=4)}")
    log.info(f"✅ Report saved to {report_path}")

    if not metrics_data.empty:
        display = metrics_data.copy()
        display["mean"] = display["mean"].apply(lambda x: f"{x:.4g}" if pd.notnull(x) else "")
        display["std"] = display["std"].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
        log.info(f'\n{display[["metric", "mean", "std"]]}')
        metrics_data.to_csv(report_path.with_suffix(".csv"), index=False)
        log.info(f"✅ Metrics data saved to {report_path.with_suffix('.csv')}")

        if not skip_email:
            email_subject = f"✅ {mode_label} | {task_desc}"
            if baseline_cfg and baseline_cfg.get("enabled", False):
                bc = final_report.get("baseline_comparison")
                if bc:
                    email_subject = f"✅ {task_desc} | {bc['metric']} {bc['improvement_pct']:+.2f}% vs baseline"

            sweep_id = extra_fields.get("sweep_id", "N/A")
            report_json_abs = str(report_path.resolve()) if report_path else "N/A"
            report_csv_abs = str(report_path.with_suffix(".csv").resolve()) if report_path else "N/A"
            log_dir_abs = str(report_path.parent.resolve()) if report_path else "N/A"

            workflow_info = {
                "sweep_id": sweep_id,
                "sweep_description": sweep_description,
                "best_run_config": best_run_config,
                "report_json_path": report_json_abs,
                "report_csv_path": report_csv_abs,
                "log_dir": log_dir_abs,
                "best_run_host": (best_run_metadata or {}).get("host", "N/A"),
                "best_run_created_at": (best_run_metadata or {}).get("created_at", "N/A"),
                "best_run_duration": (best_run_metadata or {}).get("duration", "N/A"),
                "checkpoint_paths": checkpoint_paths or [],
            }

            try:
                send_email_with_dataframe(
                    notification_cfg, email_subject, body := "", metrics_data,
                    baseline_info=final_report.get("baseline_comparison"),
                    workflow_info=workflow_info,
                    reproduction_scripts=reproduction_scripts,
                    sweep_url=sweep_url,
                )
            except Exception as e:
                log.error(f"Email notification failed: {e}")
    else:
        log.info("No metrics data available.")

    return final_report


# ── 从 JSON report 读取已有 evaluate 结果 ──────────────────────────────────
def _evaluate_report_has_metrics(report: dict) -> bool:
    summary = report.get("evaluation_summary", {}) if isinstance(report, dict) else {}
    for value in summary.values():
        if not isinstance(value, dict):
            continue
        # New format: {"top-1": {"metrics": {"test/acc": ...}}}
        metrics = value.get("metrics")
        if isinstance(metrics, dict) and metrics:
            return True
        # Old format: {"test/acc": {"mean": ..., "std": ...}}
        if "mean" in value:
            return True
    return False


def load_evaluate_report(report_path: str, sweep_id: str) -> Optional[dict]:
    """尝试读取已有且含有效指标的 evaluate 报告。

    Returns:
        报告 dict 或 None
    """
    path = Path(report_path.format(sweep_id=sweep_id))
    if path.exists():
        with open(path) as f:
            report = json.load(f)
        if _evaluate_report_has_metrics(report):
            log.info(f"✅ Found existing evaluate report: {path}")
            return report
        log.warning(f"⚠️ Existing evaluate report has no metrics and will be ignored: {path}")
    return None


# ── Pipeline 进度管理 ──────────────────────────────────────────────────────
def pipeline_progress_file(cfg: DictConfig, sweep_id: str = "") -> Path:
    name = ".pipeline_progress"
    if sweep_id:
        name = f".pipeline_progress_{sweep_id}"
    return Path(cfg.paths.log_dir) / "workflow" / name


def load_pipeline_progress(cfg: DictConfig, sweep_id: str = "") -> set:
    f = pipeline_progress_file(cfg, sweep_id)
    if f.exists():
        done = set(line.strip() for line in f.read_text().splitlines() if line.strip())
        log.info(f"Resume checkpoint: {len(done)} stages already completed: {done}")
        return done
    return set()


def save_pipeline_progress(cfg: DictConfig, stage_name: str, sweep_id: str = ""):
    f = pipeline_progress_file(cfg, sweep_id)
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "a") as fh:
        fh.write(f"{stage_name}\n")
    log.info(f"Checkpoint saved: stage '{stage_name}' marked done.")


# ── 评估策略 (Strategy Pattern) ────────────────────────────────────────────
class EvaluationStrategy:
    """评估策略基类。"""

    def __init__(
        self,
        wandb_service: WandbService,
        tmux_service: TmuxService,
        command_builder: CommandBuilder,
        cfg: DictConfig,
        group_name: str,
        num_seeds: Optional[int] = None,
        seed_start: Optional[int] = None,
        snapshot_dir: str = "",
    ):
        self.wandb_service = wandb_service
        self.tmux_service = tmux_service
        self.command_builder = command_builder
        self.cfg = cfg
        self.group_name = group_name
        self.num_seeds = num_seeds
        self.seed_start = seed_start
        self.snapshot_dir = snapshot_dir

    def execute(self, config_overrides: list) -> str | None:
        raise NotImplementedError


class RerunStrategy(EvaluationStrategy):

    def execute(self, config_overrides: list) -> str | None:
        log.info(f"Executing RerunStrategy: Deleting existing runs in group '{self.group_name}'...")
        if not is_dry_run():
            self.wandb_service.delete_runs_in_group(self.group_name)
        else:
            log.info(f"[DRY-RUN] Would delete runs in group '{self.group_name}'")

        eval_session_name = self.group_name
        if not is_dry_run() and self.tmux_service.session_exists(eval_session_name):
            raise SessionError(f"tmux session '{eval_session_name}' already exists.")

        num_seeds = self.num_seeds if self.num_seeds is not None else self.cfg.evaluate_task.num_seeds
        seed_start = self.seed_start if self.seed_start is not None else self.cfg.evaluate_task.seed_start
        devices = self.cfg.devices

        all_commands = [
            self.command_builder.build_training_run_command(
                overrides=config_overrides,
                seed=seed_start + i,
                group_name=self.group_name,
                cwd=self.snapshot_dir,
            )
            for i in range(num_seeds)
        ]

        commands_per_device = [[] for _ in devices]
        for i, command in enumerate(all_commands):
            commands_per_device[i % len(devices)].append(command)

        if is_dry_run():
            log.info(f"[DRY-RUN] Would launch {len(devices)} worker(s) for {num_seeds} eval runs.")
            for i, device in enumerate(devices):
                for cmd in commands_per_device[i]:
                    log.info(f"  [DRY-RUN] GPU {device}: {cmd}")
            return eval_session_name

        worker_defs = []
        for i, device in enumerate(devices):
            worker_commands = commands_per_device[i]
            if not worker_commands:
                continue
            worker_script = self.command_builder.build_evaluation_worker_script(
                worker_commands, device, eval_session_name=eval_session_name
            )
            full_worker_command = f"CUDA_VISIBLE_DEVICES={device} bash -c '{worker_script}'"
            worker_defs.append({"device": device, "command": full_worker_command})

        log.info(f"🚀 Launching {len(worker_defs)} workers to handle {num_seeds} evaluation runs.")
        self.tmux_service.create_workers_session(eval_session_name, worker_defs)
        return eval_session_name


class ResumeStrategy(EvaluationStrategy):

    def execute(self, config_overrides: list) -> str | None:
        log.info(f"Executing ResumeStrategy: Checking for existing runs in group '{self.group_name}'...")
        existing_runs = self.wandb_service.get_runs_by_group(self.group_name)
        if existing_runs:
            log.info(f"✅ Found {len(existing_runs)} existing runs. Skipping execution.")
            return None
        log.warning("No existing runs found. Nothing to resume.")
        return None


# Global dry-run flag — set from workflow.py
_DRY_RUN: bool = False


def is_dry_run() -> bool:
    """Return current dry-run state. Use this instead of importing _DRY_RUN directly."""
    return _DRY_RUN
