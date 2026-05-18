# src/tasks/evaluate.py
"""Stage 2: 多随机种子评估任务。"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from wandb.apis.public import Run, Sweep

from src.services.command_builder import CommandBuilder
from src.services.tmux_service import TmuxService
from src.services.wandb_service import WandbService
from src.tasks.base import BaseTask
from src.tasks.shared import is_dry_run, safe_get_metric_scores
from src.utils import RankedLogger, WorkflowError
from src.utils.helpers import build_wandb_sweep_url, build_wandb_run_url, format_reproduction_script

log = RankedLogger(__name__, rank_zero_only=True)


class EvaluateTask(BaseTask):
    """Stage 2: 取 top_n 最优参数 → 多种子重跑 → 聚合报告。"""

    def __init__(
        self,
        cfg: DictConfig,
        wandb_service: WandbService,
        tmux_service: TmuxService,
        command_builder: CommandBuilder,
    ):
        super().__init__(cfg, wandb_service, tmux_service, command_builder)

    def run(self, sweep_id: Optional[str] = None, skip_email: bool = False):
        """执行评估任务。

        Args:
            skip_email: True 时不发邮件 (被消融/敏感性前置调用时用)
        """
        log.info("▶️ Stage 2: Launching Evaluation Task...")

        sweep_id = self.resolve_sweep_id(sweep_id, "evaluate_task")

        if is_dry_run() and (not sweep_id or sweep_id.startswith("dry-run-")):
            log.info("[DRY-RUN] Mock sweep — skipping wandb lookup, eval, aggregation.")
            return

        log.info("--- [Phase 2.1] Waiting for Sweep to finish on W&B server ---")
        sweep = self.get_sweep(
            sweep_id,
            wait=self.cfg.evaluate_task.wait_for_sweep_finish,
            wait_interval=self.cfg.evaluate_task.wait_interval_seconds,
        )

        log.info("--- [Phase 2.2] Finding top_n best runs and extracting hyperparameters ---")
        self.description = sweep.config.get("description", "N/A")
        self.task = f"{sweep.project} {sweep.id[:6]}"

        # top_n support (workflow_rules: default 2, max 3)
        top_n = self.cfg.evaluate_task.get("top_n", 2)
        if top_n > 3:
            raise WorkflowError(f"top_n={top_n} exceeds maximum allowed value of 3")

        top_runs = self.wandb_service.find_top_n_runs(
            sweep, self.cfg.evaluate_task.optimized_metric, top_n
        )
        if not top_runs:
            raise WorkflowError("Could not determine the best runs from sweep.")

        log.info(f"🏆 Found {len(top_runs)} top runs: {[r.name for r in top_runs]}")

        # 记录 eval 开始时间 (用于 rerun 模式过滤旧 checkpoint)
        self._eval_start_time = time.time()

        # Evaluate each top run (sequential, track time windows for checkpoint association)
        rank_time_windows = []  # [(rank_start_time, rank_end_time), ...]
        eval_group_names = []   # [(best_run, group_name), ...]

        for rank, best_run in enumerate(top_runs):
            rank_start = time.time()
            log.info(f"--- Evaluating top-{rank+1} run: {best_run.name} ---")

            config_overrides = self._extract_config_overrides(best_run)
            log.info(f"📋 Extracted Sweep Overrides: {config_overrides}")

            log.info(f"--- [Phase 2.3] Executing evaluation strategy for top-{rank+1} ---")
            group_name = f"eval/{sweep_id}/top-{rank+1}"
            mode = self.cfg.evaluate_task.mode

            eval_session_name = self.execute_strategy(
                config_overrides, group_name, mode
            )

            if eval_session_name:
                timeout_secs = self.cfg.evaluate_task.get("timeout_secs", 3600)
                if timeout_secs and timeout_secs > 0:
                    self.wait_for_session_with_timeout(
                        eval_session_name,
                        timeout_secs=timeout_secs,
                        interval=self.cfg.evaluate_task.wait_interval_seconds,
                    )
                else:
                    self.wait_for_session(eval_session_name, self.cfg.evaluate_task.wait_interval_seconds)

            rank_end = time.time()
            rank_time_windows.append((rank_start, rank_end))
            eval_group_names.append((best_run, group_name))

        # Aggregate all top_n results into one report
        if is_dry_run():
            log.info("[DRY-RUN] Skipping aggregation — no actual eval runs were launched.")
        else:
            log.info("--- [Phase 2.5] Aggregating results and generating report ---")
            self._aggregate_and_report(sweep, eval_group_names, rank_time_windows, skip_email=skip_email)

    def _aggregate_and_report(
        self, sweep: Sweep, eval_group_names: list, rank_time_windows: list, skip_email: bool = False
    ):
        """Aggregate per-rank results → 统一报告 + 邮件 (per-rank 展开)。"""
        import os

        report_path = Path(self.cfg.evaluate_task.report_path.format(sweep_id=sweep.id))
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # ── 通用数据 ──────────────────────────────────────────────────
        base_url = os.getenv("WANDB_BASE_URL", "")
        sweep_url = build_wandb_sweep_url(
            base_url, self.cfg.wandb.entity, self.cfg.wandb.project, sweep.id
        ) if base_url else ""

        has_target = bool(self.cfg.get("target_sweep_id"))
        mode_label = "Eval" if has_target else "Sweep+Eval"

        baseline_cfg = OmegaConf.to_container(
            self.cfg.evaluate_task.get("baseline", {}), resolve=True
        ) if self.cfg.evaluate_task.get("baseline") else None

        base_args = list(self.cfg.evaluate_task.run_command.base_args)
        num_seeds = self.cfg.evaluate_task.num_seeds
        seed_start = self.cfg.evaluate_task.seed_start
        seeds = list(range(seed_start, seed_start + num_seeds))
        test_metrics = self.cfg.evaluate_task.test_metrics

        # ── Per-rank 数据收集 ─────────────────────────────────────────
        all_final_runs = []
        rank_data = []  # 每个排名的独立数据
        checkpoint_map = {}

        project_log_dir = str(Path(self._full_cfg.paths.log_dir).resolve())

        for rank_idx, (best_run, group_name) in enumerate(eval_group_names):
            rank_label = f"top-{rank_idx + 1}"
            rank_start, rank_end = rank_time_windows[rank_idx]

            # 获取该 group 的 runs
            runs = self.wandb_service.get_runs_by_group(group_name)
            all_final_runs.extend(runs)

            # Per-rank metrics
            rank_metrics = {}
            for metric in test_metrics:
                scores = safe_get_metric_scores(runs, metric)
                if scores:
                    series = pd.Series(scores)
                    rank_metrics[metric] = {
                        "mean": series.mean(),
                        "std": series.std(),
                        "count": len(scores),
                        "values": series.tolist(),
                    }

            # Per-rank config
            config_overrides = self._extract_config_overrides(best_run)
            config_dict = {}
            for item in config_overrides:
                if "=" in item:
                    key, value = item.split("=", 1)
                    config_dict[key] = value
            beautified_config = json.dumps(config_dict, indent=4, ensure_ascii=False)

            # Per-rank metadata
            best_run_metadata = self.extract_run_metadata(best_run) if best_run else {}

            # Per-rank checkpoints (用时间窗口过滤: since_time ~ until_time)
            rank_ckpts = self.collect_checkpoint_paths(
                project_log_dir, since_time=rank_start, until_time=rank_end
            )
            checkpoint_map[rank_label] = rank_ckpts

            # Per-rank reproduction script
            cmd = format_reproduction_script(
                base_args=base_args,
                overrides=config_overrides,
                seeds=seeds,
                group_name=group_name,
            )

            rank_data.append({
                "rank_label": rank_label,
                "best_run_name": best_run.name if best_run else "N/A",
                "run_url": build_wandb_run_url(
                    os.getenv("WANDB_BASE_URL", ""),
                    self.cfg.wandb.entity, self.cfg.wandb.project,
                    best_run.id,
                ) if best_run else "N/A",
                "best_run_config": beautified_config,
                "best_run_metadata": best_run_metadata,
                "metrics": rank_metrics,
                "checkpoint_paths": rank_ckpts,
                "reproduction_script": {"label": rank_label, "command": cmd},
            })

            log.info(f"📊 {rank_label}: metrics={rank_metrics}")

        # ── 保存报告 JSON ─────────────────────────────────────────────
        final_report = {
            "sweep_id": sweep.id,
            "top_n": len(eval_group_names),
            "evaluation_summary": {},
        }
        for rd in rank_data:
            final_report["evaluation_summary"][rd["rank_label"]] = {
                "best_run_name": rd["best_run_name"],
                "best_run_config": rd["best_run_config"],
                "metrics": rd["metrics"],
                "checkpoints": rd["checkpoint_paths"],
            }

        # Baseline comparison (基于 top-1)
        if baseline_cfg and baseline_cfg.get("enabled", False):
            baseline_metric = baseline_cfg.get("metric", "")
            baseline_value = baseline_cfg.get("value", None)
            if baseline_metric and baseline_value is not None:
                top1_metrics = rank_data[0]["metrics"] if rank_data else {}
                if baseline_metric in top1_metrics:
                    current_mean = top1_metrics[baseline_metric]["mean"]
                    improvement = current_mean - baseline_value
                    improvement_pct = (improvement / baseline_value) * 100 if baseline_value != 0 else 0.0
                    final_report["baseline_comparison"] = {
                        "metric": baseline_metric,
                        "baseline_value": baseline_value,
                        "current_mean": current_mean,
                        "improvement": round(improvement, 6),
                        "improvement_pct": round(improvement_pct, 2),
                    }

        with open(report_path, "w") as f:
            json.dump(final_report, f, indent=4)
        log.info(f"✅ Report saved to {report_path}")

        # ── 保存 eval checkpoints JSON ────────────────────────────────
        self.save_eval_checkpoints(
            sweep.id, checkpoint_map,
            str(report_path.parent.resolve())
        )

        # ── 发送邮件 ─────────────────────────────────────────────────
        if not skip_email and rank_data:
            from src.utils.helpers import send_eval_email
            send_eval_email(
                cfg=self.cfg.notification,
                sweep_id=sweep.id,
                sweep_url=sweep_url,
                sweep_description=getattr(self, "description", "N/A"),
                mode_label=mode_label,
                task_desc=self.task,
                rank_data=rank_data,
                baseline_info=final_report.get("baseline_comparison"),
                report_json=str(report_path.resolve()),
                report_csv=str(report_path.with_suffix(".csv").resolve()),
                log_dir=str(report_path.parent.resolve()),
            )

        # 保存 CSV (向后兼容: 所有 runs 合并)
        if all_final_runs:
            data = []
            for metric in test_metrics:
                scores = safe_get_metric_scores(all_final_runs, metric)
                if scores:
                    series = pd.Series(scores)
                    data.append([metric, series.mean(), series.std(), series.tolist()])
            if data:
                metrics_df = pd.DataFrame(data, columns=["metric", "mean", "std", "values"])
                metrics_df.to_csv(report_path.with_suffix(".csv"), index=False)

    @staticmethod
    def _extract_config_overrides(best_run: Run) -> list:
        """从 best_run.config 提取 Hydra override 列表。"""
        config_overrides = []
        for key, value in best_run.config.items():
            if "." in key:
                val_str = f'"{value}"' if isinstance(value, list) else str(value)
                config_overrides.append(f"{key}={val_str}")
        return config_overrides