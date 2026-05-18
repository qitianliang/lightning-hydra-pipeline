# src/tasks/override.py
"""Stage 3: 消融覆盖任务 (也用于 grid 子任务)。"""

import json
import sys
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf
from wandb.apis.public import Run, Sweep

from src.services.command_builder import CommandBuilder
from src.services.tmux_service import TmuxService
from src.services.wandb_service import WandbService
from src.tasks.base import BaseTask
from src.tasks.shared import is_dry_run, build_aggregation_report
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class OverrideTask(BaseTask):
    """Stage 3/4: 继承最优参数 + 强制覆盖 → 多种子评估。"""

    def __init__(
        self,
        cfg: DictConfig,
        wandb_service: WandbService,
        tmux_service: TmuxService,
        command_builder: CommandBuilder,
    ):
        super().__init__(cfg, wandb_service, tmux_service, command_builder)

    def run(self, sweep_id: Optional[str] = None):
        """执行消融/覆盖任务。"""
        log.info(f"▶️ Stage: Launching Override Task [{self.cfg.override_task.name}]...")

        sweep_id = self.resolve_sweep_id(sweep_id, "override_task")

        if is_dry_run() and (not sweep_id or sweep_id.startswith("dry-run-")):
            log.info("[DRY-RUN] Mock sweep — skipping wandb lookup, override eval.")
            return

        log.info("--- [Phase 1] Waiting for Sweep to finish on W&B server ---")
        sweep = self.get_sweep(
            sweep_id,
            wait=self.cfg.override_task.wait_for_sweep_finish,
            wait_interval=self.cfg.override_task.wait_interval_seconds,
        )

        # 获取最优参数
        log.info("--- [Phase 2] Finding best run and extracting hyperparameters ---")
        best_run = self.wandb_service.find_best_run(sweep, self.cfg.override_task.optimized_metric)
        self.description = f"Override Study: {self.cfg.override_task.name} (Base Sweep: {sweep_id})"
        self.task = f"{sweep.project}.{sweep.name}.{self.cfg.override_task.name}"

        # 构建基础 Overrides 并注入消融变量
        best_run_data, config_overrides, config_dict, _ = self.get_best_run_overrides(
            sweep, self.cfg.override_task.optimized_metric
        )
        best_run = best_run_data  # use the one from get_best_run_overrides
        self.best_run_config = config_dict

        log.info(f"✅ Best Run Config: {config_dict}")
        log.info(f"📋 Extracted Sweep Overrides: {config_overrides}")

        override_overrides = self.cfg.override_task.overrides
        override_list = []
        for key, value in override_overrides.items():
            if "." in key:
                val_str = f'"{value}"' if isinstance(value, list) else str(value)
                override_list.append(f"{key}={val_str}")
        log.info(f"🚀 Override Overrides: {override_list}")

        # 合并参数：消融变量放在最后，确保覆盖
        final_overrides = config_overrides + override_list
        final_config_dict = {}
        for item in final_overrides:
            if "=" in item:
                key, value = item.split("=", 1)
                final_config_dict[key] = value

        self.current_run_config = final_config_dict
        log.info(f"🚀 Final Config: {final_config_dict}")

        # 构建包含消融名称的 Group Name
        override_name = self.cfg.override_task.name
        group_name = f"override/{sweep_id}/{override_name}"

        log.info(f"🧪 Override Config: {final_overrides}")
        log.info(f"🔗 Eval Group: {group_name}")
        log.info("--- [Phase 3] Executing evaluation strategy ---")

        mode = self.cfg.override_task.mode
        override_num_seeds = self.cfg.override_task.num_seeds
        override_seed_start = self.cfg.override_task.seed_start

        eval_session_name = self.execute_strategy(
            final_overrides, group_name, mode,
            num_seeds=override_num_seeds, seed_start=override_seed_start,
        )

        self.wait_for_session(eval_session_name, self.cfg.override_task.wait_interval_seconds)

        if is_dry_run():
            log.info("[DRY-RUN] Skipping override aggregation — no actual eval runs were launched.")
        else:
            log.info("--- [Phase 5] Aggregating results and generating report ---")
            self._aggregate_and_report(sweep, best_run, group_name)

    def _aggregate_and_report(self, sweep: Sweep, best_run: Run, group_name: str):
        """Aggregate runs → shared build_aggregation_report()。"""
        log.info(f"Aggregating results for group '{group_name}'...")
        final_runs = self.wandb_service.get_runs_by_group(group_name)
        report_path = Path(self.cfg.override_task.report_path.format(group_name=group_name))
        extra_fields = {
            "sweep_id": sweep.id,
            "best_run_from_sweep": best_run.name,
            "best_run_config": best_run.config,
            "override_config": self.current_run_config,
        }
        override_test_metrics = self.cfg.override_task.get(
            "test_metrics", self.cfg.evaluate_task.test_metrics
        )
        build_aggregation_report(
            final_runs=final_runs,
            report_path=report_path,
            test_metrics=override_test_metrics,
            extra_fields=extra_fields,
            notification_cfg=self.cfg.notification,
            task_desc=self.task,
            best_run_config=str(self.current_run_config),
            sweep_description=getattr(self, "description", "N/A"),
        )