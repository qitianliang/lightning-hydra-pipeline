# src/utils/cleanup_context.py

import logging
import shutil
from contextlib import ContextDecorator
from pathlib import Path
from typing import Optional

from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class WandbCleanupHandler(ContextDecorator):
    """A robust, DDP-safe, and sweep-safe context manager for handling wandb cleanup.

    1. Copies the specific run-* directory to the Hydra output folder.
    2. If it's a sweep run, it also copies the specific `config-*.yaml`.

    This uses `copytree` instead of `move` to avoid interfering with the live
    wandb service, which is critical for complex scenarios like wandb sweep.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.trainer: Optional[Trainer] = None
        log.info("WandbCleanupHandler initialized.")

    def set_trainer(self, trainer: Trainer):
        self.trainer = trainer
        log.info("Trainer instance registered with the cleanup handler.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        log.info(f"Exiting context. Starting final cleanup (exception type: {exc_type})...")

        if not isinstance(self.trainer, Trainer):
            log.warning("Trainer was not set on the cleanup handler. Skipping cleanup.")
            return

        if not self.trainer.is_global_zero:
            self.trainer.strategy.barrier()
            log.info(f"Rank {self.trainer.global_rank} finished cleanup.")
            return

        self.trainer.strategy.barrier()
        log.info("Rank 0 is proceeding with cleanup after barrier.")

        wandb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, WandbLogger):
                wandb_logger = logger
                break

        if wandb_logger and wandb_logger.experiment and wandb_logger.experiment.dir:
            try:
                # --- 任务一：复制 run-* 目录 ---
                source_run_dir = Path(wandb_logger.experiment.dir).parent
                if not source_run_dir.name.startswith(("run-", "offline-run-")):
                    log.warning(
                        f"Directory '{source_run_dir.name}' does not look like a wandb run dir. Skipping."
                    )
                    return

                destination_dir = Path(self.cfg.paths.output_dir)
                final_run_destination = destination_dir / "wandb_run"

                log.info("=" * 50)
                # --- 关键改动：使用 copytree 而不是 move ---
                log.info("Copying specific wandb run directory...")
                log.info(f"    Source:      {source_run_dir}")
                log.info(f"    Destination: {final_run_destination}")
                shutil.copytree(str(source_run_dir), str(final_run_destination))
                log.info("Successfully copied specific wandb run directory.")

                # --- 任务二：复制 sweep 配置文件 ---
                if wandb_logger.experiment.sweep_id:
                    # ... (这部分逻辑保持不变，依然是 copy) ...
                    sweep_id = wandb_logger.experiment.sweep_id
                    run_id = wandb_logger.experiment.id
                    wandb_base_dir = source_run_dir.parent
                    source_config_file = (
                        wandb_base_dir / f"sweep-{sweep_id}" / f"config-{run_id}.yaml"
                    )

                    if source_config_file.exists():
                        final_config_path = destination_dir / "sweep_params.yaml"
                        log.info(
                            f"Copying sweep config from '{source_config_file}' to '{final_config_path}'..."
                        )
                        shutil.copy(str(source_config_file), str(final_config_path))
                        log.info("Successfully copied sweep config.")
                    else:
                        log.warning(
                            f"Could not find sweep config file at expected path: {source_config_file}"
                        )

                log.info("=" * 50)

            except Exception as e:
                log.error(f"Error during cleanup: {e}", exc_info=True)
        else:
            log.info("No active WandbLogger run found. No cleanup needed.")
