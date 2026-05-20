# src/tasks/sweep.py
"""Stage 1: 超参数搜索任务。"""

import os
import re
import subprocess  # nosec B404, B603
import sys
from pathlib import Path

import yaml
from omegaconf import DictConfig

from src.services.command_builder import CommandBuilder
from src.services.tmux_service import TmuxService
from src.tasks.shared import is_dry_run
from src.utils import RankedLogger, SweepError

log = RankedLogger(__name__, rank_zero_only=True)


class SweepTask:
    """Stage 1: 创建 wandb sweep → 多 GPU agent → 等待完成。"""

    def __init__(
        self,
        cfg: DictConfig,
        tmux_service: TmuxService,
        command_builder: CommandBuilder,
        subprocess_module=subprocess,
    ):
        self.cfg = cfg.workflow
        self.tmux_service = tmux_service
        self.command_builder = command_builder
        self.subprocess = subprocess_module

    def run(self) -> str:
        """Create wandb sweep, launch agents, wait for completion. Returns sweep_id."""
        log.info("▶️ Stage 1: Launching Sweep Task...")
        command = self.command_builder.build_wandb_sweep_command()

        if is_dry_run():
            log.info(f"[DRY-RUN] Would run: {' '.join(command)}")
            log.info("[DRY-RUN] Sweep creation skipped. Returning mock sweep_id.")
            return "dry-run-mock-sweep"

        try:
            result = self.subprocess.run(command, capture_output=True, text=True, check=True)
            full_output = result.stdout + "\n" + result.stderr
            match = re.search(r"wandb agent ([\S]+)", full_output)
            if not match:
                raise RuntimeError("Could not parse Sweep Path from wandb output.")
            full_sweep_path = match.group(1)
            sweep_id = full_sweep_path.split("/")[-1]
            log.info(f"✅ Sweep created successfully! ID: {sweep_id}")
        except Exception as e:
            raise SweepError(f"Error creating sweep: {e}") from e

        session_name = f"{self.cfg.general.tmux_session_name}_{sweep_id}"
        if self.tmux_service.session_exists(session_name):
            raise SessionError(f"tmux session '{session_name}' already exists.")

        worker_defs = []
        for device in self.cfg.devices:
            agent_command_script = self.command_builder.build_agent_worker_command(full_sweep_path)
            full_worker_command = f"CUDA_VISIBLE_DEVICES={device} bash -c '{agent_command_script}'"
            worker_defs.append({"device": device, "command": full_worker_command})

        self.tmux_service.create_workers_session(session_name, worker_defs)

        # Save status file + latest symlink
        status_dir = Path(self.cfg.general.status_dir)
        status_dir.mkdir(parents=True, exist_ok=True)
        status_file_path = status_dir / f"status_{sweep_id}.yaml"
        status = {
            "sweep_id": sweep_id,
            "sweep_path": full_sweep_path,
            "tmux_session_name": session_name,
            "sweep_config": self.cfg.sweep_task.config_path,
        }
        with open(status_file_path, "w") as f:
            yaml.dump(status, f)

        latest_file_path = Path(self.cfg.general.latest_status_file)
        if latest_file_path.is_symlink() or latest_file_path.exists():
            latest_file_path.unlink()
        os.symlink(status_file_path.name, latest_file_path)

        log.info(f"Monitor Sweep: tmux attach -t {session_name}")
        self.tmux_service.wait_for_session_to_close(
            session_name, self.cfg.evaluate_task.wait_interval_seconds
        )

        return sweep_id