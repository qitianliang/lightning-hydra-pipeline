import os
import stat
import tempfile
from pathlib import Path
from typing import List

from omegaconf import DictConfig

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class CommandBuilder:
    """Builds shell commands for conda-run, wandb sweep, and training workers."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.conda_env = self.cfg.sweep_task.conda_env
        self.wandb_project = self.cfg.wandb.project
        self.wandb_entity = self.cfg.wandb.entity

        self._api_key, self._base_url = self._load_wandb_credentials()
        self.cmd = self._init_conda_cmd()

    def _init_conda_cmd(self) -> str:
        return "-p" if os.sep in self.conda_env else "-n"

    def _load_wandb_credentials(self) -> tuple:
        """Read W&B credentials from environment. Raises if missing."""
        log.info("Reading WANDB_API_KEY and WANDB_BASE_URL from environment...")
        api_key = os.getenv("WANDB_API_KEY")
        base_url = os.getenv("WANDB_BASE_URL")

        if not api_key or not base_url:
            msg = (
                "Missing required environment variables for W&B Local.\n"
                "Please set both 'WANDB_API_KEY' and 'WANDB_BASE_URL' in your shell.\n"
                "Example:\n"
                "  export WANDB_API_KEY='your_local_key'\n"
                "  export WANDB_BASE_URL='http://your.server.ip:port'"
            )
            log.error(msg)
            raise RuntimeError(msg)

        log.info("✅ W&B environment variables found.")
        return api_key, base_url

    def _wandb_env_exports(self) -> str:
        """Return shell-compatible `export` lines (safe for heredoc, NOT for argv)."""
        return (
            f"export WANDB_API_KEY='{self._api_key}'\n"
            f"export WANDB_BASE_URL='{self._base_url}'"
        )

    def _write_wandb_env_script(self, inner_command: str, prefix: str = "wandb") -> str:
        """Write a temp shell wrapper that exports W&B creds then runs inner_command.

        Returns the script path (caller should NOT delete — /tmp cleanup handles it).
        """
        fd, script_path = tempfile.mkstemp(suffix=".sh", prefix=f"{prefix}_")
        with os.fdopen(fd, "w") as f:
            f.write("#!/usr/bin/env bash\n")
            f.write("set -euo pipefail\n")
            f.write(self._wandb_env_exports())
            f.write("\n\n")
            f.write(inner_command)
            f.write("\n")
        os.chmod(script_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)  # 0700
        return script_path

    def build_wandb_sweep_command(self) -> list:
        sweep_config_path = self.cfg.sweep_task.config_path

        command = [
            "conda",
            "run",
            self.cmd,
            self.conda_env,
            "--no-capture-output",
            "wandb",
            "sweep",
            sweep_config_path,
            "--project",
            self.wandb_project,
            "--entity",
            self.wandb_entity,
        ]
        return command

    def build_agent_worker_command(self, sweep_path: str) -> str:
        max_failures = self.cfg.sweep_task.get("agent_max_initial_failures", 5)

        inner = (
            f"export WANDB_AGENT_MAX_INITIAL_FAILURES={max_failures}\n"
            f"conda run {self.cmd} {self.conda_env} --no-capture-output wandb agent {sweep_path}\n"
        )
        script_path = self._write_wandb_env_script(inner, prefix="wandb_agent")
        return f"bash {script_path}"

    def build_evaluation_worker_script(
        self, commands: List[str], device: str, eval_session_name: str
    ) -> str:
        """Build a sequential-task bash script for an evaluation GPU worker."""
        worker_log_dir = Path(f"logs/workflow/evaluate/{eval_session_name}")
        worker_log_dir.mkdir(parents=True, exist_ok=True)
        worker_log_file = worker_log_dir / f"gpu_{device}.log"

        # Strip inline env prefixes from commands — env vars are exported at script top
        stripped_commands = []
        for cmd in commands:
            for var in ("WANDB_API_KEY", "WANDB_BASE_URL"):
                # Remove 'VAR='value'' or 'VAR=value' prefix patterns
                while f"{var}=" in cmd:
                    # Find the end of this env var assignment
                    idx = cmd.index(f"{var}=")
                    # Find the end of the value (next space not inside quotes)
                    rest = cmd[idx + len(var) + 1:]
                    if rest.startswith("'") or rest.startswith('"'):
                        quote = rest[0]
                        end = rest.index(quote, 1) + 1
                    else:
                        end = rest.index(" ") if " " in rest else len(rest)
                    cmd = cmd[:idx] + cmd[idx + len(var) + 1 + end + 1:]
                # Clean up double spaces left behind
                cmd = cmd.replace("  ", " ").strip()
            stripped_commands.append(cmd)

        # 將每個指令用括號包起來，確保其原子性，然後用 '&&' 連接
        chained_command = " && \\\n".join([f"({cmd})" for cmd in stripped_commands])

        # 生成一個只包含最核心邏輯的極簡腳本
        # 注意: export 放在 exec tee 之前，避免憑證寫入日誌文件
        worker_script = f"""#!/usr/bin/env bash
set -euo pipefail

# ── W&B credentials (exported before tee, never captured in logs) ──
{self._wandb_env_exports()}

# ── Setup logging ──
mkdir -p '{worker_log_dir}'
exec > >(tee '{worker_log_file}') 2>&1

# ── Start ──
echo "Worker for GPU {device} started at $(date). Logging to {worker_log_file}"
echo "Executing {len(commands)} tasks sequentially..."
echo "---"

{chained_command}

echo "---"
echo "✅ All tasks for GPU {device} finished successfully at $(date)."
exit
"""
        return worker_script

    def build_training_run_command(self, overrides: list, seed: int, group_name: str) -> str:
        """為單次的多種子評估實驗構建訓練指令。"""
        # 複製基礎指令列表以避免修改原始配置
        base_args = list(self.cfg.evaluate_task.run_command.base_args)

        # 在已有的 'python' 後面插入 '-u' 開啟無緩衝日誌輸出
        if base_args and base_args[0] == "python":
            base_args.insert(1, "-u")

        train_cmd_parts = (
            base_args + overrides + [f"seed={seed}", f"logger.wandb.group={group_name}"]
        )

        # 直接拼接成最終的 python 指令
        # Note: W&B credentials are exported by the calling wrapper script/environment,
        # not inlined here, to avoid leaking secrets via /proc/PID/cmdline.
        final_command = " ".join(train_cmd_parts)
        log.info(f"{self.conda_env=}")
        return f"conda run {self.cmd} {self.conda_env} --no-capture-output {final_command}"
