import shutil
import subprocess  # nosec B404
import time
from typing import List

from src.utils import RankedLogger

log = RankedLogger(__name__)


class TmuxService:
    """Manages tmux sessions for GPU-parallel training workers."""

    def session_exists(self, session_name: str) -> bool:
        tmux_path = shutil.which("tmux")
        result = subprocess.run(
            [tmux_path, "has-session", "-t", session_name],
            capture_output=True,
            text=True,
            check=False,  # nosec B603
        )  # nosec B603, B607
        return result.returncode == 0

    def create_workers_session(self, session_name: str, commands_per_worker: List[dict]):
        """Create a tmux session with one window per GPU worker, each running a sequential task list."""
        if not commands_per_worker:
            log.info("[yellow]Warning: No worker commands provided to create session.[/yellow]")
            return

        is_first_window = True
        for worker_info in commands_per_worker:
            device = worker_info["device"]
            command = worker_info["command"]
            window_name = f"GPU-{device}"

            if is_first_window:
                tmux_path = shutil.which("tmux")
                subprocess.run(
                    [
                        tmux_path,
                        "new-session",
                        "-d",
                        "-s",
                        session_name,
                        "-n",
                        window_name,
                        command,
                    ],
                    check=False,
                )  # nosec B603, B607
                is_first_window = False
            else:
                tmux_path = shutil.which("tmux")
                subprocess.run(
                    [tmux_path, "new-window", "-t", session_name, "-n", window_name, command],
                    check=False,
                )  # nosec B603, B607

    def wait_for_session_to_close(self, session_name: str, interval_seconds: int):
        """Block until the tmux session exits, polling every interval_seconds."""
        log.info(
            f"Waiting for tmux session '{session_name}' to close. Checking every {interval_seconds}s..."
        )
        while self.session_exists(session_name):
            log.info(f"  -> Session '{session_name}' is still running. Waiting...")
            time.sleep(interval_seconds)
        log.info(f"✅ Tmux session '{session_name}' finished.")

    def kill_session(self, session_name: str, graceful_timeout: int = 10):
        """Kill a tmux session, with graceful SIGTERM before force kill.

        Sends Ctrl+C (SIGINT) first so wandb can flush and mark runs as "failed"
        instead of leaving them in "running" (orphan) state on the server.

        Args:
            graceful_timeout: seconds to wait after SIGINT before kill-session
        """
        tmux_path = shutil.which("tmux")
        log.info(f"Gracefully stopping session: {session_name} (Ctrl+C)...")
        subprocess.run(
            [tmux_path, "send-keys", "-t", session_name, "C-c"],
            check=False,
        )
        time.sleep(min(graceful_timeout, 10))  # cap at 10s so timeout isn't wasted

        log.info(f"Force-killing tmux session: {session_name}")
        subprocess.run(
            [tmux_path, "kill-session", "-t", session_name], check=False
        )  # nosec B603, B607
