import os
import shutil
import subprocess  # nosec B404

import setproctitle
from omegaconf import DictConfig, OmegaConf

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def _get_git_user_name():
    try:
        git_path = shutil.which("git")
        result = subprocess.run(
            [git_path, "config", "--get", "user.name"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=os.getcwd(),
            text=True,
            check=True,
        )  # nosec B603, B607
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, TypeError):
        return "unknown"


def set_process_title(cfg: DictConfig, user_name=None):
    if user_name is None:
        user_name = _get_git_user_name()
    task_name = cfg.get("task_name", "unknown_task")
    title = f"{user_name}.{task_name}"
    setproctitle.setproctitle(title)


# Usage:
# set_process_title(cfg["task_name"])
