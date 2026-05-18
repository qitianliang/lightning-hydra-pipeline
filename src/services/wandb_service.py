import sys
from typing import List, Optional

import wandb

# 為了類型提示，我們需要 Run 和 Sweep 的類型
# 如果 wandb 未完全支持類型提示，可以使用 'typing.Any'
try:
    from wandb.apis.public import Run, Sweep
except ImportError:
    Run = Sweep = object  # Fallback for type hinting
from src.utils import RankedLogger, WorkflowError

log = RankedLogger(__name__)


class WandbService:
    """一個專門負責與 Weights & Biases API 交互的服務類。 封裝了所有 wandb API 調用，將業務邏輯與 API 細節解耦。"""

    def __init__(self, entity: str, project: str):
        """初始化 WandbService。

        Args:
            entity (str): W&B 的實體（用戶名或團隊名）。
            project (str): W&B 的項目名。
            console (Console): Rich console 用於打印日誌。
        """
        self.entity = entity
        self.project = project
        self.api = wandb.Api()

    def get_sweep(self, sweep_id: str) -> Optional[Sweep]:
        """根據 Sweep ID 獲取 Sweep 對象。"""
        try:
            sweep_path = f"{self.entity}/{self.project}/{sweep_id}"
            return self.api.sweep(sweep_path)
        except wandb.errors.CommError as e:
            log.info(f"[bold red]Error fetching sweep '{sweep_path}': {e}[/bold red]")
            return None

    def find_best_run(self, sweep: Sweep, metric: str) -> Optional[Run]:
        """從一個 sweep 中根據指定的指標找到最佳的 run。

        Args:
            sweep (Sweep): W&B Sweep 對象。
            metric (str): 用於排序的指標名稱。

        Returns:
            Optional[Run]: 最佳的 Run 對象，如果沒有找到則返回 None。
        """
        runs = sorted(
            sweep.runs,
            key=lambda run: run.summary.get(metric, -1),
            reverse=True,  # 假設指標總是越大越好
        )
        if not runs:
            return None
        return runs[0]

    def find_top_n_runs(self, sweep: Sweep, metric: str, top_n: int) -> List[Run]:
        """從一個 sweep 中根據指定的指標找到前 N 個最佳 run。

        Args:
            sweep (Sweep): W&B Sweep 對象。
            metric (str): 用於排序的指標名稱。
            top_n (int): 取前 N 個（最大 3）。

        Returns:
            List[Run]: 前 N 個最佳 Run 對象列表。
        """
        if top_n > 3:
            raise WorkflowError(f"top_n={top_n} exceeds maximum allowed value of 3")

        runs = sorted(
            sweep.runs,
            key=lambda run: run.summary.get(metric, -1),
            reverse=True,  # 假設指標總是越大越好
        )
        if not runs:
            return []
        return runs[:top_n]

    def get_runs_by_group(self, group_name: str, force_refresh: bool = False) -> List[Run]:
        """根據 group 名稱獲取所有相關的 runs。

        使用專用 API 實例 (非 sweep.runs 緩存) 確保每次查詢結果新鮮。
        避免每次調用都新建 wandb.Api() — 改用惰性初始化緩存實例。

        Args:
            force_refresh: 強制重建 API 實例 (繞過 SDK 連接池緩存)
        """
        log.info(
            f"Calling get_runs_by_group with group_name: '{group_name}' (type: {type(group_name)})"
        )

        if force_refresh or not hasattr(self, "_runs_api"):
            self._runs_api = wandb.Api()
        runs = self._runs_api.runs(
            f"{self.entity}/{self.project}", filters={"group": group_name}
        )
        return list(runs)

    def delete_runs_in_group(self, group_name: str) -> int:
        """刪除指定 group 中的所有 runs。"""
        runs_to_delete = self.get_runs_by_group(group_name)
        if not runs_to_delete:
            log.info(f"No runs found in group '{group_name}' to delete.")
            return 0

        log.info(f"Found {len(runs_to_delete)} runs in group '{group_name}'. Deleting them...")
        for run in runs_to_delete:
            log.info(f"  -> Deleting run: {run.name}")
            run.delete()
        return len(runs_to_delete)
