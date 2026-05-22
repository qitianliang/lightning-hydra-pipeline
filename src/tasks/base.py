# src/tasks/base.py
"""BaseTask — 所有工作流任务的抽象基类, 抽取公共逻辑。"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from omegaconf import DictConfig, OmegaConf
from wandb.apis.public import Run, Sweep

from src.services.command_builder import CommandBuilder
from src.services.sandbox_service import SandboxService
from src.services.tmux_service import TmuxService
from src.services.wandb_service import WandbService
from src.tasks.shared import (
    RerunStrategy,
    ResumeStrategy,
    is_dry_run,
    load_evaluate_report,
)
from src.utils import RankedLogger, SweepError, WorkflowError

log = RankedLogger(__name__, rank_zero_only=True)


class BaseTask:
    """工作流任务基类 — 封装 sweep_id 解析、最优参数获取、策略执行等公共逻辑。"""

    def __init__(
        self,
        cfg: DictConfig,
        wandb_service: WandbService,
        tmux_service: TmuxService,
        command_builder: CommandBuilder,
        sandbox_service: SandboxService = None,
    ):
        self._full_cfg = cfg  # 保留完整 cfg, 供创建子任务用
        self.cfg = cfg.workflow
        self.wandb_service = wandb_service
        self.tmux_service = tmux_service
        self.command_builder = command_builder
        self.sandbox_service = sandbox_service
        self._active_sandboxes: List[Path] = []

    # ── 代码快照准备 ────────────────────────────────────────────────────
    def snapshot_enabled(self) -> bool:
        """Return whether code snapshot isolation is enabled for workflow tasks."""
        return bool(self.cfg.get("snapshot", {}).get("enabled", True))

    def prepare_sandbox(
        self,
        sweep_id: str,
        task_name: str,
        rank: int = 1,
    ) -> str:
        """调用 sandbox_service 构建沙盒目录，返回绝对路径字符串。"""
        if self.sandbox_service is None or not self.snapshot_enabled():
            return ""
        sandbox_path = self.sandbox_service.setup_sandbox(
            sweep_id=sweep_id,
            task_name=task_name,
            rank=rank,
        )
        self._active_sandboxes.append(sandbox_path)
        return str(sandbox_path)

    def teardown_sandboxes(self, archive_logs: bool = False):
        """销毁所有已创建沙盒。"""
        if not self.sandbox_service:
            return
        for sandbox_path in self._active_sandboxes:
            if sandbox_path.exists():
                self.sandbox_service.teardown(sandbox_path, archive_logs=archive_logs)
        self._active_sandboxes.clear()

    # ── sweep_id 解析 ───────────────────────────────────────────────────
    def resolve_sweep_id(self, sweep_id: Optional[str] = None, task_cfg_key: str = "evaluate_task") -> str:
        """解析 sweep_id: 参数 > 配置 > latest 软链接。

        Args:
            task_cfg_key: 从哪个配置段读取 target_sweep_id (evaluate_task / ablation_task / sensitivity_task)
        """
        if sweep_id:
            log.info(f"Using active sweep ID from current workflow: {sweep_id}")
            return sweep_id

        task_cfg = self.cfg.get(task_cfg_key, {})
        if task_cfg.get("target_sweep_id"):
            sweep_id = task_cfg.target_sweep_id
            log.info(f"Using target sweep ID from config: {sweep_id}")
            return sweep_id

        latest_status_file = Path(self.cfg.general.latest_status_file)
        if not latest_status_file.exists():
            raise WorkflowError(f"No Sweep ID provided and latest.yaml {latest_status_file} not found.")
        import yaml
        with open(latest_status_file.resolve()) as f:
            status = yaml.safe_load(f)
            sweep_id = status["sweep_id"]
        log.info(f"Using Sweep ID from latest pointer: {sweep_id}")
        return sweep_id

    # ── 获取 sweep 对象 ────────────────────────────────────────────────
    def get_sweep(self, sweep_id: str, wait: bool = True, wait_interval: int = 15) -> Sweep:
        """获取 Sweep 对象, 可选等待完成。"""
        import time
        sweep = self.wandb_service.get_sweep(sweep_id)
        if not sweep:
            raise SweepError(f"Sweep {sweep_id} could not be found on W&B.")

        if wait:
            while sweep.state != "FINISHED":
                log.info(f"  -> Sweep state is currently '{sweep.state}'. Waiting...")
                time.sleep(wait_interval)
                sweep = self.wandb_service.get_sweep(sweep_id)
            log.info("✅ Sweep has FINISHED on W&B server.")
        return sweep

    # ── 获取最优 run 并提取 config overrides ───────────────────────────
    def get_best_run_overrides(self, sweep: Sweep, optimized_metric: str, eval_rank: int = 1) -> tuple:
        """获取第 N 优 run, 提取 config overrides 和 config dict。

        Args:
            eval_rank: 排名 (1=best, 2=2nd best, ...)

        Returns:
            (best_run, config_overrides_list, config_dict, beautified_json_str)
        """
        if eval_rank <= 1:
            best_run = self.wandb_service.find_best_run(sweep, optimized_metric)
        else:
            top_runs = self.wandb_service.find_top_n_runs(sweep, optimized_metric, eval_rank)
            best_run = top_runs[eval_rank - 1] if len(top_runs) >= eval_rank else None
        if not best_run:
            raise SweepError(f"Could not determine the rank-{eval_rank} run from sweep.")
        log.info(f"🏆 Using rank-{eval_rank} run: {best_run.name}")

        config_overrides = []
        for key, value in best_run.config.items():
            if "." in key:
                val_str = f'"{value}"' if isinstance(value, list) else str(value)
                config_overrides.append(f"{key}={val_str}")

        config_dict = {}
        for item in config_overrides:
            if "=" in item:
                key, value = item.split("=", 1)
                config_dict[key] = value

        beautified = json.dumps(config_dict, indent=4, ensure_ascii=False)
        return best_run, config_overrides, config_dict, beautified

    # ── 执行评估策略 ──────────────────────────────────────────────────
    def execute_strategy(
        self,
        config_overrides: list,
        group_name: str,
        mode: str,
        num_seeds: Optional[int] = None,
        seed_start: Optional[int] = None,
        snapshot_dir: str = "",
    ) -> str | None:
        """执行 rerun/resume 策略, 返回 session name 或 None。"""
        strategy = None
        if mode == "rerun":
            strategy = RerunStrategy(
                self.wandb_service, self.tmux_service, self.command_builder,
                self.cfg, group_name, num_seeds, seed_start, snapshot_dir,
            )
        elif mode == "resume":
            strategy = ResumeStrategy(
                self.wandb_service, self.tmux_service, self.command_builder,
                self.cfg, group_name, num_seeds, seed_start, snapshot_dir,
            )
        else:
            raise WorkflowError(f"Unknown evaluation mode '{mode}'.")

        return strategy.execute(config_overrides)

    # ── 等待 tmux session 结束 ────────────────────────────────────────
    def wait_for_session(self, session_name: str, interval: int = 15):
        """等待 tmux session 关闭。"""
        if session_name and not is_dry_run():
            self.tmux_service.wait_for_session_to_close(session_name, interval)
        elif is_dry_run() and session_name:
            log.info(f"[DRY-RUN] Skipping wait for session '{session_name}'")

    # ── 并行执行多组实验 ──────────────────────────────────────────────
    def execute_parallel_strategy(
        self,
        experiments: List[Dict],
        session_name: str,
        mode: str = "rerun",
        snapshot_dir: str = "",
    ) -> str | None:
        """多实验组并行执行: 收集所有 run → 按 device 槽位 round-robin → 单 tmux session。

        支持 devices=[0,0] 单卡多进程: 槽位数=len(devices), 每槽位独立 worker。

        Args:
            experiments: [{"group_name": str, "overrides": list, "num_seeds": int, "seed_start": int}]
            session_name: tmux session 名称
            mode: "rerun" | "resume"

        Returns:
            session_name 或 None
        """
        if mode == "resume":
            log.info("Resume mode: skipping parallel execution.")
            return None

        devices = self.cfg.devices

        # 展平所有实验的 commands
        all_commands = []
        for exp in experiments:
            num_seeds = exp.get("num_seeds", self.cfg.evaluate_task.get("num_seeds", 1))
            seed_start = exp.get("seed_start", self.cfg.evaluate_task.get("seed_start", 42))
            for i in range(num_seeds):
                cmd = self.command_builder.build_training_run_command(
                    overrides=exp["overrides"],
                    seed=seed_start + i,
                    group_name=exp["group_name"],
                    cwd=snapshot_dir,
                )
                all_commands.append(cmd)

        if not all_commands:
            log.warning("No commands to execute in parallel strategy.")
            return None

        # 按 device 槽位 round-robin 分配
        commands_per_device: List[List[str]] = [[] for _ in devices]
        for i, cmd in enumerate(all_commands):
            commands_per_device[i % len(devices)].append(cmd)

        if is_dry_run():
            log.info(f"[DRY-RUN] Would launch {len(devices)} worker(s) for {len(all_commands)} runs.")
            for i, device in enumerate(devices):
                for cmd in commands_per_device[i]:
                    log.info(f"  [DRY-RUN] GPU {device}: {cmd}")
            return session_name

        # Dry-run 时跳过 rerun 删除；真实 rerun 按 group 去重删除，避免 sensitivity 同组多 combo 重复查询/删除。
        if not is_dry_run():
            seen_groups = set()
            for exp in experiments:
                group_name = exp["group_name"]
                if group_name in seen_groups:
                    continue
                self.wandb_service.delete_runs_in_group(group_name)
                seen_groups.add(group_name)

        if self.tmux_service.session_exists(session_name):
            raise SessionError(f"tmux session '{session_name}' already exists.")

        # 为每个 device 槽位创建 worker
        worker_defs = []
        for i, device in enumerate(devices):
            worker_commands = commands_per_device[i]
            if not worker_commands:
                continue
            worker_script = self.command_builder.build_evaluation_worker_script(
                worker_commands, device, eval_session_name=session_name
            )
            full_worker_command = f"CUDA_VISIBLE_DEVICES={device} bash -c '{worker_script}'"
            worker_defs.append({"device": device, "command": full_worker_command})

        log.info(
            f"🚀 Launching {len(worker_defs)} workers for {len(experiments)} experiments "
            f"({len(all_commands)} total runs, {len(devices)} device slots)."
        )
        self.tmux_service.create_workers_session(session_name, worker_defs)
        return session_name

    # ── 带超时等待 tmux session ──────────────────────────────────────
    def wait_for_session_with_timeout(
        self,
        session_name: str,
        timeout_secs: int = 600,
        interval: int = 15,
    ):
        """等待 tmux session 完成, 超时则 kill。

        Args:
            session_name: tmux session 名称
            timeout_secs: 超时秒数 (默认 600 = 10min). 0 或负数 = 无超时
            interval: 轮询间隔秒数
        """
        if is_dry_run():
            log.info(f"[DRY-RUN] Skipping wait for session '{session_name}' (timeout={timeout_secs}s)")
            return

        if not session_name:
            return

        # 0 或负数 → 无超时, 退化为 wait_for_session
        if not timeout_secs or timeout_secs <= 0:
            self.wait_for_session(session_name, interval)
            return

        start = time.time()
        log.info(f"⏳ Waiting for session '{session_name}' (timeout={timeout_secs}s)...")

        while True:
            elapsed = time.time() - start

            # 检查 session 是否已结束
            if not self.tmux_service.session_exists(session_name):
                log.info(f"✅ Session '{session_name}' completed in {elapsed:.0f}s")
                return

            # 超时检查
            if elapsed >= timeout_secs:
                log.warning(
                    f"⏰ Timeout! Session '{session_name}' exceeded {timeout_secs}s. Killing..."
                )
                try:
                    self.tmux_service.kill_session(session_name)
                    log.warning(f"💀 Session '{session_name}' killed due to timeout.")
                except Exception as e:
                    log.error(f"Failed to kill session '{session_name}': {e}")
                return

            # 等待轮询
            log.info(
                f"  -> Session '{session_name}' still running "
                f"({elapsed:.0f}s/{timeout_secs}s)..."
            )
            time.sleep(interval)

    # ── 提取 best run 元信息 ──────────────────────────────────────────
    @staticmethod
    def extract_run_metadata(run: Run) -> dict:
        """从 W&B Run 对象提取元信息: 主机、运行时间、时长。

        W&B 本地 server 不记录 host/hostname 属性,
        因此使用 platform.node() 获取当前机器名。

        Returns:
            {"host": str, "created_at": str, "duration": str}
        """
        import platform
        from datetime import datetime, timezone

        # W&B 本地 server 不提供 run.host, 使用 platform.node()
        host = getattr(run, "host", None) or platform.node()
        created_at = getattr(run, "created_at", "")

        # 格式化 created_at
        created_str = "N/A"
        duration_str = "N/A"
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                created_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")

                # 计算时长: 优先用 summary._runtime
                runtime_secs = run.summary.get("_runtime", None) if run.summary else None
                if runtime_secs is not None:
                    mins, secs = divmod(int(runtime_secs), 60)
                    hours, mins = divmod(mins, 60)
                    duration_str = f"{hours}h{mins}m{secs}s" if hours else f"{mins}m{secs}s"
            except (ValueError, TypeError):
                created_str = str(created_at)

        return {"host": host, "created_at": created_str, "duration": duration_str}

    @staticmethod
    def collect_checkpoint_paths(log_dir: str, group_name: str = "", since_time: float = None, until_time: float = None) -> list:
        """扫描日志目录收集 model checkpoint 绝对路径。

        扫描 logs/<project>/runs/<timestamp>/checkpoints/*.ckpt 等模式。

        Args:
            log_dir: 日志根目录绝对路径
            group_name: wandb group name (可选, 用于过滤)
            since_time: Unix timestamp, 只收集 mtime >= since_time 的 checkpoint
            until_time: Unix timestamp, 只收集 mtime < until_time 的 checkpoint (用于 per-rank 隔离)

        Returns:
            绝对路径列表
        """
        log_path = Path(log_dir)
        ckpt_records = []
        seen = set()
        expected_group_arg = f"logger.wandb.group={group_name}" if group_name else ""

        for ckpt in log_path.glob("**/checkpoints/*.ckpt"):
            # 只保留 early stopping checkpoint (排除 last.ckpt)
            if ckpt.name == "last.ckpt":
                continue

            try:
                mtime = ckpt.stat().st_mtime
            except OSError:
                continue

            if since_time is not None and mtime < since_time:
                continue
            if until_time is not None and mtime >= until_time:
                continue

            if expected_group_arg:
                metadata_path = ckpt.parent.parent / "wandb_run" / "files" / "wandb-metadata.json"
                try:
                    metadata = json.loads(metadata_path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                args = metadata.get("args", [])
                if expected_group_arg not in args:
                    continue

            resolved = str(ckpt.resolve())
            if resolved not in seen:
                ckpt_records.append((mtime, resolved))
                seen.add(resolved)

        ckpt_records.sort(key=lambda item: item[0])
        return [path for _, path in ckpt_records]

    # ── Eval Checkpoint 持久化 ──────────────────────────────────────
    @staticmethod
    def save_eval_checkpoints(sweep_id: str, checkpoint_map: dict, report_dir: str):
        """将 eval checkpoint 映射保存到本地 JSON 文件。

        Args:
            sweep_id: W&B sweep ID
            checkpoint_map: {"top-1": ["/abs/path/seed42.ckpt", ...], "top-2": [...]}
            report_dir: 报告目录路径
        """
        out_path = Path(report_dir) / f"eval_checkpoints_{sweep_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"sweep_id": sweep_id, "eval_checkpoints": checkpoint_map}, f, indent=4)
        log.info(f"✅ Eval checkpoints saved to {out_path}")

    @staticmethod
    def load_eval_checkpoints(sweep_id: str, report_dir: str) -> dict:
        """读取 eval checkpoint 映射。

        Returns:
            {"top-1": [...], "top-2": [...]} 或空 dict
        """
        in_path = Path(report_dir) / f"eval_checkpoints_{sweep_id}.json"
        if in_path.exists():
            with open(in_path) as f:
                data = json.load(f)
            log.info(f"✅ Loaded eval checkpoints from {in_path}")
            return data.get("eval_checkpoints", {})
        return {}

    # ── 获取或运行 evaluate 结果 ──────────────────────────────────────
    def ensure_evaluate_results(self, sweep_id: str, sweep: Sweep, evaluate_task) -> Optional[dict]:
        """获取 evaluate 报告: 有则返回, 无则运行 evaluate (不发邮件)。

        Args:
            evaluate_task: EvaluateTask 实例 (用于触发运行)

        Returns:
            evaluate 报告 dict, 或 None (dry-run 时)
        """
        # 先尝试读取已有报告
        report_path_fmt = self.cfg.evaluate_task.report_path
        existing = load_evaluate_report(report_path_fmt, sweep_id)
        if existing:
            return existing

        # 无报告 → 运行 evaluate, skip_email=True
        log.info("🔄 No existing evaluate report found. Running evaluate first (email skipped)...")
        if is_dry_run() and (not sweep_id or sweep_id.startswith("dry-run-")):
            log.info("[DRY-RUN] Skipping evaluate run for ablation/sensitivity prerequisite.")
            return None

        # 临时禁用邮件
        original_enabled = self.cfg.notification.enabled
        OmegaConf.update(self.cfg, "notification.enabled", False, merge=True)
        try:
            evaluate_task.run(sweep_id=sweep_id)
        finally:
            OmegaConf.update(self.cfg, "notification.enabled", original_enabled, merge=True)

        # 再次读取
        return load_evaluate_report(report_path_fmt, sweep_id)