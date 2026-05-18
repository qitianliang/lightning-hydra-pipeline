# src/tasks/ablation.py
"""消融实验模式 — 支持多消融组并行, 超时控制, 统一汇总邮件。"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from src.services.command_builder import CommandBuilder
from src.services.tmux_service import TmuxService
from src.services.wandb_service import WandbService
from src.tasks.base import BaseTask
from src.tasks.evaluate import EvaluateTask
from src.tasks.shared import is_dry_run, safe_get_metric_scores
from src.utils import RankedLogger, SweepError
from src.utils.email_templates import build_ablation_email, send_email_with_mimemultipart
from src.utils.helpers import build_wandb_sweep_url, build_wandb_run_url, build_wandb_group_url, format_reproduction_script, validate_config_keys

log = RankedLogger(__name__, rank_zero_only=True)


class AblationTask(BaseTask):
    """消融实验: 支持多消融组并行执行, 超时控制, 汇总邮件通知。

    流程:
    1. 前置检查: sweep 必须存在, 否则报错
    2. 获取 evaluate 结果 (无则运行, 不发邮件)
    3. 构建所有消融实验组
    4. 并行/串行执行 (配置化 parallel)
    5. 统一收集所有消融结果 + full model 结果
    6. 生成对比表 → 发邮件
    """

    def __init__(
        self,
        cfg: DictConfig,
        wandb_service: WandbService,
        tmux_service: TmuxService,
        command_builder: CommandBuilder,
    ):
        super().__init__(cfg, wandb_service, tmux_service, command_builder)

    def run(self, sweep_id: Optional[str] = None):
        log.info("🧪 ▶️ Launching Ablation Task...")

        sweep_id = self._resolve_sweep_id_with_check(sweep_id)

        if is_dry_run() and (not sweep_id or sweep_id.startswith("dry-run-")):
            log.info("[DRY-RUN] Skipping ablation — no valid sweep_id.")
            return

        # ── Step 1: 获取 sweep ─────────────────────────────────────────
        sweep = self.get_sweep(
            sweep_id,
            wait=self.cfg.override_task.get("wait_for_sweep_finish", True),
            wait_interval=self.cfg.override_task.get("wait_interval_seconds", 15),
        )

        # ── Step 2: 获取 evaluate 结果 (无则运行, 不发邮件) ─────────────
        evaluate_task = EvaluateTask(
            self._full_cfg, self.wandb_service, self.tmux_service, self.command_builder
        )
        eval_report = self.ensure_evaluate_results(sweep_id, sweep, evaluate_task)

        if eval_report is None:
            log.warning("No evaluate report available. Using empty baseline for ablation comparison.")

        # ── Step 3: 获取最优参数 (支持 eval_rank) ──────────────────────
        eval_rank = self.cfg.evaluate_task.get("eval_rank", 1)

        # 提取 full model metrics (按 eval_rank)
        full_model_metrics = self._extract_full_model_metrics(eval_report, eval_rank=eval_rank)
        best_run, config_overrides, config_dict, beautified = self.get_best_run_overrides(
            sweep, self.cfg.override_task.optimized_metric, eval_rank=eval_rank
        )

        # ── Step 4: 构建消融实验组 ──────────────────────────────────────
        ablation_cfg = self.cfg.get("ablation_task", {})
        components = ablation_cfg.get("components", [])

        if not components:
            log.warning("No ablation components defined in ablation_task.components. Nothing to do.")
            return

        num_seeds = ablation_cfg.get("num_seeds", self.cfg.override_task.num_seeds)
        seed_start = ablation_cfg.get("seed_start", self.cfg.override_task.seed_start)
        timeout_secs = ablation_cfg.get("timeout_secs", 600)
        parallel = ablation_cfg.get("parallel", True)
        mode = self.cfg.override_task.mode

        # 为每个消融组件构建实验描述
        experiments: List[Dict] = []
        for comp in components:
            comp_name = comp.name
            comp_overrides = OmegaConf.to_container(comp.get("overrides", {}), resolve=True)

            # 构建消融 override 列表
            override_list = self._build_override_list(comp_overrides)
            final_overrides = config_overrides + override_list

            rank_prefix = f"r{eval_rank}/" if eval_rank > 1 else ""
            group_name = f"ablation/{sweep_id}/{rank_prefix}{comp_name}"

            experiments.append({
                "group_name": group_name,
                "overrides": final_overrides,
                "num_seeds": num_seeds,
                "seed_start": seed_start,
                "_meta": {"name": comp_name, "comp_overrides": comp_overrides},
            })

            log.info(f"  🧪 Ablation group: [{comp_name}] → {group_name}")

        # ── Step 5: 执行实验 ────────────────────────────────────────────
        if parallel and len(experiments) > 1:
            # 并行模式: 一次 launch 所有消融组
            session_name = (
                f"{self.cfg.general.tmux_session_name}_ablation_{sweep_id}"
            )
            log.info(f"🚀 Parallel ablation: {len(experiments)} groups in session '{session_name}'")
            self.execute_parallel_strategy(experiments, session_name, mode)
            self.wait_for_session_with_timeout(session_name, timeout_secs=timeout_secs)
        else:
            # 串行模式: 逐个执行
            for exp in experiments:
                comp_name = exp["_meta"]["name"]
                log.info(f"\n========== 🧪 Ablation: [{comp_name}] ==========")
                eval_session_name = self.execute_strategy(
                    exp["overrides"], exp["group_name"], mode,
                    num_seeds=num_seeds, seed_start=seed_start,
                )
                self.wait_for_session_with_timeout(
                    eval_session_name, timeout_secs=timeout_secs,
                    interval=self.cfg.override_task.get("wait_interval_seconds", 15),
                )

        # ── Step 6: 收集消融结果 ────────────────────────────────────────
        test_metrics = ablation_cfg.get(
            "test_metrics", self.cfg.evaluate_task.test_metrics
        )
        ablation_results = self._collect_ablation_results(
            experiments, test_metrics, ablation_cfg, sweep_id, eval_rank=eval_rank
        )

        # ── Step 7: 构建复现数据 + 发送消融邮件 ─────────────────────────
        reproduction_scripts, group_urls = self._build_reproduction_data(experiments, sweep_id)

        import os
        base_url = os.getenv("WANDB_BASE_URL", "")
        sweep_url = build_wandb_sweep_url(base_url, self.cfg.wandb.entity, self.cfg.wandb.project, sweep_id) if base_url else ""

        # dry-run 下也输出复现脚本和 URLs
        if is_dry_run():
            self._log_reproduction_data(reproduction_scripts, sweep_url, group_urls)

        if not is_dry_run() and ablation_results:
            self._send_ablation_email(
                sweep_id, sweep, best_run, config_dict,
                full_model_metrics, ablation_results,
                reproduction_scripts=reproduction_scripts,
                group_urls=group_urls,
                eval_rank=eval_rank,
            )
        elif is_dry_run():
            log.info("[DRY-RUN] Skipping ablation email.")
        else:
            log.warning("No ablation results collected. Skipping email.")

    # ── 辅助方法 ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_full_model_metrics(eval_report: Optional[dict], eval_rank: int = 1) -> dict:
        """从 evaluate 报告提取 full model metrics (指定 rank)。"""
        full_model_metrics = {}
        if eval_report and "evaluation_summary" in eval_report:
            summary = eval_report["evaluation_summary"]
            rank_key = f"top-{eval_rank}"
            # 新格式: {"top-N": {"metrics": {"test/acc": {"mean": ..., "std": ...}}}}
            if rank_key in summary and "metrics" in summary.get(rank_key, {}):
                full_model_metrics = summary[rank_key]["metrics"]
            # 回退: top-1 (eval_rank=1 的默认)
            elif "top-1" in summary and "metrics" in summary.get("top-1", {}):
                full_model_metrics = summary["top-1"]["metrics"]
            # 旧格式: {"test/acc": {"mean": ..., "std": ...}}
            else:
                for metric_name, metric_data in summary.items():
                    if isinstance(metric_data, dict) and "mean" in metric_data:
                        full_model_metrics[metric_name] = metric_data
        return full_model_metrics

    @staticmethod
    def _build_override_list(comp_overrides: dict) -> list:
        """将 override dict 转为 Hydra override 列表。"""
        override_list = []
        for key, value in comp_overrides.items():
            if "." in key:
                val_str = f'"{value}"' if isinstance(value, list) else str(value)
                override_list.append(f"{key}={val_str}")
        return override_list

    def _collect_ablation_results(
        self,
        experiments: List[Dict],
        test_metrics: list,
        ablation_cfg: DictConfig,
        sweep_id: str,
        eval_rank: int = 1,
    ) -> List[dict]:
        """统一收集所有消融组结果。"""
        import pandas as pd

        ablation_results = []

        for exp in experiments:
            comp_name = exp["_meta"]["name"]
            comp_overrides = exp["_meta"]["comp_overrides"]
            group_name = exp["group_name"]

            if is_dry_run():
                log.info(f"[DRY-RUN] Skipping result collection for ablation '{comp_name}'")
                continue

            abl_runs = self.wandb_service.get_runs_by_group(group_name)
            abl_metrics = {}
            for metric in test_metrics:
                scores = safe_get_metric_scores(abl_runs, metric)
                if scores:
                    series = pd.Series(scores)
                    abl_metrics[metric] = {
                        "mean": series.mean(),
                        "std": series.std(),
                        "values": series.tolist(),
                    }

            # 保存单个消融报告
            report_path_str = str(ablation_cfg.get("report_path", "logs/final_reports/ablation_{sweep_id}"))
            rank_suffix = f"_r{eval_rank}" if eval_rank > 1 else ""
            report_dir = Path(report_path_str.format(sweep_id=f"{sweep_id}{rank_suffix}"))
            report_file = report_dir / f"{comp_name}.json"
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, "w") as f:
                json.dump({
                    "name": comp_name,
                    "overrides": comp_overrides,
                    "metrics": abl_metrics,
                }, f, indent=4)

            ablation_results.append({
                "name": comp_name,
                "metrics": abl_metrics,
                "overrides": comp_overrides,
            })

        return ablation_results

    def _resolve_sweep_id_with_check(self, sweep_id: Optional[str]) -> str:
        """解析 sweep_id, 如果不存在则报错退出。"""
        sweep_id = self.resolve_sweep_id(sweep_id, "ablation_task")
        # 验证 sweep 确实存在
        sweep = self.wandb_service.get_sweep(sweep_id)
        if not sweep:
            raise SweepError(f"Sweep {sweep_id} not found on W&B. Ablation requires a completed sweep.")
        return sweep_id

    def _build_reproduction_data(
        self,
        experiments: List[Dict],
        sweep_id: str,
    ) -> tuple:
        """构建复现脚本列表和 group URL 字典。

        Returns:
            (reproduction_scripts, group_urls)
        """
        import os

        base_url = os.getenv("WANDB_BASE_URL", "")
        entity = self.cfg.wandb.entity
        project = self.cfg.wandb.project
        base_args = list(self.cfg.evaluate_task.run_command.base_args)

        reproduction_scripts = []
        group_urls = {}

        for exp in experiments:
            comp_name = exp["_meta"]["name"]
            group_name = exp["group_name"]
            num_seeds = exp.get("num_seeds", self.cfg.override_task.num_seeds)
            seed_start = exp.get("seed_start", self.cfg.override_task.seed_start)
            seeds = list(range(seed_start, seed_start + num_seeds))

            cmd = format_reproduction_script(
                base_args=base_args,
                overrides=exp["overrides"],
                seeds=seeds,
                group_name=group_name,
            )
            reproduction_scripts.append({"label": comp_name, "command": cmd})

            if base_url:
                group_urls[comp_name] = build_wandb_group_url(
                    base_url, entity, project, group_name
                )

        return reproduction_scripts, group_urls

    def _log_reproduction_data(
        self,
        reproduction_scripts: list,
        sweep_url: str,
        group_urls: dict,
    ):
        """dry-run 模式下日志输出复现脚本和 URLs。"""
        for scr in reproduction_scripts:
            log.info(f"📋 Reproduction Script [{scr['label']}]:")
            log.info(f"   {scr['command']}")
        if sweep_url:
            log.info(f"🔗 Sweep URL: {sweep_url}")
        for name, url in group_urls.items():
            log.info(f"🔗 Group URL [{name}]: {url}")

    def _send_ablation_email(
        self,
        sweep_id: str,
        sweep,
        best_run,
        best_config_dict: dict,
        full_model_metrics: dict,
        ablation_results: List[dict],
        reproduction_scripts: list = None,
        group_urls: dict = None,
        eval_rank: int = 1,
    ):
        """发送消融对比邮件。"""
        import json as json_mod
        import os

        subject = f"🧪 Ablation | {sweep.project} {sweep_id[:6]}"

        # 构建 sweep URL
        base_url = os.getenv("WANDB_BASE_URL", "")
        sweep_url = build_wandb_sweep_url(base_url, self.cfg.wandb.entity, self.cfg.wandb.project, sweep_id) if base_url else ""

        # 提取 best run 元信息
        best_run_metadata = self.extract_run_metadata(best_run) if best_run else {}
        run_url = build_wandb_run_url(
            base_url, self.cfg.wandb.entity, self.cfg.wandb.project,
            best_run.id,
        ) if best_run and base_url else "N/A"

        rank_suffix = f"_r{eval_rank}" if eval_rank > 1 else ""
        report_path_resolved = str(self.cfg.evaluate_task.report_path).format(sweep_id=sweep_id)

        # 收集 checkpoint 路径: 优先读取 eval 保存的 JSON, 回退到扫描
        report_dir = str(Path(report_path_resolved).parent.resolve())
        eval_ckpt_map = self.load_eval_checkpoints(sweep_id, report_dir)
        rank_key = f"top-{eval_rank}"
        if eval_ckpt_map:
            checkpoint_paths = eval_ckpt_map.get(rank_key, eval_ckpt_map.get("top-1", []))
            for key, paths in eval_ckpt_map.items():
                if key != "top-1":
                    for p in paths:
                        if p not in checkpoint_paths:
                            checkpoint_paths.append(p)
            log.info(f"📋 Loaded {len(checkpoint_paths)} eval checkpoints from JSON")
        else:
            project_log_dir = str(Path(self._full_cfg.paths.log_dir).resolve())
            checkpoint_paths = self.collect_checkpoint_paths(project_log_dir)

        workflow_info = {
            "sweep_id": sweep_id,
            "sweep_description": sweep.config.get("description", "N/A"),
            "best_run_config": json_mod.dumps(best_config_dict, indent=4, ensure_ascii=False),
            "report_json_path": str(Path(report_path_resolved).resolve()),
            "report_csv_path": str(Path(report_path_resolved).with_suffix(".csv").resolve()),
            "log_dir": str(Path(report_path_resolved).parent.resolve()),
            "sweep_url": sweep_url,
            "best_run_host": best_run_metadata.get("host", "N/A"),
            "run_url": run_url,
            "best_run_created_at": best_run_metadata.get("created_at", "N/A"),
            "best_run_duration": best_run_metadata.get("duration", "N/A"),
            "checkpoint_paths": checkpoint_paths,
        }

        msg = build_ablation_email(
            notification_cfg=self.cfg.notification,
            subject=subject,
            full_model_metrics=full_model_metrics,
            ablation_results=ablation_results,
            workflow_info=workflow_info,
            reproduction_scripts=reproduction_scripts,
            group_urls=group_urls,
            eval_rank=eval_rank,
        )
        send_email_with_mimemultipart(self.cfg.notification, msg, subject=subject, mode="ablation")
