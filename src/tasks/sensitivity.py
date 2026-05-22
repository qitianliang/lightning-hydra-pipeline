# src/tasks/sensitivity.py
"""参数敏感性模式 — 支持多敏感性研究列表, 并行执行, 多图邮件汇总。"""

import itertools
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.services.command_builder import CommandBuilder
from src.services.tmux_service import TmuxService
from src.services.wandb_service import WandbService
from src.tasks.base import BaseTask
from src.tasks.evaluate import EvaluateTask
from src.tasks.shared import is_dry_run, safe_get_metric_scores
from src.utils import RankedLogger, SweepError, WorkflowError
from src.utils.email_templates import build_sensitivity_email, send_email_with_mimemultipart
from src.utils.helpers import (
    build_wandb_sweep_url,
    build_wandb_run_url,
    build_wandb_group_url,
    format_reproduction_script,
    validate_config_keys,
)
from src.utils.visualization import plot_sensitivity_1d, plot_sensitivity_2d

log = RankedLogger(__name__, rank_zero_only=True)


class SensitivityTask(BaseTask):
    """参数敏感性分析: 支持 sensitivities 列表, 并行执行, 多图邮件。

    支持 1D (单参数) 和 2D (双参数笛卡尔积) grid。
    每个 study 一个 W&B group, 所有 combos × seeds 在同一 group 下。

    流程:
    1. 前置检查: sweep 必须存在
    2. 获取 evaluate 结果 (无则运行, 不发邮件)
    3. 从 sweep 取最优参数
    4. 解析 sensitivities 列表 → 展开所有参数组合
    5. 并行执行所有实验 (per-study group)
    6. 按 study 收集结果 → 按 config 分 combo → 绘图
    7. 保存汇总报告 → 邮件
    """

    def run(self, sweep_id: Optional[str] = None):
        log.info("📐 ▶️ Launching Sensitivity Task...")
        try:
            self._run_sensitivity(sweep_id)
        finally:
            self.teardown_sandboxes()

    def _run_sensitivity(self, sweep_id: Optional[str] = None):
        if is_dry_run() and (not sweep_id or sweep_id.startswith("dry-run-")):
            log.info("[DRY-RUN] Skipping sensitivity — no valid sweep_id.")
            return

        sweep_id = self._resolve_sweep_id_with_check(sweep_id)

        # ── Step 1: 获取 sweep ─────────────────────────────────────────
        sens_cfg = self.cfg.get("sensitivity_task", {})
        sweep = self.get_sweep(
            sweep_id,
            wait=sens_cfg.get("wait_for_sweep_finish", True),
            wait_interval=sens_cfg.get("wait_interval_seconds", 15),
        )

        # ── Step 2: 获取 evaluate 结果 (无则运行, 不发邮件) ─────────────
        evaluate_task = EvaluateTask(
            self._full_cfg,
            self.wandb_service,
            self.tmux_service,
            self.command_builder,
            self.sandbox_service,
        )
        eval_report = self.ensure_evaluate_results(sweep_id, sweep, evaluate_task)

        # ── Step 2.5: eval_rank (全局使用) ─────────────────────────────
        eval_rank = sens_cfg.get("eval_rank", 1)
        best_params_metrics = self._extract_best_params_metrics(eval_report, eval_rank=eval_rank)

        # ── Step 3: 获取最优参数 (支持 eval_rank) ──────────────────────
        optimized_metric = sens_cfg.get(
            "optimized_metric",
            self.cfg.evaluate_task.get("optimized_metric", "val/acc_best"),
        )
        best_run, config_overrides, config_dict, beautified = self.get_best_run_overrides(
            sweep, optimized_metric, eval_rank=eval_rank
        )

        # ── Step 3.5: 准备代码快照 ──────────────────────────────────────
        snapshot_dir = ""
        if self.sandbox_service and self.snapshot_enabled():
            log.info(f"📦 Preparing sandbox for sensitivity (rank-{eval_rank})")
            snapshot_dir = self.prepare_sandbox(
                sweep_id=sweep_id,
                task_name="sensitivity",
                rank=eval_rank,
            )
            log.info(f"   → sandbox dir: {snapshot_dir}")

        # ── Step 4: 解析 sensitivities 列表 ────────────────────────────
        studies = self._parse_sensitivity_studies(sens_cfg)

        if not studies:
            raise WorkflowError("No sensitivity studies defined.")

        # 校验 param_grid keys
        for study in studies:
            keys = list(study["param_grid"].keys())
            validated = validate_config_keys(keys, source=f"sensitivity/{study['name']}")
            if len(validated) != len(keys):
                invalid = set(keys) - set(validated)
                raise WorkflowError(
                    f"Study [{study['name']}] has invalid keys: {invalid}. "
                    f"Fix config (e.g. 'optimizer.lr' → 'model.optimizer.lr')."
                )

        test_metrics = sens_cfg.get("test_metrics", self.cfg.evaluate_task.test_metrics)
        primary_metric = sens_cfg.get("primary_metric", test_metrics[0] if test_metrics else "test/acc")
        num_seeds = sens_cfg.get("num_seeds", self.cfg.evaluate_task.get("num_seeds", 1))
        seed_start = sens_cfg.get("seed_start", self.cfg.evaluate_task.get("seed_start", 42))
        timeout_secs = sens_cfg.get("timeout_secs", 600)
        figure_width = sens_cfg.get("figure_width", 3.5)
        figure_dpi = sens_cfg.get("figure_dpi", 300)
        mode = sens_cfg.get("mode", self.cfg.evaluate_task.get("mode", "rerun"))

        # ── Step 5: 展开所有 study 的参数组合 → 实验 ──────────────────
        # 每个 study 一个 group_name
        study_groups: Dict[str, str] = {}  # study_name → group_name
        all_experiments: List[Dict] = []

        for study in studies:
            study_name = study["name"]
            param_grid = study["param_grid"]
            axis_labels = study.get("axis_labels", {})

            param_keys = list(param_grid.keys())
            param_values_lists = [v if isinstance(v, list) else [v] for v in param_grid.values()]
            combinations = list(itertools.product(*param_values_lists))

            # Grid 组合数上限检查 (从配置读取, 默认100)
            max_grid = self.cfg.sweep_task.get("max_grid_combinations", 100)
            if len(combinations) > max_grid:
                raise WorkflowError(
                    f"Grid组合数 {len(combinations)} > {max_grid} (study: {study_name})"
                )

            is_2d = len(param_keys) == 2
            is_1d = len(param_keys) == 1
            log.info(
                f"📐 Study [{study_name}]: {len(combinations)} combinations "
                f"({'2D' if is_2d else '1D' if is_1d else f'{len(param_keys)}D'})"
            )

            # 一个 study 一个 group
            rank_prefix = f"r{eval_rank}/" if eval_rank > 1 else ""
            group_name = f"sensitivity/{sweep_id}/{rank_prefix}{study_name}"
            study_groups[study_name] = group_name

            for combo_idx, combo_values in enumerate(combinations):
                combo_dict = dict(zip(param_keys, combo_values))

                # 构建 override 列表
                override_list = self._build_override_list(combo_dict)

                # 短名称: study_name + param 缩写 (lin1_32.lin2_64)
                short_parts = []
                for k, v in combo_dict.items():
                    short_k = str(k).split(".")[-1]
                    short_parts.append(f"{short_k}_{v}")
                combo_name = f"{study_name}/{'.'.join(short_parts)}"

                all_experiments.append({
                    "group_name": group_name,  # 同 study 共享 group
                    "overrides": config_overrides + override_list,
                    "num_seeds": num_seeds,
                    "seed_start": seed_start,
                    "_meta": {
                        "study_name": study_name,
                        "combo_dict": combo_dict,
                        "combo_values": combo_values,
                        "combo_name": combo_name,
                    },
                })

                log.info(f"  -> [{study_name} Grid {combo_idx+1}/{len(combinations)}] {combo_name}")

        total_runs = len(all_experiments) * num_seeds
        log.info(
            f"📐 Total: {len(studies)} studies, {len(all_experiments)} combos, "
            f"{total_runs} runs in {len(study_groups)} groups"
        )

        # ── Step 6: 并行执行所有实验 ───────────────────────────────────
        session_name = f"{self.cfg.general.tmux_session_name}_sensitivity_{sweep_id}"
        log.info(f"🚀 Parallel sensitivity: {len(all_experiments)} experiments in session '{session_name}'")
        self.execute_parallel_strategy(all_experiments, session_name, mode, snapshot_dir=snapshot_dir)
        self.wait_for_session_with_timeout(session_name, timeout_secs=timeout_secs)

        # ── Step 7: 按 study 收集结果 ──────────────────────────────────
        study_results: Dict[str, dict] = {}
        study_plots: Dict[str, Tuple[Optional[Path], Optional[Path]]] = {}

        rank_suffix_dir = f"_rank{eval_rank}"
        report_path_str = str(sens_cfg.get("report_path", "logs/final_reports/sensitivity_{sweep_id}.json"))
        output_dir = Path(report_path_str.format(sweep_id=f"{sweep_id}{rank_suffix_dir}").rsplit(".", 1)[0])
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for study in studies:
            study_name = study["name"]
            param_grid = study["param_grid"]
            axis_labels = study.get("axis_labels", {})
            param_keys = list(param_grid.keys())
            param_values_lists = [v if isinstance(v, list) else [v] for v in param_grid.values()]
            combinations = list(itertools.product(*param_values_lists))
            group_name = study_groups[study_name]

            if is_dry_run():
                log.info(f"[DRY-RUN] Skipping result collection for study '{study_name}'")
                study_results[study_name] = {
                    "combo_metrics": {},
                    "param_grid_results": [],
                    "param_keys": param_keys,
                    "param_values_lists": param_values_lists,
                    "combinations": combinations,
                }
                study_plots[study_name] = (None, None)
                continue

            # 从 W&B group 获取所有 runs, 按 config 中的变参分 combo。
            # W&B Local 删除旧 group 后可能短时间返回 stale runs；这里强制刷新并等待本轮 runs 可见。
            import time as _time

            expected_runs = len(combinations) * num_seeds
            runs = []
            for attempt in range(6):
                runs = self.wandb_service.get_runs_by_group(group_name, force_refresh=True)
                if len(runs) >= expected_runs:
                    break
                if attempt < 5:
                    log.warning(
                        f"Study [{study_name}] group '{group_name}': "
                        f"got {len(runs)}/{expected_runs} runs after refresh; retrying..."
                    )
                    _time.sleep(5)
            combo_metrics: dict = {}
            param_grid_results: List[dict] = []

            for combo_values in combinations:
                combo_dict = dict(zip(param_keys, combo_values))
                # 筛选属于该 combo 的 runs (按 config 匹配)
                combo_runs = self._filter_runs_by_combo(runs, combo_dict)
                metrics = {}
                for metric in test_metrics:
                    scores = safe_get_metric_scores(combo_runs, metric)
                    if scores:
                        series = pd.Series(scores)
                        metrics[metric] = {
                            "mean": series.mean(),
                            "std": series.std() if len(scores) > 1 else 0.0,
                            "values": series.tolist(),
                        }

                # NaN 校验
                has_valid = False
                for m_name, m_val in metrics.items():
                    if not math.isnan(m_val["mean"]):
                        has_valid = True
                        break

                if not has_valid:
                    log.warning(
                        f"⚠️ Study [{study_name}] combo {combo_dict}: "
                        f"all metrics are NaN. Skipping this combo."
                    )
                    continue

                combo_metrics[combo_values] = metrics
                param_desc = ", ".join(f"{k}={v}" for k, v in combo_dict.items())
                param_grid_results.append({
                    "param_desc": param_desc,
                    "metrics": metrics,
                    "combo_dict": combo_dict,
                })

            valid_count = len(combo_metrics)
            total_count = len(combinations)
            if valid_count < total_count:
                log.warning(
                    f"⚠️ Study [{study_name}]: {valid_count}/{total_count} combos have valid metrics"
                )

            study_results[study_name] = {
                "combo_metrics": combo_metrics,
                "param_grid_results": param_grid_results,
                "param_keys": param_keys,
                "param_values_lists": param_values_lists,
                "combinations": combinations,
            }

            # ── 绘图 ────────────────────────────────────────────────
            png_path, pdf_path = None, None
            if combo_metrics:
                is_1d = len(param_keys) == 1
                is_2d = len(param_keys) == 2

                if is_1d:
                    png_path, pdf_path = self._plot_1d(
                        param_keys[0], combinations, combo_metrics,
                        primary_metric, axis_labels, output_dir,
                        figure_width, figure_dpi, study_name=study_name,
                        group_name=group_name,
                    )
                elif is_2d:
                    png_path, pdf_path = self._plot_2d(
                        param_keys, param_values_lists, combinations,
                        combo_metrics, primary_metric, axis_labels,
                        output_dir, figure_width, figure_dpi, study_name=study_name,
                        group_name=group_name,
                    )
                else:
                    log.warning(
                        f"Study [{study_name}] with {len(param_keys)} params: "
                        "no plot support. Only tables will be generated."
                    )
            else:
                log.warning(f"⚠️ Study [{study_name}]: no valid combos for plotting.")

            study_plots[study_name] = (png_path, pdf_path)

        # ── Step 8: 保存汇总报告 ───────────────────────────────────────
        if not is_dry_run():
            _rp = str(sens_cfg.get("report_path", "logs/final_reports/sensitivity_{sweep_id}.json"))
            rank_suffix = f"_rank{eval_rank}"
            report_path = Path(_rp.format(sweep_id=f"{sweep_id}{rank_suffix}"))
            report_path.parent.mkdir(parents=True, exist_ok=True)

            all_grid_results = []
            for study_name, sr in study_results.items():
                for r in sr["param_grid_results"]:
                    all_grid_results.append({
                        "study": study_name,
                        "param_desc": r["param_desc"],
                        "metrics": r["metrics"],
                    })

            with open(report_path, "w") as f:
                json.dump({
                    "sweep_id": sweep_id,
                    "primary_metric": primary_metric,
                    "best_params_metrics": best_params_metrics,
                    "studies": list(study_results.keys()),
                    "param_grid_results": all_grid_results,
                }, f, indent=4)
            log.info(f"✅ Sensitivity report saved to {report_path}")

        # ── Step 9: 构建复现数据 + 发送邮件 ────────────────────────────
        reproduction_scripts, group_urls = self._build_reproduction_data(
            all_experiments, sweep_id
        )

        import os as _os
        _base_url = _os.getenv("WANDB_BASE_URL", "")
        sweep_url = build_wandb_sweep_url(_base_url, self.cfg.wandb.entity, self.cfg.wandb.project, sweep_id) if _base_url else ""

        # dry-run 下也输出复现脚本和 URLs
        if is_dry_run():
            self._log_reproduction_data(reproduction_scripts, sweep_url, group_urls)
            log.info("[DRY-RUN] Skipping sensitivity email.")
        else:
            # 汇总所有 study 的结果和图
            all_grid_results = []
            image_paths: List[Tuple[Path, Path, str]] = []  # (png, pdf, group_name)

            for study_name, sr in study_results.items():
                all_grid_results.extend(sr["param_grid_results"])
                png_path, pdf_path = study_plots.get(study_name, (None, None))
                if png_path and pdf_path:
                    image_paths.append((png_path, pdf_path, study_groups[study_name]))

            # 只要有有效的 grid results 就发邮件
            if all_grid_results:
                self._send_sensitivity_email(
                    sweep_id, sweep, best_run, config_dict,
                    best_params_metrics, all_grid_results,
                    image_paths,
                    reproduction_scripts=reproduction_scripts,
                    group_urls=group_urls,
                    eval_rank=eval_rank,
                )
            else:
                worker_log_root = Path("logs/workflow/evaluate").resolve()
                raise WorkflowError(
                    f"No valid sensitivity results collected for sweep {sweep_id}. "
                    "Training workers likely failed. "
                    f"Check worker logs under {worker_log_root}."
                )

    # ── 辅助方法 ────────────────────────────────────────────────────────

    @staticmethod
    def _get_nested_config(config: dict, dotted_key: str):
        """从嵌套 dict 中按点分路径取值。

        W&B 存储 Hydra config 为嵌套 dict, 如:
          config['model']['net']['lin1_size'] = 32
        而非 config['model.net.lin1_size'] = 32
        """
        keys = dotted_key.split(".")
        val = config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return None
        return val

    @staticmethod
    def _filter_runs_by_combo(runs: list, combo_dict: dict) -> list:
        """从 runs 列表中筛选属于特定 combo 的 runs。

        匹配逻辑: run.config 中 combo_dict 的所有 key 值一致。
        支持 W&B 嵌套 dict 存储格式 (如 model.net.lin1_size → config['model']['net']['lin1_size'])。
        """
        filtered = []
        for run in runs:
            match = True
            for key, value in combo_dict.items():
                # 先尝试 flat key (run.config.get), 再尝试 nested dict
                run_val = run.config.get(key)
                if run_val is None:
                    run_val = SensitivityTask._get_nested_config(run.config, key)
                if run_val is None:
                    match = False
                    break
                # 数值比较: 兼容 int/float/str
                try:
                    if float(run_val) != float(value):
                        match = False
                        break
                except (ValueError, TypeError):
                    if str(run_val) != str(value):
                        match = False
                        break
            if match:
                filtered.append(run)
        return filtered

    @staticmethod
    def _extract_best_params_metrics(eval_report: Optional[dict], eval_rank: int = 1) -> dict:
        """从 evaluate 报告提取 best params metrics (指定 rank)。"""
        best_params_metrics = {}
        if eval_report and "evaluation_summary" in eval_report:
            summary = eval_report["evaluation_summary"]
            rank_key = f"top-{eval_rank}"
            # 新格式: {"top-N": {"metrics": {"test/acc": {"mean": ..., "std": ...}}}}
            if rank_key in summary and "metrics" in summary.get(rank_key, {}):
                best_params_metrics = summary[rank_key]["metrics"]
            # 回退: top-1
            elif "top-1" in summary and "metrics" in summary.get("top-1", {}):
                best_params_metrics = summary["top-1"]["metrics"]
            # 旧格式: {"test/acc": {"mean": ..., "std": ...}}
            else:
                for metric_name, metric_data in summary.items():
                    if isinstance(metric_data, dict) and "mean" in metric_data:
                        best_params_metrics[metric_name] = metric_data
        return best_params_metrics

    def _parse_sensitivity_studies(self, sens_cfg: DictConfig) -> List[dict]:
        """解析敏感性研究列表, 向后兼容单个 param_grid。

        Returns:
            [{"name": str, "param_grid": dict, "axis_labels": dict}, ...]
        """
        sensitivities = sens_cfg.get("sensitivities", None)

        if sensitivities is not None and len(sensitivities) > 0:
            # 新格式: sensitivities 列表
            studies = []
            for s in sensitivities:
                study = {
                    "name": s.get("name", "unnamed_study"),
                    "param_grid": OmegaConf.to_container(s.get("param_grid", {}), resolve=True),
                    "axis_labels": OmegaConf.to_container(s.get("axis_labels", {}), resolve=True),
                }
                if not study["param_grid"]:
                    log.warning(f"Study '{study['name']}' has empty param_grid. Skipping.")
                    continue
                studies.append(study)
            return studies

        # 向后兼容: 单个 param_grid
        param_grid = OmegaConf.to_container(sens_cfg.get("param_grid", {}), resolve=True)
        if param_grid:
            axis_labels = OmegaConf.to_container(sens_cfg.get("axis_labels", {}), resolve=True)
            log.info("📐 Using legacy param_grid format (wrapped as single study).")
            return [{
                "name": "sensitivity",
                "param_grid": param_grid,
                "axis_labels": axis_labels,
            }]

        return []

    @staticmethod
    def _build_override_list(combo_dict: dict) -> list:
        """将 combo dict 转为 Hydra override 列表。"""
        override_list = []
        for key, value in combo_dict.items():
            if "." in key:
                val_str = f'"{value}"' if isinstance(value, list) else str(value)
                override_list.append(f"{key}={val_str}")
        return override_list

    # ── 绘图辅助方法 ────────────────────────────────────────────────────

    def _plot_1d(
        self, param_key, combinations, combo_metrics,
        primary_metric, axis_labels, output_dir,
        figure_width, figure_dpi, study_name: str = "sensitivity",
        group_name: str = "",
    ):
        """1D 折线图。"""
        param_values = [combo[0] for combo in combinations]
        means = []
        stds = []
        for combo in combinations:
            m = combo_metrics.get(combo, {}).get(primary_metric, {})
            means.append(m.get("mean", 0) if m and not math.isnan(m.get("mean", 0)) else 0)
            stds.append(m.get("std", 0) if m and not math.isnan(m.get("std", 0)) else 0)

        xlabel = axis_labels.get(param_key, param_key)
        ylabel = primary_metric

        return plot_sensitivity_1d(
            param_name=param_key,
            title="",
            param_values=param_values,
            metric_means=means,
            metric_stds=stds,
            xlabel=xlabel,
            ylabel=ylabel,
            output_dir=output_dir,
            figure_width=figure_width,
            figure_dpi=figure_dpi,
        )

    def _plot_2d(
        self, param_keys, param_values_lists, combinations,
        combo_metrics, primary_metric, axis_labels,
        output_dir, figure_width, figure_dpi,
        study_name: str = "sensitivity",
        group_name: str = "",
    ):
        """2D 热力图。"""
        x_values = param_values_lists[0]
        y_values = param_values_lists[1]

        # 构建 metric_matrix: shape (len_y, len_x)
        metric_matrix = np.zeros((len(y_values), len(x_values)))
        for combo in combinations:
            x_idx = x_values.index(combo[0])
            y_idx = y_values.index(combo[1])
            m = combo_metrics.get(combo, {}).get(primary_metric, {})
            if m and not math.isnan(m.get("mean", 0)):
                metric_matrix[y_idx, x_idx] = m["mean"]

        xlabel = axis_labels.get(param_keys[0], param_keys[0])
        ylabel = axis_labels.get(param_keys[1], param_keys[1])

        return plot_sensitivity_2d(
            param_x_name=param_keys[0],
            title="",
            param_y_name=param_keys[1],
            param_x_values=x_values,
            param_y_values=y_values,
            metric_matrix=metric_matrix,
            xlabel=xlabel,
            ylabel=ylabel,
            output_dir=output_dir,
            figure_width=figure_width,
            figure_dpi=figure_dpi,
        )

    # ── 其他辅助 ────────────────────────────────────────────────────────

    def _resolve_sweep_id_with_check(self, sweep_id: Optional[str]) -> str:
        """解析 sweep_id, 不存在则报错。"""
        sweep_id = self.resolve_sweep_id(sweep_id, "sensitivity_task")
        sweep = self.wandb_service.get_sweep(sweep_id)
        if not sweep:
            raise SweepError(f"Sweep {sweep_id} not found. Sensitivity requires a completed sweep.")
        return sweep_id

    def _build_reproduction_data(
        self,
        experiments: List[Dict],
        sweep_id: str,
    ) -> tuple:
        """构建复现脚本列表和 group URL 字典。

        按 study 聚合: 一个 study 一条脚本 + 一个 group URL。
        变参 (param_grid keys) 合并为 [v1,v2,...] 格式,
        多个 seeds 合并为 [s1,s2,...] 格式。

        Returns:
            (reproduction_scripts, group_urls)
        """
        import os

        base_url = os.getenv("WANDB_BASE_URL", "")
        entity = self.cfg.wandb.entity
        project = self.cfg.wandb.project
        base_args = list(self.cfg.evaluate_task.run_command.base_args)

        # 按 study_name 分组
        study_exps: Dict[str, List[Dict]] = {}
        for exp in experiments:
            sn = exp["_meta"]["study_name"]
            study_exps.setdefault(sn, []).append(exp)

        reproduction_scripts = []
        group_urls = {}

        for study_name, exps in study_exps.items():
            group_name = exps[0]["group_name"]
            num_seeds = exps[0].get("num_seeds", self.cfg.sensitivity_task.get("num_seeds", 1))
            seed_start = exps[0].get("seed_start", self.cfg.sensitivity_task.get("seed_start", 42))
            seeds = list(range(seed_start, seed_start + num_seeds))

            # 识别变参 key: 所有 combo_dict 的 key 的并集
            varying_keys = set()
            for exp in exps:
                varying_keys.update(exp["_meta"]["combo_dict"].keys())

            # 提取 base overrides (非变参部分, 去重保序)
            base_overrides = []
            seen_keys = set()
            for exp in exps:
                for ovr in exp["overrides"]:
                    # 解析 key=value 中的 key
                    ovr_key = ovr.split("=")[0] if "=" in ovr else ovr
                    if ovr_key not in varying_keys and ovr_key not in seen_keys:
                        base_overrides.append(ovr)
                        seen_keys.add(ovr_key)

            # 变参合并: key=[v1,v2,...] 格式
            merged_overrides = list(base_overrides)
            for key in sorted(varying_keys):
                # 收集该 key 的所有值 (去重保序)
                values = []
                seen_vals = set()
                for exp in exps:
                    val = exp["_meta"]["combo_dict"].get(key)
                    val_str = str(val)
                    if val_str not in seen_vals:
                        values.append(val)
                        seen_vals.add(val_str)
                # 合并为 key=[v1,v2,...]
                if len(values) == 1:
                    merged_overrides.append(f"{key}={values[0]}")
                else:
                    merged_overrides.append(f"{key}=[{','.join(str(v) for v in values)}]")

            cmd = format_reproduction_script(
                base_args=base_args,
                overrides=merged_overrides,
                seeds=seeds,
                group_name=group_name,
            )
            reproduction_scripts.append({"label": study_name, "command": cmd})

            if base_url:
                group_urls[study_name] = build_wandb_group_url(
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

    def _send_sensitivity_email(
        self,
        sweep_id: str,
        sweep,
        best_run,
        best_config_dict: dict,
        best_params_metrics: dict,
        param_grid_results: List[dict],
        image_paths: List[Tuple[Path, Path]],
        reproduction_scripts: list = None,
        group_urls: dict = None,
        eval_rank: int = 1,
    ):
        """发送参数敏感性邮件 (支持多图嵌入 + 多 PDF 附件)。"""
        import json as json_mod
        import os

        subject = f"📐 Sensitivity | {sweep.project} {sweep_id} rank{eval_rank}"

        # 构建 sweep URL
        base_url = os.getenv("WANDB_BASE_URL", "")
        sweep_url = build_wandb_sweep_url(base_url, self.cfg.wandb.entity, self.cfg.wandb.project, sweep_id) if base_url else ""

        # 提取 best run 元信息
        best_run_metadata = self.extract_run_metadata(best_run) if best_run else {}
        run_url = build_wandb_run_url(
            base_url, self.cfg.wandb.entity, self.cfg.wandb.project,
            best_run.id,
        ) if best_run and base_url else "N/A"

        rank_suffix = f"_rank{eval_rank}"
        eval_report_path = str(self.cfg.evaluate_task.report_path).format(sweep_id=sweep_id)
        sensitivity_report_path = Path(
            str(self.cfg.sensitivity_task.get("report_path", "logs/final_reports/sensitivity_{sweep_id}.json"))
            .format(sweep_id=f"{sweep_id}{rank_suffix}")
        ).resolve()

        # 收集 checkpoint 路径: 优先读取 eval 保存的 JSON, 回退到扫描
        report_dir = str(Path(eval_report_path).parent.resolve())
        eval_ckpt_map = self.load_eval_checkpoints(sweep_id, report_dir)
        rank_key = f"top-{eval_rank}"
        checkpoint_paths = []
        if eval_ckpt_map:
            checkpoint_paths = list(eval_ckpt_map.get(rank_key, []))
            log.info(f"📋 Loaded {len(checkpoint_paths)} eval checkpoints from JSON (rank_key={rank_key})")

        if not checkpoint_paths:
            project_log_dir = str(Path(self._full_cfg.paths.log_dir).resolve())
            group_name = f"eval/{sweep_id}/{rank_key}"
            fallback_ckpts = self.collect_checkpoint_paths(project_log_dir, group_name=group_name)
            num_seeds = self.cfg.evaluate_task.get("num_seeds", 0)
            checkpoint_paths = fallback_ckpts[-num_seeds:] if num_seeds else fallback_ckpts
            if checkpoint_paths:
                log.warning(
                    f"⚠️ Eval checkpoint JSON empty for {rank_key}; "
                    f"using latest {len(checkpoint_paths)} checkpoint(s) from group '{group_name}'."
                )
        workflow_info = {
            "sweep_id": sweep_id,
            "sweep_description": sweep.config.get("description", "N/A"),
            "best_run_config": json_mod.dumps(best_config_dict, indent=4, ensure_ascii=False),
            "report_label": "Sensitivity Report (JSON)",
            "report_json_path": str(sensitivity_report_path),
            "report_csv_path": "N/A",
            "log_dir": str(sensitivity_report_path.parent),
            "sweep_url": sweep_url,
            "best_run_host": best_run_metadata.get("host", "N/A"),
            "run_url": run_url,
            "best_run_created_at": best_run_metadata.get("created_at", "N/A"),
            "best_run_duration": best_run_metadata.get("duration", "N/A"),
            "checkpoint_paths": checkpoint_paths,
        }

        msg = build_sensitivity_email(
            notification_cfg=self.cfg.notification,
            subject=subject,
            best_params_metrics=best_params_metrics,
            param_grid_results=param_grid_results,
            image_paths=image_paths,
            workflow_info=workflow_info,
            reproduction_scripts=reproduction_scripts,
            group_urls=group_urls,
            eval_rank=eval_rank,
        )
        send_email_with_mimemultipart(self.cfg.notification, msg, subject=subject, mode="sensitivity")
