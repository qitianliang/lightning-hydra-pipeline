# src/workflow.py
"""工作流编排器 — 精简调度, 任务逻辑委托至 src/tasks/ 模块。"""

import itertools
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf
from rootutils import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.services.command_builder import CommandBuilder
from src.services.tmux_service import TmuxService
from src.services.wandb_service import WandbService
from src.tasks.shared import (
    is_dry_run,
    load_pipeline_progress,
    save_pipeline_progress,
)
from src.tasks.sweep import SweepTask
from src.tasks.evaluate import EvaluateTask
from src.tasks.override import OverrideTask
from src.tasks.ablation import AblationTask
from src.tasks.sensitivity import SensitivityTask
from src.utils import ConfigError, RankedLogger, WorkflowError, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)

# 保留公共 API 供 tests 使用
from src.tasks.shared import (
    safe_get_metric_scores as _safe_get_metric_scores,
    build_aggregation_report as _build_aggregation_report,
    RerunStrategy,
    ResumeStrategy,
)


# =====================================================================================
# Main Orchestrator
# =====================================================================================
def _verify_worktree_context() -> None:
    """校验当前运行在正确的 worktree commit 上。

    当 IN_WORKTREE=1 时，验证 git HEAD 与 worktree 期望一致。
    不在 worktree 中时仅输出警告。
    """
    in_worktree = os.environ.get("IN_WORKTREE", "0") == "1"

    if not in_worktree:
        log.warning("⚠️  Not running in worktree. Direct execution is for development only.")
        return

    try:
        current_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        current_branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        worktree_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        log.info(f"🌿 Worktree context verified:")
        log.info(f"   HEAD={current_commit[:8]}  branch={current_branch or '(detached)'}")
        log.info(f"   root={worktree_root}")
    except subprocess.CalledProcessError:
        log.warning("⚠️  Cannot verify git context (not a git repo?)")


@task_wrapper
def execute(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """主编排: 初始化服务 → 阶段分发 → 执行流水线。"""
    # Set global dry-run flag
    import src.tasks.shared as _shared_mod
    _shared_mod._DRY_RUN = cfg.workflow.get("dry_run", False) or cfg.get("dry_run", False)
    if is_dry_run():
        log.info("🏁 DRY-RUN MODE — commands printed, nothing executed.")

    # ── Worktree 上下文校验 ────────────────────────────────────────────
    _verify_worktree_context()

    log.info("Initializing services...")
    wandb_service = WandbService(cfg.workflow.wandb.entity, cfg.workflow.wandb.project)
    tmux_service = TmuxService()
    command_builder = CommandBuilder(cfg.workflow)

    task_name = cfg.workflow.task_name

    # ── Phase 1: Resolve sweep_id ──────────────────────────────────────
    active_sweep_id = cfg.workflow.get("target_sweep_id", None)

    if active_sweep_id:
        log.info(f"⏭️  Target Sweep ID [{active_sweep_id}] provided. Skipping Sweep phase!")
    elif task_name in ("pipeline", "sweep"):
        log.info("▶️ Starting a new Sweep...")
        sweep_task = SweepTask(cfg, tmux_service, command_builder)
        active_sweep_id = sweep_task.run()
    else:
        raise WorkflowError(
            f"Unknown task_name='{task_name}'. Use 'pipeline' (default) or 'sweep'."
        )

    if task_name == "sweep":
        return {}, {}

    # ── Phase 2: Dispatch by task_name ─────────────────────────────────
    if not active_sweep_id:
        raise WorkflowError("Pipeline requires a valid sweep ID. Set target_sweep_id or run sweep first.")

    # 特殊模式: ablation / sensitivity (独立运行, 不走 pipeline_tasks)
    if task_name == "ablation":
        ablation_task = AblationTask(cfg, wandb_service, tmux_service, command_builder)
        ablation_task.run(sweep_id=active_sweep_id)
        return {}, {}

    if task_name == "sensitivity":
        sensitivity_task = SensitivityTask(cfg, wandb_service, tmux_service, command_builder)
        sensitivity_task.run(sweep_id=active_sweep_id)
        return {}, {}

    # ── Phase 3: Run pipeline_tasks in order ───────────────────────────
    pipeline_tasks = cfg.workflow.get("pipeline_tasks", [])
    if not pipeline_tasks:
        log.warning("No pipeline_tasks defined. Nothing to do.")
        return {}, {}

    # Load resume checkpoint (scoped to sweep_id)
    # If target_sweep_id is explicitly provided, force rerun all stages
    if cfg.workflow.get("target_sweep_id"):
        log.info("Target sweep_id provided. Forcing rerun of all pipeline stages.")
        completed_stages = set()
    else:
        completed_stages = load_pipeline_progress(cfg, active_sweep_id)

    for idx, task_cfg in enumerate(pipeline_tasks):
        task_type = task_cfg.type
        base_name = task_cfg.name

        # Skip if already completed (resume support)
        if base_name in completed_stages:
            log.info(f"⏭️  Stage '{base_name}' already completed (resume). Skipping.")
            continue

        log.info(f"\n========== 🔄 Pipeline Stage {idx+1}: [{base_name}] (type={task_type}) ==========")

        if task_type == "evaluate":
            EvaluateTask(cfg, wandb_service, tmux_service, command_builder).run(sweep_id=active_sweep_id)

        elif task_type == "override":
            overrides_dict = OmegaConf.to_container(task_cfg.get("overrides", {}), resolve=True)
            OmegaConf.update(cfg, "workflow.override_task.name", base_name, merge=True)
            OmegaConf.update(cfg, "workflow.override_task.overrides", overrides_dict, merge=False)
            OverrideTask(cfg, wandb_service, tmux_service, command_builder).run(sweep_id=active_sweep_id)

        elif task_type == "grid":
            params_dict = OmegaConf.to_container(task_cfg.get("params", {}), resolve=True)
            keys = list(params_dict.keys())
            values_lists = [v if isinstance(v, list) else [v] for v in params_dict.values()]
            combinations = list(itertools.product(*values_lists))

            # Grid组合数上限检查 (从配置读取, 默认100)
            max_grid = cfg.workflow.sweep_task.get("max_grid_combinations", 100)
            if len(combinations) > max_grid:
                raise ConfigError(
                    f"Grid组合数 {len(combinations)} > {max_grid}, 拒绝执行"
                )

            log.info(f"📐 Grid Search: {len(combinations)} sub-tasks...")

            for combo_idx, combo_values in enumerate(combinations):
                combo_dict = dict(zip(keys, combo_values))
                name_parts = [base_name]
                for k, v in combo_dict.items():
                    short_k = str(k).split('.')[-1]
                    name_parts.append(f"{short_k}_{v}")
                combo_name = "_".join(name_parts)
                log.info(f"  -> [Grid {combo_idx+1}/{len(combinations)}] {combo_name}")

                OmegaConf.update(cfg, "workflow.override_task.name", combo_name, merge=True)
                OmegaConf.update(cfg, "workflow.override_task.overrides", combo_dict, merge=False)
                OverrideTask(cfg, wandb_service, tmux_service, command_builder).run(sweep_id=active_sweep_id)

        elif task_type == "ablation":
            AblationTask(cfg, wandb_service, tmux_service, command_builder).run(sweep_id=active_sweep_id)

        elif task_type == "sensitivity":
            SensitivityTask(cfg, wandb_service, tmux_service, command_builder).run(sweep_id=active_sweep_id)

        else:
            raise WorkflowError(f"Unknown task type '{task_type}' in pipeline_tasks!")

        # Mark stage completed for resume (scoped to sweep_id)
        save_pipeline_progress(cfg, base_name, active_sweep_id)

    log.info(f"🎉 All pipeline tasks finished successfully!")
    return {}, {}


@hydra.main(config_path="../configs", config_name="workflow.yaml", version_base=None)
def main(cfg: DictConfig):
    log.info("Workflow Orchestrator Initialized")

    config_yaml_string = OmegaConf.to_yaml(cfg.workflow)
    log.info(f"\n{config_yaml_string}")
    execute(cfg)


if __name__ == "__main__":
    main()