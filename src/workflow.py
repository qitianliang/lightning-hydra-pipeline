# src/workflow.py
"""工作流编排器 — 精简调度, 任务逻辑委托至 src/tasks/ 模块。"""

import os
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
from src.tasks.ablation import AblationTask
from src.tasks.sensitivity import SensitivityTask
from src.utils import RankedLogger, WorkflowError, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)

# 保留公共 API 供 tests 使用
from src.tasks.shared import (
    safe_get_metric_scores as _safe_get_metric_scores,
    RerunStrategy,
    ResumeStrategy,
)


# =====================================================================================
# Main Orchestrator
# =====================================================================================
@task_wrapper
def execute(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """主编排: 初始化服务 → 阶段分发 → 执行流水线。"""
    # Set global dry-run flag
    import src.tasks.shared as _shared_mod
    _shared_mod._DRY_RUN = cfg.workflow.get("dry_run", False) or cfg.get("dry_run", False)
    if is_dry_run():
        log.info("🏁 DRY-RUN MODE — commands printed, nothing executed.")

    log.info("Initializing services...")
    wandb_service = WandbService(cfg.workflow.wandb.entity, cfg.workflow.wandb.project)
    tmux_service = TmuxService()
    command_builder = CommandBuilder(cfg.workflow)

    from src.services.sandbox_service import SandboxService
    snapshot_cfg = cfg.workflow.get("snapshot", {})
    sandbox_service = SandboxService(
        cfg.workflow.wandb.entity,
        cfg.workflow.wandb.project,
        allow_source_overwrite=snapshot_cfg.get("allow_source_overwrite", False),
        allow_legacy_config_fallbacks=snapshot_cfg.get("legacy_config_fallbacks", True),
    )

    task_name = cfg.workflow.task_name

    # ── Phase 1: Resolve sweep_id ──────────────────────────────────────
    active_sweep_id = cfg.workflow.get("target_sweep_id", None)

    if active_sweep_id:
        log.info(f"⏭️  Target Sweep ID [{active_sweep_id}] provided. Skipping Sweep phase!")
    elif task_name in ("pipeline", "sweep"):
        log.info("▶️ Starting a new Sweep...")
        sweep_task = SweepTask(cfg, tmux_service, command_builder)
        active_sweep_id = sweep_task.run()
        # Publish current source code to W&B for real sweeps only. Dry-run must stay side-effect free.
        if is_dry_run():
            log.info(f"[DRY-RUN] Active sweep id: {active_sweep_id}")
            log.info("[DRY-RUN] Skipping source artifact publish.")
        else:
            sandbox_service.publish_source_to_wandb(active_sweep_id)
    elif task_name == "agent":
        log.info(f"▶️ Starting Agent mode for Sweep ID: {active_sweep_id}")
        if not active_sweep_id:
            raise WorkflowError("Agent mode requires target_sweep_id.")
        sandbox_dir = sandbox_service.setup_sandbox(active_sweep_id, task_name="agent")
        log.info(f"🚀 Sandbox created at {sandbox_dir}. Launching wandb agent...")
        import subprocess
        sweep_path = f"{cfg.workflow.wandb.entity}/{cfg.workflow.wandb.project}/{active_sweep_id}"
        subprocess.run(["wandb", "agent", sweep_path], cwd=sandbox_dir)
        return {}, {}
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
        ablation_task = AblationTask(cfg, wandb_service, tmux_service, command_builder, sandbox_service)
        ablation_task.run(sweep_id=active_sweep_id)
        return {}, {}

    if task_name == "sensitivity":
        sensitivity_task = SensitivityTask(cfg, wandb_service, tmux_service, command_builder, sandbox_service)
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
            EvaluateTask(cfg, wandb_service, tmux_service, command_builder, sandbox_service).run(sweep_id=active_sweep_id)

        elif task_type == "ablation":
            AblationTask(cfg, wandb_service, tmux_service, command_builder, sandbox_service).run(sweep_id=active_sweep_id)

        elif task_type == "sensitivity":
            SensitivityTask(cfg, wandb_service, tmux_service, command_builder, sandbox_service).run(sweep_id=active_sweep_id)

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