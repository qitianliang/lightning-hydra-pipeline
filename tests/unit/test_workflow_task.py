import json
import subprocess  # nosec B404
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import yaml
from omegaconf import DictConfig, open_dict

# 導入 wandb 類型用於 spec
from wandb.apis.public import Run, Sweep

# 導入我們的生產代碼
from src.services.command_builder import CommandBuilder
from src.tasks.evaluate import EvaluateTask
from src.tasks.sweep import SweepTask
from src.tasks.override import OverrideTask
from src.tasks.ablation import AblationTask
from src.tasks.sensitivity import SensitivityTask
from src.tasks.shared import is_dry_run
from src.utils.exceptions import SweepError, WorkflowError


class TestEvaluateTask:
    """針對 EvaluateTask 類的集成測試套件。"""

    def test_evaluate_task_rerun_success_flow(
        self,
        cfg_workflow: DictConfig,
        mock_wandb_service: MagicMock,
        mock_tmux_service: MagicMock,
        mocker: MagicMock,
        tmp_path: Path,
    ):
        """測試 `rerun` 模式下的成功執行流程（top_n=1）。"""
        # ===============================================
        # 1. Arrange (準備測試環境和 mock 返回值)
        # ===============================================

        # Disable dry_run so the actual flow executes
        with open_dict(cfg_workflow):
            cfg_workflow.workflow.dry_run = False
            cfg_workflow.workflow.evaluate_task.mode = "rerun"
            cfg_workflow.workflow.evaluate_task.top_n = 1

        mock_tmux_service.session_exists.return_value = False

        # 準備假的狀態文件
        status_dir = Path(cfg_workflow.workflow.general.status_dir)
        status_dir.mkdir(parents=True, exist_ok=True)
        latest_status_file = Path(cfg_workflow.workflow.general.latest_status_file)
        actual_status_file = status_dir / "status_test_sweep_123.yaml"
        status_content = {
            "sweep_id": "test_sweep_123",
            "sweep_path": "DSLog/mnist-workflow-demo/test_sweep_123",
            "tmux_session_name": "test_sweep_session",
        }
        with open(actual_status_file, "w") as f:
            yaml.dump(status_content, f)
        latest_status_file.symlink_to(actual_status_file)

        # 配置 mock Sweep 對象
        mock_sweep = MagicMock(spec=Sweep, state="FINISHED")
        type(mock_sweep).id = PropertyMock(return_value="test_sweep_123")
        type(mock_sweep).project = PropertyMock(return_value="test-project")
        type(mock_sweep).name = PropertyMock(return_value="test-sweep-name")
        mock_sweep.config = {}

        # 配置 mock Best Run 對象
        mock_best_run = MagicMock(spec=Run)
        mock_best_run.config = {"model.lr": 0.01, "data.batch_size": 64}
        type(mock_best_run).id = PropertyMock(return_value="best_run_abc")
        type(mock_best_run).name = PropertyMock(return_value="best-run-name")
        mock_best_run.summary = {"test/acc": 0.97, "test/loss": 0.11}

        mock_wandb_service.get_sweep.return_value = mock_sweep
        # EvaluateTask now uses find_top_n_runs
        mock_wandb_service.find_top_n_runs.return_value = [mock_best_run]

        # 模擬評估後產生的新 runs
        mock_eval_run_1 = MagicMock(spec=Run, summary={"test/acc": 0.98, "test/loss": 0.1})
        mock_eval_run_2 = MagicMock(spec=Run, summary={"test/acc": 0.96, "test/loss": 0.12})
        new_runs = [mock_eval_run_1, mock_eval_run_2]

        # 模擬 delete_runs_in_group 和 get_runs_by_group 之間的交互
        call_count = {"delete": 0}

        def delete_side_effect(group_name: str) -> int:
            call_count["delete"] += 1
            return 0

        mock_wandb_service.delete_runs_in_group.side_effect = delete_side_effect
        mock_wandb_service.get_runs_by_group.return_value = new_runs

        # Patch DRY_RUN to False
        mocker.patch("src.tasks.shared._DRY_RUN", False)

        # ===============================================
        # 2. Act (執行被測試的代碼)
        # ===============================================

        mock_command_builder = MagicMock(spec=CommandBuilder)
        mock_command_builder.build_training_run_command.return_value = "python src/train.py seed=42"
        evaluate_task = EvaluateTask(
            cfg_workflow, mock_wandb_service, mock_tmux_service, mock_command_builder
        )
        evaluate_task.run()

        # ===============================================
        # 3. Assert (斷言行為是否符合預期)
        # ===============================================

        # 斷言 find_top_n_runs 被調用
        mock_wandb_service.find_top_n_runs.assert_called_once()

        # 斷言報告已生成且內容正確
        report_path = Path(
            cfg_workflow.workflow.evaluate_task.report_path.format(sweep_id="test_sweep_123")
        )
        assert report_path.exists()

        with open(report_path) as f:
            report_data = json.load(f)

        assert report_data["sweep_id"] == "test_sweep_123"
        # Per-rank format: evaluation_summary.top-1.best_run_name
        assert "top-1" in report_data["evaluation_summary"]
        assert report_data["evaluation_summary"]["top-1"]["best_run_name"] == "best-run-name"
        assert "test/acc" in report_data["evaluation_summary"]["top-1"]["metrics"]
        assert report_data["evaluation_summary"]["top-1"]["metrics"]["test/acc"]["mean"] == pytest.approx(0.97)
        # top_n field present
        assert report_data.get("top_n") == 1


class TestSweepTask:
    """Integration test suite for the SweepTask class."""

    def test_sweep_task_success_flow(
        self,
        cfg_workflow: DictConfig,
        mock_tmux_service: MagicMock,
        mocker: MagicMock,
    ):
        """Tests the successful, "happy path" execution of the SweepTask."""
        # ===============================================
        # 1. Arrange
        # ===============================================

        # 1a. 創建一個完全功能的 mock subprocess 模塊
        mock_subprocess = MagicMock()
        mock_proc = MagicMock(spec=subprocess.CompletedProcess)
        mock_proc.stdout = "Some leading text... wandb agent DSLog/mnist-workflow-demo/test_sweep_abc ...and some trailing text."
        mock_proc.stderr = ""
        mock_subprocess.run.return_value = mock_proc
        mock_subprocess.CalledProcessError = subprocess.CalledProcessError

        # 1b. Patch DRY_RUN to False for SweepTask test
        mocker.patch("src.tasks.shared._DRY_RUN", False)
        mocker.patch("src.tasks.sweep.yaml.dump")
        mocker.patch("src.tasks.sweep.os.symlink")

        # 1c. 配置 mock_tmux_service
        mock_tmux_service.session_exists.return_value = False

        # ===============================================
        # 2. Act
        # ===============================================
        mock_command_builder = MagicMock(spec=CommandBuilder)
        task = SweepTask(
            cfg_workflow,
            mock_tmux_service,
            mock_command_builder,
            subprocess_module=mock_subprocess,
        )
        task.run()

        # ===============================================
        # 3. Assert
        # ===============================================

        assert mock_subprocess.run.called, "'subprocess.run' was not called!"
        mock_command_builder.build_wandb_sweep_command.assert_called_once()

        expected_session_name = "mnist_workflow_test_sweep_abc"
        mock_tmux_service.session_exists.assert_called_once_with(expected_session_name)
        mock_tmux_service.create_workers_session.assert_called_once()
        mock_tmux_service.wait_for_session_to_close.assert_called_once()

    def test_sweep_task_handles_creation_failure(
        self,
        cfg_workflow: DictConfig,
        mock_tmux_service: MagicMock,
        mocker: MagicMock,
    ):
        mocker.patch("src.tasks.shared._DRY_RUN", False)
        mock_command_builder = MagicMock(spec=CommandBuilder)
        mock_subprocess = MagicMock()
        mock_subprocess.run.side_effect = subprocess.CalledProcessError(1, "cmd")
        mock_subprocess.CalledProcessError = subprocess.CalledProcessError

        with pytest.raises(SweepError):
            task = SweepTask(cfg_workflow, mock_tmux_service, mock_command_builder, subprocess_module=mock_subprocess)
            task.run()

    def test_sweep_task_handles_parsing_failure(
        self,
        cfg_workflow: DictConfig,
        mock_tmux_service: MagicMock,
        mocker: MagicMock,
    ):
        mocker.patch("src.tasks.shared._DRY_RUN", False)
        mock_proc = MagicMock(
            spec=subprocess.CompletedProcess,
            stdout="Some other unrelated output",
            stderr="",
        )
        mock_subprocess = MagicMock()
        mock_subprocess.run.return_value = mock_proc
        mock_subprocess.CalledProcessError = subprocess.CalledProcessError

        with pytest.raises(SweepError):
            mock_command_builder = MagicMock(spec=CommandBuilder)
            task = SweepTask(cfg_workflow, mock_tmux_service, mock_command_builder, subprocess_module=mock_subprocess)
            task.run()


class TestTopNValidation:
    """Test top_n boundary enforcement (workflow_rules: max 3)."""

    def test_top_n_exceeds_max_raises(
        self,
        cfg_workflow: DictConfig,
        mock_wandb_service: MagicMock,
        mock_tmux_service: MagicMock,
        mocker: MagicMock,
    ):
        """top_n > 3 should sys.exit."""
        with open_dict(cfg_workflow):
            cfg_workflow.workflow.dry_run = False
            cfg_workflow.workflow.evaluate_task.top_n = 5

        mock_command_builder = MagicMock(spec=CommandBuilder)
        evaluate_task = EvaluateTask(
            cfg_workflow, mock_wandb_service, mock_tmux_service, mock_command_builder
        )

        # Need a valid sweep to get past Phase 2.1
        mock_sweep = MagicMock(spec=Sweep, state="FINISHED")
        type(mock_sweep).id = PropertyMock(return_value="test_sweep")
        type(mock_sweep).project = PropertyMock(return_value="test-proj")
        type(mock_sweep).name = PropertyMock(return_value="test-name")
        mock_wandb_service.get_sweep.return_value = mock_sweep

        with pytest.raises(WorkflowError):
            evaluate_task.run(sweep_id="test_sweep")


class TestAblationTaskParallel:
    """Test AblationTask parallel execution with multiple components."""

    def test_ablation_parallel_builds_experiments(
        self,
        cfg_workflow: DictConfig,
        mock_wandb_service: MagicMock,
        mock_tmux_service: MagicMock,
        mocker: MagicMock,
        tmp_path: Path,
    ):
        """Test parallel ablation builds correct experiment groups from components list."""
        with open_dict(cfg_workflow):
            cfg_workflow.workflow.dry_run = False
            cfg_workflow.workflow.task_name = "ablation"
            cfg_workflow.workflow.ablation_task.parallel = True
            cfg_workflow.workflow.ablation_task.timeout_secs = 600
            cfg_workflow.workflow.ablation_task.num_seeds = 1
            cfg_workflow.workflow.ablation_task.components = [
                {"name": "no_lin1", "overrides": {"model.net.lin1_size": 32}},
                {"name": "no_lin2", "overrides": {"model.net.lin2_size": 32}},
            ]

        mock_tmux_service.session_exists.return_value = False
        mocker.patch("src.tasks.shared._DRY_RUN", False)

        mock_sweep = MagicMock(spec=Sweep, state="FINISHED")
        type(mock_sweep).id = PropertyMock(return_value="test_sweep")
        type(mock_sweep).project = PropertyMock(return_value="test-proj")
        type(mock_sweep).name = PropertyMock(return_value="test-name")
        mock_sweep.config = {}

        mock_best_run = MagicMock(spec=Run)
        mock_best_run.config = {"model.lr": 0.01}
        type(mock_best_run).id = PropertyMock(return_value="best_run_abc")
        type(mock_best_run).name = PropertyMock(return_value="best-run-name")
        mock_best_run.summary = {"test/acc": 0.95}

        mock_wandb_service.get_sweep.return_value = mock_sweep
        mock_wandb_service.find_best_run.return_value = mock_best_run
        mock_wandb_service.delete_runs_in_group.return_value = 0
        mock_wandb_service.get_runs_by_group.return_value = []

        # Prepare evaluate report
        report_dir = tmp_path / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = report_dir / "final_report.json"
        report_data = {
            "evaluation_summary": {
                "test/acc": {"mean": 0.95, "std": 0.01},
                "test/loss": {"mean": 0.15, "std": 0.02},
            }
        }
        with open(report_file, "w") as f:
            json.dump(report_data, f)

        with open_dict(cfg_workflow):
            cfg_workflow.workflow.evaluate_task.report_path = str(report_file)

        mock_command_builder = MagicMock(spec=CommandBuilder)
        mock_command_builder.build_training_run_command.return_value = "python src/train.py seed=42"

        task = AblationTask(
            cfg_workflow, mock_wandb_service, mock_tmux_service, mock_command_builder
        )
        task.run(sweep_id="test_sweep")

        # Assert parallel session was created (create_workers_session called)
        mock_tmux_service.create_workers_session.assert_called_once()
        call_args = mock_tmux_service.create_workers_session.call_args
        session_name = call_args[0][0]
        assert "ablation" in session_name

        # Assert delete_runs_in_group was called for each component
        assert mock_wandb_service.delete_runs_in_group.call_count == 2

    def test_ablation_serial_mode(
        self,
        cfg_workflow: DictConfig,
        mock_wandb_service: MagicMock,
        mock_tmux_service: MagicMock,
        mocker: MagicMock,
        tmp_path: Path,
    ):
        """Test serial ablation executes components one by one."""
        with open_dict(cfg_workflow):
            cfg_workflow.workflow.dry_run = True
            cfg_workflow.workflow.ablation_task.parallel = False
            cfg_workflow.workflow.ablation_task.components = [
                {"name": "no_lin1", "overrides": {"model.net.lin1_size": 32}},
            ]

        mocker.patch("src.tasks.shared._DRY_RUN", True)

        mock_sweep = MagicMock(spec=Sweep, state="FINISHED")
        type(mock_sweep).id = PropertyMock(return_value="test_sweep")
        type(mock_sweep).project = PropertyMock(return_value="test-proj")
        type(mock_sweep).name = PropertyMock(return_value="test-name")
        mock_sweep.config = {}

        mock_best_run = MagicMock(spec=Run)
        mock_best_run.config = {}
        type(mock_best_run).id = PropertyMock(return_value="best_run_abc")
        type(mock_best_run).name = PropertyMock(return_value="best-run-name")

        mock_wandb_service.get_sweep.return_value = mock_sweep
        mock_wandb_service.find_best_run.return_value = mock_best_run

        mock_command_builder = MagicMock(spec=CommandBuilder)
        mock_command_builder.build_training_run_command.return_value = "python src/train.py seed=42"

        task = AblationTask(
            cfg_workflow, mock_wandb_service, mock_tmux_service, mock_command_builder
        )
        # Should complete without error in dry-run
        task.run(sweep_id="dry-run-test")

    def test_ablation_dry_run_expands_default_components_without_side_effects(
        self,
        cfg_workflow: DictConfig,
        mock_wandb_service: MagicMock,
        mock_tmux_service: MagicMock,
        mocker: MagicMock,
        tmp_path: Path,
    ):
        """Dry-run ablation should exercise component expansion without W&B mutations/tmux."""
        mocker.patch("src.tasks.shared._DRY_RUN", True)

        report_file = tmp_path / "reports" / "optimized_results_test_sweep.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps({
            "evaluation_summary": {
                "top-1": {
                    "metrics": {
                        "test/acc": {"mean": 0.95, "std": 0.01},
                        "test/loss": {"mean": 0.15, "std": 0.02},
                    }
                }
            }
        }))

        with open_dict(cfg_workflow):
            cfg_workflow.workflow.dry_run = True
            cfg_workflow.workflow.task_name = "ablation"
            cfg_workflow.workflow.evaluate_task.report_path = str(report_file)
            cfg_workflow.workflow.ablation_task.parallel = True
            cfg_workflow.workflow.ablation_task.num_seeds = 3
            cfg_workflow.workflow.ablation_task.seed_start = 42
            cfg_workflow.workflow.ablation_task.report_path = str(tmp_path / "reports" / "ablation_{sweep_id}.json")
            cfg_workflow.workflow.ablation_task.components = [
                {"name": "no_lin1_bn", "overrides": {"model.net.lin1_size": 32}},
                {"name": "no_lin2_bn", "overrides": {"model.net.lin2_size": 32}},
            ]

        mock_sweep = MagicMock(spec=Sweep, state="FINISHED")
        type(mock_sweep).id = PropertyMock(return_value="test_sweep")
        type(mock_sweep).project = PropertyMock(return_value="test-proj")
        type(mock_sweep).name = PropertyMock(return_value="test-name")
        mock_sweep.config = {}

        mock_best_run = MagicMock(spec=Run)
        mock_best_run.config = {"model.optimizer.lr": 0.01, "data.batch_size": 64}
        type(mock_best_run).id = PropertyMock(return_value="best_run_abc")
        type(mock_best_run).name = PropertyMock(return_value="best-run-name")

        mock_wandb_service.get_sweep.return_value = mock_sweep
        mock_wandb_service.find_best_run.return_value = mock_best_run

        def build_command(overrides, seed, group_name, cwd=""):
            return f"python src/train.py {' '.join(overrides)} seed={seed} logger.wandb.group={group_name}"

        mock_command_builder = MagicMock(spec=CommandBuilder)
        mock_command_builder.build_training_run_command.side_effect = build_command

        task = AblationTask(
            cfg_workflow, mock_wandb_service, mock_tmux_service, mock_command_builder
        )
        task.run(sweep_id="test_sweep")

        assert mock_command_builder.build_training_run_command.call_count == 6
        calls = mock_command_builder.build_training_run_command.call_args_list
        groups = [call.kwargs["group_name"] for call in calls]
        seeds = [call.kwargs["seed"] for call in calls]
        override_sets = [call.kwargs["overrides"] for call in calls]

        assert groups.count("ablation/test_sweep/no_lin1_bn") == 3
        assert groups.count("ablation/test_sweep/no_lin2_bn") == 3
        assert sorted(set(seeds)) == [42, 43, 44]
        assert any("model.net.lin1_size=32" in overrides for overrides in override_sets)
        assert any("model.net.lin2_size=32" in overrides for overrides in override_sets)
        assert all("model.optimizer.lr=0.01" in overrides for overrides in override_sets)
        assert all("data.batch_size=64" in overrides for overrides in override_sets)

        mock_wandb_service.delete_runs_in_group.assert_not_called()
        mock_tmux_service.create_workers_session.assert_not_called()
        assert not (tmp_path / "reports" / "ablation_test_sweep.json").exists()

    def test_ablation_email_points_to_ablation_report_directory(
        self,
        cfg_workflow: DictConfig,
        mock_wandb_service: MagicMock,
        mock_tmux_service: MagicMock,
        mocker: MagicMock,
        tmp_path: Path,
    ):
        """Ablation emails should link to ablation reports, not evaluate reports."""
        with open_dict(cfg_workflow):
            cfg_workflow.workflow.evaluate_task.report_path = str(tmp_path / "reports" / "eval_{sweep_id}.json")
            cfg_workflow.workflow.ablation_task.report_path = str(tmp_path / "reports" / "ablation_{sweep_id}")

        mock_sweep = MagicMock(spec=Sweep)
        type(mock_sweep).id = PropertyMock(return_value="test_sweep")
        type(mock_sweep).project = PropertyMock(return_value="test-proj")
        mock_sweep.config = {"description": "debug sweep"}

        mock_best_run = MagicMock(spec=Run)
        type(mock_best_run).id = PropertyMock(return_value="best_run_abc")
        mock_best_run.config = {}

        captured = {}

        def capture_email(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        mocker.patch("src.tasks.ablation.build_ablation_email", side_effect=capture_email)
        send_mock = mocker.patch("src.tasks.ablation.send_email_with_mimemultipart")

        task = AblationTask(
            cfg_workflow, mock_wandb_service, mock_tmux_service, MagicMock(spec=CommandBuilder)
        )
        task.load_eval_checkpoints = MagicMock(return_value={})
        task.collect_checkpoint_paths = MagicMock(return_value=[])

        task._send_ablation_email(
            sweep_id="test_sweep",
            sweep=mock_sweep,
            best_run=mock_best_run,
            best_config_dict={"model.optimizer.lr": "0.01"},
            full_model_metrics={"test/acc": {"mean": 0.95, "std": 0.01}},
            ablation_results=[{"name": "no_lin1_bn", "metrics": {"test/acc": {"mean": 0.9, "std": 0.02}}}],
            reproduction_scripts=[],
            group_urls={},
        )

        workflow_info = captured["workflow_info"]
        assert workflow_info["report_label"] == "Ablation Report Directory"
        assert workflow_info["report_json_path"].endswith("reports/ablation_test_sweep")
        assert "eval_test_sweep" not in workflow_info["report_json_path"]
        assert workflow_info["report_csv_path"] == "N/A"
        send_mock.assert_called_once()


class TestSensitivityTaskList:
    """Test SensitivityTask with sensitivities list format."""

    def test_sensitivity_parses_sensitivities_list(
        self,
        cfg_workflow: DictConfig,
        mock_wandb_service: MagicMock,
        mock_tmux_service: MagicMock,
        mocker: MagicMock,
    ):
        """Test sensitivities list is correctly parsed into studies."""
        mocker.patch("src.tasks.shared._DRY_RUN", True)

        mock_sweep = MagicMock(spec=Sweep, state="FINISHED")
        type(mock_sweep).id = PropertyMock(return_value="test_sweep")
        type(mock_sweep).project = PropertyMock(return_value="test-proj")
        type(mock_sweep).name = PropertyMock(return_value="test-name")
        mock_sweep.config = {}

        mock_best_run = MagicMock(spec=Run)
        mock_best_run.config = {}
        type(mock_best_run).id = PropertyMock(return_value="best_run_abc")
        type(mock_best_run).name = PropertyMock(return_value="best-run-name")

        mock_wandb_service.get_sweep.return_value = mock_sweep
        mock_wandb_service.find_best_run.return_value = mock_best_run

        mock_command_builder = MagicMock(spec=CommandBuilder)
        mock_command_builder.build_training_run_command.return_value = "python src/train.py seed=42"

        with open_dict(cfg_workflow):
            cfg_workflow.workflow.dry_run = True
            cfg_workflow.workflow.sensitivity_task.num_seeds = 1
            cfg_workflow.workflow.sensitivity_task.sensitivities = [
                {
                    "name": "width_sensitivity",
                    "param_grid": {"model.net.lin1_size": [32, 64]},
                    "axis_labels": {"model.net.lin1_size": "First Layer Size"},
                },
                {
                    "name": "lr_sensitivity",
                    "param_grid": {"optimizer.lr": [0.001, 0.01]},
                    "axis_labels": {"optimizer.lr": "Learning Rate"},
                },
            ]

        task = SensitivityTask(
            cfg_workflow, mock_wandb_service, mock_tmux_service, mock_command_builder
        )
        # Mock ensure_evaluate_results to avoid full evaluate flow
        mocker.patch.object(
            type(task), "ensure_evaluate_results",
            return_value={"evaluation_summary": {"test/acc": {"mean": 0.95, "std": 0.01}}}
        )

        task.run(sweep_id="test_sweep")

        # In dry-run, build_training_run_command should be called for each combo
        # 2 width combos + 2 lr combos = 4
        assert mock_command_builder.build_training_run_command.call_count == 4

    def test_sensitivity_dry_run_expands_default_studies_without_side_effects(
        self,
        cfg_workflow: DictConfig,
        mock_wandb_service: MagicMock,
        mock_tmux_service: MagicMock,
        mocker: MagicMock,
        tmp_path: Path,
    ):
        """Dry-run sensitivity should expand the default 2D+1D studies fully."""
        mocker.patch("src.tasks.shared._DRY_RUN", True)

        mock_sweep = MagicMock(spec=Sweep, state="FINISHED")
        type(mock_sweep).id = PropertyMock(return_value="test_sweep")
        type(mock_sweep).project = PropertyMock(return_value="test-proj")
        type(mock_sweep).name = PropertyMock(return_value="test-name")
        mock_sweep.config = {}

        mock_best_run = MagicMock(spec=Run)
        mock_best_run.config = {"model.optimizer.lr": 0.01, "data.batch_size": 64}
        type(mock_best_run).id = PropertyMock(return_value="best_run_abc")
        type(mock_best_run).name = PropertyMock(return_value="best-run-name")

        mock_wandb_service.get_sweep.return_value = mock_sweep
        mock_wandb_service.find_best_run.return_value = mock_best_run

        def build_command(overrides, seed, group_name, cwd=""):
            return f"python src/train.py {' '.join(overrides)} seed={seed} logger.wandb.group={group_name}"

        mock_command_builder = MagicMock(spec=CommandBuilder)
        mock_command_builder.build_training_run_command.side_effect = build_command

        report_path = tmp_path / "reports" / "sensitivity_test_sweep.json"
        with open_dict(cfg_workflow):
            cfg_workflow.workflow.dry_run = True
            cfg_workflow.workflow.task_name = "sensitivity"
            cfg_workflow.workflow.sensitivity_task.num_seeds = 3
            cfg_workflow.workflow.sensitivity_task.seed_start = 42
            cfg_workflow.workflow.sensitivity_task.report_path = str(report_path)
            cfg_workflow.workflow.sensitivity_task.sensitivities = [
                {
                    "name": "width_sensitivity",
                    "param_grid": {
                        "model.net.lin1_size": [32, 64, 128],
                        "model.net.lin2_size": [32, 64, 128],
                    },
                    "axis_labels": {
                        "model.net.lin1_size": "First Layer Size",
                        "model.net.lin2_size": "Second Layer Size",
                    },
                },
                {
                    "name": "lr_sensitivity",
                    "param_grid": {"model.optimizer.lr": [0.0001, 0.001, 0.01]},
                    "axis_labels": {"model.optimizer.lr": "Learning Rate"},
                },
            ]

        task = SensitivityTask(
            cfg_workflow, mock_wandb_service, mock_tmux_service, mock_command_builder
        )
        mocker.patch.object(
            type(task),
            "ensure_evaluate_results",
            return_value={
                "evaluation_summary": {
                    "top-1": {
                        "metrics": {
                            "test/acc": {"mean": 0.95, "std": 0.01},
                            "test/loss": {"mean": 0.15, "std": 0.02},
                        }
                    }
                }
            },
        )

        task.run(sweep_id="test_sweep")

        # width_sensitivity: 3x3 combos; lr_sensitivity: 3 combos; each has 3 seeds.
        assert mock_command_builder.build_training_run_command.call_count == 36
        calls = mock_command_builder.build_training_run_command.call_args_list
        groups = [call.kwargs["group_name"] for call in calls]
        seeds = [call.kwargs["seed"] for call in calls]
        override_sets = [call.kwargs["overrides"] for call in calls]

        assert groups.count("sensitivity/test_sweep/width_sensitivity") == 27
        assert groups.count("sensitivity/test_sweep/lr_sensitivity") == 9
        assert sorted(set(seeds)) == [42, 43, 44]
        assert any(
            "model.net.lin1_size=32" in overrides and "model.net.lin2_size=32" in overrides
            for overrides in override_sets
        )
        assert any(
            "model.net.lin1_size=128" in overrides and "model.net.lin2_size=128" in overrides
            for overrides in override_sets
        )
        assert any("model.optimizer.lr=0.0001" in overrides for overrides in override_sets)
        assert any("model.optimizer.lr=0.001" in overrides for overrides in override_sets)
        assert any("model.optimizer.lr=0.01" in overrides for overrides in override_sets)
        assert all("data.batch_size=64" in overrides for overrides in override_sets)

        mock_wandb_service.delete_runs_in_group.assert_not_called()
        mock_tmux_service.create_workers_session.assert_not_called()
        assert not report_path.exists()

    def test_sensitivity_email_points_to_sensitivity_report_json(
        self,
        cfg_workflow: DictConfig,
        mock_wandb_service: MagicMock,
        mock_tmux_service: MagicMock,
        mocker: MagicMock,
        tmp_path: Path,
    ):
        """Sensitivity emails should link to the sensitivity report, not the eval report."""
        with open_dict(cfg_workflow):
            cfg_workflow.workflow.evaluate_task.report_path = str(tmp_path / "reports" / "eval_{sweep_id}.json")
            cfg_workflow.workflow.sensitivity_task.report_path = str(tmp_path / "reports" / "sensitivity_{sweep_id}.json")

        mock_sweep = MagicMock(spec=Sweep)
        type(mock_sweep).id = PropertyMock(return_value="test_sweep")
        type(mock_sweep).project = PropertyMock(return_value="test-proj")
        mock_sweep.config = {"description": "debug sweep"}

        mock_best_run = MagicMock(spec=Run)
        type(mock_best_run).id = PropertyMock(return_value="best_run_abc")
        mock_best_run.config = {}

        captured = {}

        def capture_email(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        mocker.patch("src.tasks.sensitivity.build_sensitivity_email", side_effect=capture_email)
        send_mock = mocker.patch("src.tasks.sensitivity.send_email_with_mimemultipart")

        task = SensitivityTask(
            cfg_workflow, mock_wandb_service, mock_tmux_service, MagicMock(spec=CommandBuilder)
        )
        task.load_eval_checkpoints = MagicMock(return_value={})
        task.collect_checkpoint_paths = MagicMock(return_value=[])

        task._send_sensitivity_email(
            sweep_id="test_sweep",
            sweep=mock_sweep,
            best_run=mock_best_run,
            best_config_dict={"model.optimizer.lr": "0.01"},
            best_params_metrics={"test/acc": {"mean": 0.95, "std": 0.01}},
            param_grid_results=[{"param_desc": "model.optimizer.lr=0.001", "metrics": {"test/acc": {"mean": 0.94, "std": 0.02}}}],
            image_paths=[],
            reproduction_scripts=[],
            group_urls={},
        )

        workflow_info = captured["workflow_info"]
        assert workflow_info["report_label"] == "Sensitivity Report (JSON)"
        assert workflow_info["report_json_path"].endswith("reports/sensitivity_test_sweep.json")
        assert "eval_test_sweep" not in workflow_info["report_json_path"]
        assert workflow_info["report_csv_path"] == "N/A"
        assert workflow_info["log_dir"].endswith("reports")
        send_mock.assert_called_once()

    def test_sensitivity_backward_compat_param_grid(
        self,
        cfg_workflow: DictConfig,
        mock_wandb_service: MagicMock,
        mock_tmux_service: MagicMock,
        mocker: MagicMock,
    ):
        """Test legacy param_grid format still works (backward compatibility)."""
        mocker.patch("src.tasks.shared._DRY_RUN", True)

        mock_sweep = MagicMock(spec=Sweep, state="FINISHED")
        type(mock_sweep).id = PropertyMock(return_value="test_sweep")
        type(mock_sweep).project = PropertyMock(return_value="test-proj")
        type(mock_sweep).name = PropertyMock(return_value="test-name")
        mock_sweep.config = {}

        mock_best_run = MagicMock(spec=Run)
        mock_best_run.config = {}
        type(mock_best_run).id = PropertyMock(return_value="best_run_abc")
        type(mock_best_run).name = PropertyMock(return_value="best-run-name")

        mock_wandb_service.get_sweep.return_value = mock_sweep
        mock_wandb_service.find_best_run.return_value = mock_best_run

        mock_command_builder = MagicMock(spec=CommandBuilder)
        mock_command_builder.build_training_run_command.return_value = "python src/train.py seed=42"

        with open_dict(cfg_workflow):
            cfg_workflow.workflow.dry_run = True
            cfg_workflow.workflow.sensitivity_task.num_seeds = 1
            # No sensitivities list, use legacy param_grid
            cfg_workflow.workflow.sensitivity_task.sensitivities = []
            cfg_workflow.workflow.sensitivity_task.param_grid = {
                "optimizer.lr": [0.001, 0.01],
            }

        task = SensitivityTask(
            cfg_workflow, mock_wandb_service, mock_tmux_service, mock_command_builder
        )
        # Mock ensure_evaluate_results to avoid full evaluate flow
        mocker.patch.object(
            type(task), "ensure_evaluate_results",
            return_value={"evaluation_summary": {"test/acc": {"mean": 0.95, "std": 0.01}}}
        )

        task.run(sweep_id="test_sweep")

        # 2 lr combos
        assert mock_command_builder.build_training_run_command.call_count == 2

    def test_sensitivity_no_sweep_exits(
        self,
        cfg_workflow: DictConfig,
        mock_wandb_service: MagicMock,
        mock_tmux_service: MagicMock,
        mocker: MagicMock,
    ):
        """Test sensitivity exits if sweep not found."""
        mocker.patch("src.tasks.shared._DRY_RUN", False)
        mock_wandb_service.get_sweep.return_value = None

        mock_command_builder = MagicMock(spec=CommandBuilder)

        task = SensitivityTask(
            cfg_workflow, mock_wandb_service, mock_tmux_service, mock_command_builder
        )

        with pytest.raises(SweepError):
            task.run(sweep_id="nonexistent_sweep")


class TestBaseTaskParallelStrategy:
    """Test BaseTask.execute_parallel_strategy and wait_for_session_with_timeout."""

    def test_parallel_strategy_devices_0_0(
        self,
        cfg_workflow: DictConfig,
        mock_wandb_service: MagicMock,
        mock_tmux_service: MagicMock,
        mocker: MagicMock,
    ):
        """Test devices=[0,0] creates 2 workers on same GPU."""
        with open_dict(cfg_workflow):
            cfg_workflow.workflow.dry_run = True
            cfg_workflow.workflow.sweep_task.devices = [0, 0]

        mocker.patch("src.tasks.shared._DRY_RUN", True)

        mock_command_builder = MagicMock(spec=CommandBuilder)
        mock_command_builder.build_training_run_command.return_value = "python src/train.py seed=42"

        from src.tasks.base import BaseTask
        task = BaseTask(
            cfg_workflow, mock_wandb_service, mock_tmux_service, mock_command_builder
        )

        experiments = [
            {"group_name": "grp1", "overrides": ["model.lr=0.01"], "num_seeds": 1, "seed_start": 42},
            {"group_name": "grp2", "overrides": ["model.lr=0.001"], "num_seeds": 1, "seed_start": 42},
        ]

        result = task.execute_parallel_strategy(experiments, "test_session", "rerun")

        # Should return session name in dry-run
        assert result == "test_session"
        # build_training_run_command called twice (1 per experiment × 1 seed)
        assert mock_command_builder.build_training_run_command.call_count == 2

    def test_wait_for_session_with_timeout(
        self,
        cfg_workflow: DictConfig,
        mock_wandb_service: MagicMock,
        mock_tmux_service: MagicMock,
        mocker: MagicMock,
    ):
        """Test wait_for_session_with_timeout completes when session closes."""
        mocker.patch("src.tasks.shared._DRY_RUN", False)

        # Session exists on first check, gone on second
        mock_tmux_service.session_exists.side_effect = [True, False]

        mock_command_builder = MagicMock(spec=CommandBuilder)

        from src.tasks.base import BaseTask
        task = BaseTask(
            cfg_workflow, mock_wandb_service, mock_tmux_service, mock_command_builder
        )

        # Should complete without timeout
        task.wait_for_session_with_timeout("test_session", timeout_secs=30, interval=1)
        # session_exists called at least once
        assert mock_tmux_service.session_exists.called

    def test_wait_for_session_timeout_kills(
        self,
        cfg_workflow: DictConfig,
        mock_wandb_service: MagicMock,
        mock_tmux_service: MagicMock,
        mocker: MagicMock,
    ):
        """Test wait_for_session_with_timeout kills session on timeout."""
        mocker.patch("src.tasks.shared._DRY_RUN", False)

        # Session always exists (simulating stuck)
        mock_tmux_service.session_exists.return_value = True

        mock_command_builder = MagicMock(spec=CommandBuilder)

        from src.tasks.base import BaseTask
        task = BaseTask(
            cfg_workflow, mock_wandb_service, mock_tmux_service, mock_command_builder
        )

        # Very short timeout to trigger kill quickly
        import time
        start = time.time()
        task.wait_for_session_with_timeout("test_session", timeout_secs=2, interval=1)
        elapsed = time.time() - start

        # Should have called kill_session
        mock_tmux_service.kill_session.assert_called_once_with("test_session")
        assert elapsed < 5  # Should complete quickly after timeout
