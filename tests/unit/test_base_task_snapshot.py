import pytest
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from src.tasks.base import BaseTask
from src.services.command_builder import CommandBuilder
from src.services.sandbox_service import SandboxService


class TestBaseTaskSnapshot:
    @pytest.fixture
    def cfg(self):
        return OmegaConf.create({
            "workflow": {
                "general": {"latest_status_file": "/tmp/latest.yaml"},
                "sweep_task": {"conda_env": "test_env"},
                "evaluate_task": {
                    "num_seeds": 2,
                    "seed_start": 42,
                    "mode": "rerun",
                    "run_command": {"base_args": ["python", "src/train.py"]},
                },
                "devices": [0],
                "wandb": {"entity": "test", "project": "test"},
            }
        })

    @pytest.fixture
    def mock_services(self):
        return {
            "wandb": MagicMock(),
            "tmux": MagicMock(),
            "sandbox": MagicMock(),
        }

    def test_base_task_without_sandbox(self, cfg, mock_services):
        """sandbox_service=None 时属性为 None"""
        task = BaseTask(
            cfg, mock_services["wandb"], mock_services["tmux"],
            CommandBuilder(cfg.workflow),
        )
        assert task.sandbox_service is None
        assert task.prepare_sandbox("s1", "eval") == ""

    def test_base_task_with_sandbox(self, cfg, mock_services):
        """sandbox_service 传入后可用"""
        mock_services["sandbox"].setup_sandbox.return_value = "/tmp/sandbox"
        task = BaseTask(
            cfg, mock_services["wandb"], mock_services["tmux"],
            CommandBuilder(cfg.workflow), mock_services["sandbox"],
        )
        assert task.sandbox_service is mock_services["sandbox"]
        result = task.prepare_sandbox("s1", "eval", rank=1)
        assert result == "/tmp/sandbox"
        mock_services["sandbox"].setup_sandbox.assert_called_once_with(
            sweep_id="s1", task_name="eval", rank=1,
        )


    def test_command_builder_adds_host_runtime_overrides_for_sandbox(self, cfg):
        """Sandbox payload should write logs/checkpoints back to the host repo."""
        cb = CommandBuilder(cfg.workflow)

        cmd = cb.build_training_run_command(
            overrides=["model.optimizer.lr=0.01"],
            seed=42,
            group_name="g1",
            cwd="/tmp/sandbox",
        )

        assert cmd.startswith("cd '/tmp/sandbox' && ")
        assert "paths.root_dir=" in cmd
        assert "paths.data_dir=" in cmd
        assert "paths.log_dir=" in cmd
        assert "logger.wandb.save_dir=" in cmd

    def test_execute_strategy_passes_snapshot_dir(self, cfg, mock_services):
        """execute_strategy 将 snapshot_dir 传给 RerunStrategy"""
        with patch("src.tasks.base.RerunStrategy") as MockStrategy:
            mock_strategy = MagicMock()
            MockStrategy.return_value = mock_strategy
            mock_strategy.execute.return_value = "test_session"

            task = BaseTask(
                cfg, mock_services["wandb"], mock_services["tmux"],
                CommandBuilder(cfg.workflow),
            )
            result = task.execute_strategy(
                ["a=1"], "g1", "rerun",
                num_seeds=2, seed_start=42, snapshot_dir="/tmp/sb",
            )
            MockStrategy.assert_called_once()
            args, kwargs = MockStrategy.call_args
            # RerunStrategy(args: wandb, tmux, cmdb, cfg, group, num_seeds, seed_start, snapshot_dir)
            assert args[7] == "/tmp/sb"
            assert result == "test_session"

    def test_execute_parallel_strategy_passes_cwd(self, cfg, mock_services):
        """execute_parallel_strategy 在 build_training_run_command 中传 cwd"""
        cb = CommandBuilder(cfg.workflow)
        task = BaseTask(
            cfg, mock_services["wandb"], mock_services["tmux"],
            cb, mock_services["sandbox"],
        )
        mock_services["tmux"].session_exists.return_value = False

        experiments = [{
            "group_name": "g1",
            "overrides": ["a=1"],
            "num_seeds": 1,
            "seed_start": 42,
        }]
        with patch.object(cb, "build_training_run_command", wraps=cb.build_training_run_command) as mock_build:
            task.execute_parallel_strategy(
                experiments, "sess", "rerun", snapshot_dir="/tmp/sb",
            )
            mock_build.assert_called()
            for call in mock_build.call_args_list:
                assert call.kwargs.get("cwd") == "/tmp/sb"
