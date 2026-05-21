import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.services.sandbox_service import SandboxService
from src.utils.exceptions import WorkflowError

class TestSandboxService:
    @pytest.fixture
    def mock_project_root(self, tmp_path):
        # Create a fake project root with .project-root
        project_root = tmp_path / "mock_project"
        project_root.mkdir()
        (project_root / ".project-root").touch()
        
        # Create some fake directories to be packed
        for d in ["src", "configs", "scripts"]:
            d_path = project_root / d
            d_path.mkdir()
            (d_path / f"test_{d}.txt").write_text(f"dummy {d}")
            
        # Create some resources to be linked
        for res in ["data", ".env", "pyproject.toml"]:
            if res == "data":
                (project_root / res).mkdir()
            else:
                (project_root / res).touch()
                
        # Excluded directories
        (project_root / "logs").mkdir()
        (project_root / "__pycache__").mkdir()
        
        return project_root

    @pytest.fixture
    def sandbox_service(self, mock_project_root):
        with patch("src.services.sandbox_service.SandboxService._infer_project_root", return_value=mock_project_root):
            service = SandboxService(entity="test_entity", project="test_project")
            return service

    @patch("src.services.sandbox_service.wandb")
    def test_publish_source_to_wandb(self, mock_wandb, sandbox_service):
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        
        artifact_path = sandbox_service.publish_source_to_wandb("test_sweep_123")
        
        # Assertions
        assert artifact_path == "test_entity/test_project/code_sweep_test_sweep_123:latest"
        mock_wandb.init.assert_called_once()
        mock_wandb.Artifact.assert_called_once()
        mock_artifact = mock_wandb.Artifact.return_value
        mock_artifact.add_file.assert_called_once()
        mock_run.log_artifact.assert_called_once_with(mock_artifact)
        mock_run.finish.assert_called_once()

    @patch("src.services.sandbox_service.wandb")
    def test_setup_sandbox(self, mock_wandb, sandbox_service, tmp_path):
        # Mock WandB Api and Artifact
        mock_api = MagicMock()
        mock_wandb.Api.return_value = mock_api
        mock_artifact = MagicMock()
        mock_api.artifact.return_value = mock_artifact
        
        # Simulate artifact.download() by placing a valid tar.gz file
        def fake_download(root):
            tar_path = Path(root) / "source.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                # Add an empty file to the tar
                fake_file = tmp_path / "dummy.txt"
                fake_file.touch()
                tar.add(fake_file, arcname="dummy.txt")
            return root
            
        mock_artifact.download.side_effect = fake_download
        
        sandbox_dir = sandbox_service.setup_sandbox("test_sweep_123", task_name="eval", rank=1)
        
        assert sandbox_dir.exists()
        assert sandbox_dir.name == "eval_r1"
        assert sandbox_dir.parent.name == "sweep_test_sweep_123"
        assert sandbox_dir.parent.parent.name == "test_project"
        
        # Check that downloaded tar was extracted
        assert (sandbox_dir / "dummy.txt").exists()
        
        # Check that link_targets were symlinked
        for res in ["data", ".env", "pyproject.toml"]:
            assert (sandbox_dir / res).is_symlink()
            
        # Check that logs and wandb directories were created
        assert (sandbox_dir / "logs").exists()
        assert (sandbox_dir / "wandb").exists()
        
        # Clean up
        shutil.rmtree(sandbox_dir)

    def test_teardown(self, sandbox_service, tmp_path):
        sandbox_dir = tmp_path / "sandbox_test"
        sandbox_dir.mkdir()
        
        sandbox_service.teardown(sandbox_dir)
        
        assert not sandbox_dir.exists()
        
    def test_teardown_with_archive(self, sandbox_service, mock_project_root, tmp_path):
        # Create a mock sandbox_dir
        sandbox_parent = tmp_path / "sweep_test_123"
        sandbox_parent.mkdir()
        sandbox_dir = sandbox_parent / "eval_r1"
        sandbox_dir.mkdir()
        
        # Create a logs directory in sandbox
        sandbox_logs = sandbox_dir / "logs"
        sandbox_logs.mkdir()
        (sandbox_logs / "test_log.txt").write_text("dummy log")
        
        # Run teardown with archive_logs=True
        sandbox_service.teardown(sandbox_dir, archive_logs=True)
        
        # Verify sandbox is deleted
        assert not sandbox_dir.exists()
        
        # Verify logs were archived to the project root
        archive_dir = mock_project_root / "logs" / "sandbox_archive" / "sweep_test_123" / "eval_r1"
        assert archive_dir.exists()
        assert (archive_dir / "test_log.txt").exists()
        assert (archive_dir / "test_log.txt").read_text() == "dummy log"
