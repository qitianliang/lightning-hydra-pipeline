"""Sandbox Service - immutable W&B code snapshots for workflow replays.

Lifecycle:
  1. publish_source_to_wandb(): pack src/configs/scripts and upload once per sweep.
  2. setup_sandbox(): download the frozen source into /tmp and run workers there.
  3. teardown(): remove temporary sandboxes, optionally archiving logs.
"""

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import wandb

from src.utils import RankedLogger, WorkflowError

log = RankedLogger(__name__, rank_zero_only=True)


class SandboxService:
    def __init__(
        self,
        entity: str,
        project: str,
        allow_source_overwrite: bool = False,
        allow_legacy_config_fallbacks: bool = True,
    ):
        self.entity = entity
        self.project = project
        self.allow_source_overwrite = allow_source_overwrite
        self.allow_legacy_config_fallbacks = allow_legacy_config_fallbacks
        self.project_root = self._infer_project_root()

    @staticmethod
    def _infer_project_root() -> Path:
        """Find the project root containing .project-root."""
        current = Path.cwd().resolve()
        while current != current.parent:
            if (current / ".project-root").exists():
                return current
            current = current.parent
        return Path.cwd().resolve()

    def publish_source_to_wandb(self, sweep_id: str) -> str:
        """Pack source code and publish a sweep-bound W&B artifact.

        The artifact name is deterministic: code_sweep_<sweep_id>. Re-publishing it
        would move the :latest pointer and make old sweeps replay with new source, so
        it is refused by default.
        """
        artifact_path = self._artifact_path(sweep_id)
        if self._source_artifact_exists(artifact_path) and not self.allow_source_overwrite:
            raise WorkflowError(
                f"Source artifact already exists: {artifact_path}. Refusing to overwrite the "
                "frozen source for this sweep. Use a new sweep_id for changed code, or set "
                "workflow.snapshot.allow_source_overwrite=true only when you intentionally "
                "want to replace the source snapshot."
            )

        log.info(f"📦 Packing source code for sweep {sweep_id}...")
        manifest = self._build_source_manifest(sweep_id)

        tar_fd, tar_path = tempfile.mkstemp(suffix=".tar.gz")
        os.close(tar_fd)

        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                for target_dir in ["src", "configs", "scripts"]:
                    src_path = self.project_root / target_dir
                    if src_path.exists():
                        tar.add(src_path, arcname=target_dir, filter=self._tar_filter)
                payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
                info = tarfile.TarInfo("source_manifest.json")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))

            log.info(
                f"⬆️ Uploading source artifact code_sweep_{sweep_id} to W&B "
                f"(source_hash={manifest['source_hash'][:12]})..."
            )
            run = wandb.init(
                entity=self.entity,
                project=self.project,
                job_type="publish_code",
                name=f"publish_code_{sweep_id}",
                tags=["code_snapshot"],
                settings=wandb.Settings(silent="true"),
            )
            try:
                artifact = wandb.Artifact(
                    name=f"code_sweep_{sweep_id}",
                    type="code_snapshot",
                    description=f"Immutable source snapshot for Sweep {sweep_id}",
                    metadata=manifest,
                )
                artifact.add_file(tar_path, name="source.tar.gz")
                run.log_artifact(artifact)
                artifact.wait()
            finally:
                run.finish()

            log.info("✅ Source artifact published successfully.")
            return artifact_path

        finally:
            if os.path.exists(tar_path):
                os.remove(tar_path)

    def _artifact_path(self, sweep_id: str) -> str:
        return f"{self.entity}/{self.project}/code_sweep_{sweep_id}:latest"

    @staticmethod
    def _is_not_found_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(token in msg for token in ["not found", "does not exist", "404", "artifact not found"])

    def _source_artifact_exists(self, artifact_path: str) -> bool:
        try:
            wandb.Api().artifact(artifact_path)
            return True
        except Exception as exc:
            if self._is_not_found_error(exc):
                return False
            raise WorkflowError(f"Could not verify source artifact '{artifact_path}' before publishing: {exc}")

    def _build_source_manifest(self, sweep_id: str) -> dict:
        files = []
        combined = hashlib.sha256()
        for target_dir in ["src", "configs", "scripts"]:
            root = self.project_root / target_dir
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                rel = path.relative_to(self.project_root)
                if self._exclude_relative_path(rel):
                    continue
                data = path.read_bytes()
                file_hash = hashlib.sha256(data).hexdigest()
                rel_str = rel.as_posix()
                combined.update(rel_str.encode("utf-8"))
                combined.update(b"\0")
                combined.update(data)
                files.append({"path": rel_str, "sha256": file_hash, "bytes": len(data)})

        git_commit = self._git_output(["git", "rev-parse", "HEAD"])
        git_dirty = bool(self._git_output(["git", "status", "--porcelain"]))
        return {
            "schema_version": 1,
            "sweep_id": sweep_id,
            "source_hash": combined.hexdigest(),
            "file_count": len(files),
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "files": files,
        }

    def _git_output(self, args: list[str]) -> Optional[str]:
        try:
            result = subprocess.run(
                args,
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except Exception:
            return None

    def _tar_filter(self, tarinfo: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        """Filter files that should never be snapshotted."""
        rel = Path(tarinfo.name)
        if self._exclude_relative_path(rel):
            return None
        return tarinfo

    @staticmethod
    def _exclude_relative_path(relative_path: Path) -> bool:
        parts = set(relative_path.parts)
        if parts & {"__pycache__", ".git", "wandb", "logs"}:
            return True
        return relative_path.suffix == ".pyc"

    def setup_sandbox(self, sweep_id: str, task_name: str = "eval", rank: Optional[int] = None) -> Path:
        """Download a source artifact and build an isolated /tmp sandbox."""
        artifact_name = self._artifact_path(sweep_id)

        suffix = f"_r{rank}" if rank else f"_{uuid.uuid4().hex[:8]}"
        sandbox_dir = Path(tempfile.gettempdir()) / f"lightning-runs/{self.project}/sweep_{sweep_id}/{task_name}{suffix}"

        if sandbox_dir.exists():
            log.info(f"🗑️ Cleaning existing sandbox: {sandbox_dir}")
            shutil.rmtree(sandbox_dir)

        sandbox_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"🏗️ Building sandbox at: {sandbox_dir}")

        api = wandb.Api()
        try:
            artifact = api.artifact(artifact_name)
        except Exception as e:
            raise WorkflowError(
                f"Failed to fetch source artifact '{artifact_name}'. Did you run 'pipeline' "
                f"or 'sweep' from local machine to publish it first? Error: {e}"
            )

        download_tmp = tempfile.mkdtemp()
        try:
            download_dir = Path(artifact.download(root=download_tmp))
            tar_path = download_dir / "source.tar.gz"

            if tar_path.exists():
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(path=sandbox_dir)
            else:
                raise WorkflowError("Artifact downloaded but source.tar.gz not found.")
        finally:
            shutil.rmtree(download_tmp, ignore_errors=True)

        manifest_path = sandbox_dir / "source_manifest.json"
        legacy_artifact = not manifest_path.exists()
        if legacy_artifact:
            log.warning("Source artifact has no source_manifest.json; treating it as legacy.")
        else:
            try:
                manifest = json.loads(manifest_path.read_text())
                log.info(
                    "📦 Source snapshot loaded: "
                    f"hash={str(manifest.get('source_hash', 'unknown'))[:12]}, "
                    f"git={manifest.get('git_commit', 'unknown')}"
                )
            except Exception as exc:
                raise WorkflowError(f"Invalid source_manifest.json in source artifact: {exc}")

        link_targets = [
            "data",
            ".env",
            ".project-root",
            "pyproject.toml",
            "requirements.txt",
            "Makefile",
            "setup.py",
            "environment.yaml",
        ]
        for resource in link_targets:
            src_res = self.project_root / resource
            dst_res = sandbox_dir / resource
            if src_res.exists():
                try:
                    os.symlink(src_res, dst_res)
                    log.info(f"  🔗 Symlinked {resource}")
                except FileExistsError:
                    pass

        self._ensure_runtime_config_fallbacks(sandbox_dir, legacy_artifact=legacy_artifact)

        (sandbox_dir / "logs").mkdir(exist_ok=True)
        (sandbox_dir / "wandb").mkdir(exist_ok=True)

        log.info(f"✅ Sandbox ready at: {sandbox_dir}")
        return sandbox_dir

    def _ensure_runtime_config_fallbacks(self, sandbox_dir: Path, legacy_artifact: bool) -> None:
        """Patch only legacy artifacts that predate required runtime configs."""
        fallback_files = [
            Path("configs/logger/wandb.yaml"),
        ]
        for relative_path in fallback_files:
            src_file = self.project_root / relative_path
            dst_file = sandbox_dir / relative_path
            if dst_file.exists():
                continue

            if not legacy_artifact or not self.allow_legacy_config_fallbacks:
                raise WorkflowError(
                    f"Source snapshot is missing required runtime config '{relative_path}'. "
                    "Refusing to mix current host configs into a versioned snapshot. "
                    "Re-publish with a new sweep, or enable workflow.snapshot.legacy_config_fallbacks "
                    "only for known legacy artifacts."
                )
            if not src_file.exists():
                raise WorkflowError(f"Legacy source artifact is missing '{relative_path}', and host fallback does not exist.")

            dst_file.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(src_file, dst_file)
            log.info(f"  🔗 Symlinked missing legacy runtime config {relative_path}")

    def teardown(self, sandbox_dir: Path, archive_logs: bool = False):
        """Destroy a sandbox after the task has finished."""
        if not sandbox_dir.exists():
            return

        if archive_logs:
            sandbox_logs = sandbox_dir / "logs"
            if sandbox_logs.exists() and any(sandbox_logs.iterdir()):
                archive_dir = self.project_root / "logs" / "sandbox_archive" / sandbox_dir.parent.name / sandbox_dir.name
                archive_dir.mkdir(parents=True, exist_ok=True)
                try:
                    for item in sandbox_logs.iterdir():
                        if item.is_file():
                            shutil.copy2(item, archive_dir / item.name)
                        elif item.is_dir():
                            shutil.copytree(item, archive_dir / item.name, dirs_exist_ok=True)
                    log.info(f"💾 Archived sandbox logs to {archive_dir.relative_to(self.project_root)}")
                except Exception as e:
                    log.warning(f"Failed to archive logs: {e}")

        log.info(f"🗑️ Destroying sandbox: {sandbox_dir}")
        shutil.rmtree(sandbox_dir, ignore_errors=True)
