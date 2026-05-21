"""Sandbox Service — 不可变沙盒架构，替代原有的 git worktree 隔离方案。

生命周期：
  1. publish_source_to_wandb(): 打包当前工作区 src, configs, scripts 并上传至 W&B。
  2. setup_sandbox(): 在 /tmp 目录重建源码环境，使用软链接接入 data/.env，提供绝对的隔离。
  3. teardown(): 任务完成后清理临时沙盒。
"""

import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Optional
import uuid

import wandb

from src.utils import RankedLogger, WorkflowError

log = RankedLogger(__name__, rank_zero_only=True)


class SandboxService:
    def __init__(self, entity: str, project: str):
        self.entity = entity
        self.project = project
        self.project_root = self._infer_project_root()

    @staticmethod
    def _infer_project_root() -> Path:
        """寻找包含 .project-root 标记的项目根目录。"""
        current = Path.cwd().resolve()
        while current != current.parent:
            if (current / ".project-root").exists():
                return current
            current = current.parent
        # 回退：如果没找到，默认认为是当前目录
        return Path.cwd().resolve()

    def publish_source_to_wandb(self, sweep_id: str) -> str:
        """打包源码并作为 W&B Artifact 上传，与特定 sweep 绑定。"""
        log.info(f"📦 Packing source code for sweep {sweep_id}...")
        
        tar_fd, tar_path = tempfile.mkstemp(suffix=".tar.gz")
        os.close(tar_fd)
        
        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                for target_dir in ["src", "configs", "scripts"]:
                    src_path = self.project_root / target_dir
                    if src_path.exists():
                        tar.add(src_path, arcname=target_dir, filter=self._tar_filter)
            
            log.info(f"⬆️ Uploading source artifact code_sweep_{sweep_id} to W&B...")
            # 临时创建一个 wandb run 用于上传 artifact
            run = wandb.init(
                entity=self.entity,
                project=self.project,
                job_type="publish_code",
                name=f"publish_code_{sweep_id}",
                tags=["code_snapshot"],
                # 隐藏控制台输出
                settings=wandb.Settings(silent="true")
            )
            try:
                artifact = wandb.Artifact(
                    name=f"code_sweep_{sweep_id}",
                    type="code_snapshot",
                    description=f"Immutable source snapshot for Sweep {sweep_id}"
                )
                artifact.add_file(tar_path, name="source.tar.gz")
                run.log_artifact(artifact)
                artifact.wait()
            finally:
                run.finish()
            
            log.info("✅ Source artifact published successfully.")
            return f"{self.entity}/{self.project}/code_sweep_{sweep_id}:latest"
            
        finally:
            if os.path.exists(tar_path):
                os.remove(tar_path)

    def _tar_filter(self, tarinfo: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        """打包时过滤不需要的文件或目录。"""
        exclude_patterns = ["__pycache__", ".pyc", ".git", "wandb", "logs"]
        for p in exclude_patterns:
            if p in tarinfo.name:
                return None
        return tarinfo

    def setup_sandbox(self, sweep_id: str, task_name: str = "eval", rank: Optional[int] = None) -> Path:
        """下载代码制品，在系统的 /tmp 目录构建实验沙盒。

        Args:
            sweep_id: 要拉取的 sweep 对应的代码版本
            task_name: 任务类型名称，用于构造子目录 (e.g., eval, ablation)
            rank: 可选的标识符，如 eval 的排名
            
        Returns:
            沙盒目录的绝对路径 (Path)
        """
        artifact_name = f"{self.entity}/{self.project}/code_sweep_{sweep_id}:latest"
        
        # 动态沙盒目录，位于 /tmp，永远不会污染项目主目录
        suffix = f"_r{rank}" if rank else f"_{uuid.uuid4().hex[:8]}"
        sandbox_dir = Path(tempfile.gettempdir()) / f"lightning-runs/{self.project}/sweep_{sweep_id}/{task_name}{suffix}"
        
        if sandbox_dir.exists():
            log.info(f"🗑️ Cleaning existing sandbox: {sandbox_dir}")
            shutil.rmtree(sandbox_dir)
        
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"🏗️ Building sandbox at: {sandbox_dir}")
        
        # 1. 下载并解压冻结的代码
        api = wandb.Api()
        try:
            artifact = api.artifact(artifact_name)
        except Exception as e:
            raise WorkflowError(f"Failed to fetch source artifact '{artifact_name}'. Did you run 'pipeline' or 'sweep' from local machine to publish it first? Error: {e}")
            
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
            
        # 2. 软链接挂载宿主机的巨型资源（零拷贝）
        # 考虑更多可能的根目录配置文件，如 pyproject.toml 等
        link_targets = ["data", ".env", "pyproject.toml", "requirements.txt", "Makefile"]
        for resource in link_targets:
            src_res = self.project_root / resource
            dst_res = sandbox_dir / resource
            if src_res.exists():
                try:
                    os.symlink(src_res, dst_res)
                    log.info(f"  🔗 Symlinked {resource}")
                except FileExistsError:
                    pass
                
        # 3. 创建独立的运行输出目录，避免并发冲突
        (sandbox_dir / "logs").mkdir(exist_ok=True)
        (sandbox_dir / "wandb").mkdir(exist_ok=True)
        
        log.info(f"✅ Sandbox ready at: {sandbox_dir}")
        return sandbox_dir

    def teardown(self, sandbox_dir: Path, archive_logs: bool = False):
        """任务结束后销毁沙盒。"""
        if not sandbox_dir.exists():
            return
            
        if archive_logs:
            sandbox_logs = sandbox_dir / "logs"
            if sandbox_logs.exists() and any(sandbox_logs.iterdir()):
                # 将沙盒内产生的日志拷贝回主项目的 logs/sandbox_archive 中
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
