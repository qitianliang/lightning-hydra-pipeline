# tests/unit/test_cleanup_context.py

from pathlib import Path
from unittest.mock import MagicMock

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig

from src.utils.cleanup_context import WandbCleanupHandler


# 这是一个简化的 mock 对象，用于模拟 trainer.strategy
class MockStrategy:
    def __init__(self, global_rank: int):
        self.global_rank = global_rank
        self.barrier = MagicMock()

    @property
    def is_global_zero(self) -> bool:
        return self.global_rank == 0


# ======================================================================================
# 核心行为 1 & 2: 成功场景 (单进程) / DDP 主进程场景
# ======================================================================================
def test_cleanup_handler_copies_directory_for_rank_zero(cfg_train: DictConfig, mocker):
    """测试：在 DDP rank 0 或单进程情况下，清理处理器是否能正确复制 run 目录。

    - 覆盖核心行为 #1: 成功场景 (单进程)
    - 覆盖核心行为 #2: DDP 主进程场景
    """
    # 1. Arrange (准备环境)
    mock_shutil_copytree = mocker.patch("shutil.copytree")
    output_dir = Path(cfg_train.paths.output_dir)

    # 定义一个假的 wandb 运行目录结构，对于 mock 测试，路径字符串本身是关键
    source_run_dir = Path("/any/path/wandb/offline-run-test")
    source_wandb_run_files_dir = source_run_dir / "files"

    mock_logger = mocker.MagicMock(spec=WandbLogger)
    mock_logger.experiment.dir = str(source_wandb_run_files_dir)
    mock_logger.experiment.sweep_id = None  # 确保此测试不涉及 sweep 逻辑

    mock_trainer = mocker.MagicMock(spec=Trainer)
    mock_trainer.loggers = [mock_logger]
    mock_trainer.is_global_zero = True
    mock_trainer.strategy = MockStrategy(global_rank=0)

    handler = WandbCleanupHandler(cfg_train)

    # 2. Act (执行操作)
    with handler:
        handler.set_trainer(mock_trainer)

    # 3. Assert (验证结果)
    mock_trainer.strategy.barrier.assert_called_once()
    expected_source = str(source_run_dir)
    expected_destination = str(output_dir / "wandb_run")
    mock_shutil_copytree.assert_called_once_with(expected_source, expected_destination)


# ======================================================================================
# 核心行为 3: DDP 从属进程场景
# ======================================================================================
def test_cleanup_handler_does_nothing_for_other_ranks(cfg_train: DictConfig, mocker):
    """
    测试：在 DDP rank > 0 的情况下，清理处理器是否不做任何文件移动。
    - 覆盖核心行为 #3: DDP 从属进程场景
    """
    # 1. Arrange
    mock_shutil_copytree = mocker.patch("shutil.copytree")

    mock_trainer = mocker.MagicMock(spec=Trainer)
    mock_trainer.is_global_zero = False
    mock_trainer.strategy = MockStrategy(global_rank=1)

    handler = WandbCleanupHandler(cfg=cfg_train)

    # 2. Act
    with handler:
        handler.set_trainer(mock_trainer)

    # 3. Assert
    mock_trainer.strategy.barrier.assert_called_once()
    mock_shutil_copytree.assert_not_called()


# ======================================================================================
# 核心行为 4: 异常场景：Trainer 未设置
# ======================================================================================
def test_cleanup_handler_skips_if_trainer_not_set(cfg_train: DictConfig, mocker):
    """
    测试：如果 handler.set_trainer() 从未被调用，上下文退出时是否能优雅地处理。
    - 覆盖核心行为 #4: 异常场景：Trainer 未设置
    """
    # 1. Arrange
    mock_shutil_copytree = mocker.patch("shutil.copytree")
    handler = WandbCleanupHandler(cfg=cfg_train)

    # 2. Act
    # 进入并退出上下文，但故意不调用 handler.set_trainer()
    with handler:
        pass

    # 3. Assert
    # 验证程序没有崩溃，并且移动操作没有被尝试
    mock_shutil_copytree.assert_not_called()


# ======================================================================================
# 核心行为 5: 异常场景：源目录不存在
# ======================================================================================
def test_cleanup_handler_handles_error_if_source_dir_not_found(cfg_train: DictConfig, mocker):
    """测试：如果预期的 wandb run 目录不存在，是否会捕获异常并优雅退出。"""
    # 1. Arrange
    mock_shutil_copytree = mocker.patch("shutil.copytree")
    mock_shutil_copytree.side_effect = FileNotFoundError("Mocked: directory not found")

    source_run_dir = Path("/non/existent/path/wandb/offline-run-test")
    source_run_files_dir = source_run_dir / "files"

    mock_logger = mocker.MagicMock(spec=WandbLogger)
    mock_logger.experiment.dir = str(source_run_files_dir)
    mock_logger.experiment.sweep_id = None

    mock_trainer = mocker.MagicMock(spec=Trainer)
    mock_trainer.loggers = [mock_logger]
    mock_trainer.is_global_zero = True
    mock_trainer.strategy = MockStrategy(global_rank=0)

    handler = WandbCleanupHandler(cfg_train)

    # 2. Act
    # 我们期望这能正常运行而不抛出异常，因为异常在内部被捕获了
    with handler:
        handler.set_trainer(mock_trainer)

    # 3. Assert
    # 验证代码尝试了复制操作
    expected_source = str(source_run_dir)
    expected_destination = str(Path(cfg_train.paths.output_dir) / "wandb_run")
    mock_shutil_copytree.assert_called_once_with(expected_source, expected_destination)


def test_cleanup_handler_copies_sweep_config(cfg_train: DictConfig, mocker):
    """测试：在 sweep 运行中，是否能正确复制 run 目录和 sweep 配置文件。"""
    # 1. Arrange
    mock_shutil_copytree = mocker.patch("shutil.copytree")
    mock_shutil_copy = mocker.patch("shutil.copy")
    mocker.patch("pathlib.Path.exists", return_value=True)  # 假设 sweep 配置文件存在

    output_dir = Path(cfg_train.paths.output_dir)

    wandb_base_dir = Path("/any/path/wandb")
    source_run_dir = wandb_base_dir / "run-test"
    source_wandb_run_files_dir = source_run_dir / "files"

    mock_logger = mocker.MagicMock(spec=WandbLogger)
    mock_logger.experiment.dir = str(source_wandb_run_files_dir)
    mock_logger.experiment.sweep_id = "test_sweep_id"
    mock_logger.experiment.id = "test_run_id"

    mock_trainer = mocker.MagicMock(spec=Trainer)
    mock_trainer.loggers = [mock_logger]
    mock_trainer.is_global_zero = True
    mock_trainer.strategy = MockStrategy(global_rank=0)

    handler = WandbCleanupHandler(cfg_train)

    # 2. Act
    with handler:
        handler.set_trainer(mock_trainer)

    # 3. Assert
    # 验证 run 目录被复制
    mock_shutil_copytree.assert_called_once_with(
        str(source_run_dir), str(output_dir / "wandb_run")
    )

    # 验证 sweep 配置文件被复制
    expected_config_source = wandb_base_dir / "sweep-test_sweep_id" / "config-test_run_id.yaml"
    expected_config_dest = output_dir / "sweep_params.yaml"
    mock_shutil_copy.assert_called_once_with(
        str(expected_config_source), str(expected_config_dest)
    )


def test_cleanup_handler_skips_if_not_wandb_run_dir(cfg_train: DictConfig, mocker):
    """测试：如果目录名不符合 'run-*' 或 'offline-run-*' 格式，则跳过复制。"""
    mock_shutil_copytree = mocker.patch("shutil.copytree")
    mock_logger = MagicMock(spec=WandbLogger)
    mock_logger.experiment.dir = "/any/path/wandb/custom-run-name/files"
    mock_trainer = MagicMock(
        spec=Trainer, loggers=[mock_logger], is_global_zero=True, strategy=MockStrategy(0)
    )
    handler = WandbCleanupHandler(cfg_train)
    with handler:
        handler.set_trainer(mock_trainer)
    mock_shutil_copytree.assert_not_called()


def test_cleanup_handler_skips_if_no_wandb_logger_present(cfg_train: DictConfig, mocker):
    """测试：当 trainer 的 logger 列表中不包含 WandbLogger 时，处理器是否能优雅跳过。"""
    # 1. Arrange
    mock_shutil_copytree = mocker.patch("shutil.copytree")
    mock_other_logger = mocker.MagicMock(spec=TensorBoardLogger)

    mock_trainer = mocker.MagicMock(spec=Trainer)
    mock_trainer.loggers = [mock_other_logger]
    mock_trainer.is_global_zero = True
    # DDP 相关的 MockStrategy 依然需要，因为 barrier 会被调用
    mock_trainer.strategy = MockStrategy(global_rank=0)

    handler = WandbCleanupHandler(cfg=cfg_train)

    # 2. Act
    with handler:
        handler.set_trainer(mock_trainer)

    # 3. Assert
    # 屏障同步应该总是发生，以确保 DDP 流程的一致性
    mock_trainer.strategy.barrier.assert_called_once()

    # 核心验证：因为没有找到 WandbLogger，移动操作不应该被调用
    mock_shutil_copytree.assert_not_called()
