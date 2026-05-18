"""This file prepares config fixtures for other tests."""

from pathlib import Path

import pytest
import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for training.

    :return: A DictConfig object containing a default Hydra configuration for training.
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=[])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.01
            cfg.trainer.limit_val_batches = 0.1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for evaluation.

    :return: A DictConfig containing a default Hydra configuration for evaluation.
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="eval.yaml", return_hydra_config=True, overrides=["ckpt_path=."])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="function")
def cfg_train(cfg_train_global: DictConfig, tmp_path: Path) -> DictConfig:
    """A pytest fixture built on top of the `cfg_train_global()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_train` arg. Each test generates its own temporary logging path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.

    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global: DictConfig, tmp_path: Path) -> DictConfig:
    """A pytest fixture built on top of the `cfg_eval_global()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_eval` arg. Each test generates its own temporary logging path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.

    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_eval_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


from unittest.mock import MagicMock

# 確保項目根路徑被添加到 sys.path
# 這對於後續導入 src.services.* 至關重要
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@pytest.fixture(scope="package")
def cfg_workflow_global() -> DictConfig:
    """加載工作流的基礎配置 (`workflow.yaml`)，並為測試設置默認值。 使用 'package' 作用域，所以在整個測試包中只執行一次。"""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="workflow.yaml", return_hydra_config=True, overrides=[])

        # 為所有測試設置快速、安全的默認值
        with open_dict(cfg):
            # 默認開啟 dry_run 模式，防止測試產生副作用
            cfg.workflow.dry_run = True
            # 縮短等待時間以加速測試
            cfg.workflow.evaluate_task.wait_interval_seconds = 1
            # 減少評估的種子數量
            cfg.workflow.evaluate_task.num_seeds = 2
            # top_n = 1 for test speed
            cfg.workflow.evaluate_task.top_n = 1
            # 使用一個假的 conda 環境名
            cfg.workflow.sweep_task.conda_env = "test_env"

    return cfg


@pytest.fixture(scope="function")
def cfg_workflow(cfg_workflow_global: DictConfig, tmp_path: Path) -> DictConfig:
    """為每個測試函數提供一個獨立的、經過修改的配置副本。 使用 'function' 作用域，確保測試間的文件系統隔離。"""
    cfg = cfg_workflow_global.copy()

    # 將所有路徑指向由 pytest 提供的臨時目錄
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.workflow.general.status_dir = str(tmp_path / "status")
        cfg.workflow.evaluate_task.report_path = str(tmp_path / "reports/final_report.json")

    # `yield` 將配置提供給測試函數
    yield cfg

    # 測試函數結束後，清理 Hydra 的全局狀態
    GlobalHydra.instance().clear()


@pytest.fixture
def mock_wandb_service(mocker) -> MagicMock:
    """模擬 (Mock) WandbService，以避免在測試中產生真實的 API 調用。 返回 Mock 對象的實例。"""
    mock = mocker.patch("src.workflow.WandbService", autospec=True)
    return mock.return_value


@pytest.fixture
def mock_tmux_service(mocker) -> MagicMock:
    """模擬 (Mock) TmuxService，以避免在測試中產生真實的 tmux 子進程。 返回 Mock 對象的實例。"""
    mock = mocker.patch("src.workflow.TmuxService", autospec=True)
    return mock.return_value
