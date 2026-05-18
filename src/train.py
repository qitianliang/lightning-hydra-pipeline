import multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# rootutils: adds project root to PYTHONPATH, sets PROJECT_ROOT env,
# loads .env. See https://github.com/ashleve/rootutils

from src.utils import (
    RankedLogger,
    WandbCleanupHandler,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    set_process_title,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set process title
    set_process_title(cfg)
    metric_dict = {}
    object_dict = {}

    # 創建清理處理器的實例
    cleanup_handler = WandbCleanupHandler(cfg)

    # 使用 with 語句包裹核心邏輯
    with cleanup_handler:
        if cfg.get("seed"):
            L.seed_everything(cfg.seed, workers=True)

        log.info(f"Instantiating builder <{cfg.builder._target_}>")
        builder_partial = hydra.utils.instantiate(cfg.builder)

        log.info("Completing builder instantiation by passing the root 'cfg' object.")
        builder = builder_partial(cfg=cfg)
        model, datamodule = builder.create()

        log.info("Instantiating callbacks...")
        callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

        log.info("Instantiating loggers...")
        logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

        # Register trainer with cleanup handler
        cleanup_handler.set_trainer(trainer)
        object_dict = {
            "cfg": cfg,
            "datamodule": datamodule,
            "model": model,
            "callbacks": callbacks,
            "logger": logger,
            "trainer": trainer,
        }

        if logger:
            log.info("Logging hyperparameters!")
            log_hyperparameters(object_dict)

        if cfg.get("train"):
            log.info("Starting training!")
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        train_metrics = trainer.callback_metrics

        if cfg.get("test"):
            log.info("Starting testing!")
            if cfg.ckpt_path:
                log.info(f"Loading checkpoint: {cfg.ckpt_path}")
                ckpt_path = cfg.ckpt_path
            else:
                log.info("No ckpt_path provided!")
                ckpt_path = trainer.checkpoint_callback.best_model_path
                if ckpt_path == "":
                    log.warning("No best ckpt found!")
                    ckpt_path = None
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
            log.info(f"Best ckpt path: {ckpt_path}")

        test_metrics = trainer.callback_metrics

        # merge train and test metrics
        metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)
    if cfg.train:
        # safely retrieve metric value for hydra-based hyperparameter optimization
        metric_value = get_metric_value(
            metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
        )

        # return optimized metric
        return metric_value


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
