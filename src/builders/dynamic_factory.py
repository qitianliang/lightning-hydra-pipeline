import logging
from typing import Any, Dict, List, Tuple

import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf  # Make sure OmegaConf is imported

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class DynamicConfigModelFactory:
    """通用动态配置工厂。(最终版，修复了 OmegaConf 类型检查问题)"""

    def __init__(self, cfg: DictConfig, transfer_attributes: List):
        self.cfg = cfg
        self.transfer_rules = transfer_attributes
        self.failed = {}

    def create(self) -> Tuple[pl.LightningModule, pl.LightningDataModule]:
        log.info("Starting dynamic configuration process...")
        datamodule = hydra.utils.instantiate(self.cfg.data)
        log.info(f"Instantiated datamodule: {type(datamodule).__name__}")

        # Ensure data is downloaded (safe & idempotent, CPU-only, no DDP concern).
        # Do NOT call datamodule.setup() here — Lightning's Trainer manages setup()
        # lifecycle internally. Calling setup() early can cause double-setup issues
        # with custom datamodules that lack idempotency guards.
        datamodule.prepare_data()

        self._transfer_attributes(datamodule)

        log.info("Final model config after dynamic updates:\n" + OmegaConf.to_yaml(self.cfg.model))
        log.info(f"{self.failed.keys()=}")
        net = hydra.utils.instantiate(self.cfg.model.net, **self.failed)
        model = hydra.utils.instantiate(self.cfg.model, net=net)
        log.info(f"Instantiated model: {type(model).__name__}")

        return model, datamodule

    def _transfer_attributes(self, datamodule: pl.LightningDataModule):
        # --- KEY FIX IS HERE ---
        # Convert OmegaConf ListConfig/DictConfig to native Python list/dict
        rules = OmegaConf.to_object(self.transfer_rules)
        # ---------------------

        if not rules:
            log.info("No dynamic attributes to transfer.")
            return

        log.info(f"Transferring attributes based on rules: {rules}")
        for rule in rules:
            source_attr, target_path = "", ""
            # Now this loop works with standard Python types, so isinstance is correct
            if isinstance(rule, str):
                source_attr = rule
                target_path = f"model.net.{rule}"
            elif isinstance(rule, dict):
                source_attr = rule.get("source")
                target_path = rule.get("target")

            if not source_attr or not target_path:
                log.warning(f"Invalid rule found: {rule}. Skipping.")
                continue

            if not hasattr(datamodule, source_attr):
                log.warning(f"Attribute '{source_attr}' not found in datamodule. Skipping.")
                continue

            value = getattr(datamodule, source_attr)

            log.info(
                f"Attempting to set '{target_path}' with value '{value}' from 'datamodule.{source_attr}'."
            )
            try:
                OmegaConf.update(self.cfg, target_path, value, merge=False, force_add=True)
                updated_value = OmegaConf.select(self.cfg, target_path)
                assert value == updated_value, ""
                log.info(f"'{target_path}' successfully updated with '{value}'.")
            except Exception as e:
                self.failed.update(**{source_attr: value})
                log.error(f"Failed to update config for path '{target_path}': {e}")
