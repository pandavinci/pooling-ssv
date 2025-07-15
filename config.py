from dataclasses import dataclass, field
from typing import Any, List

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

@dataclass
class LossConfig:
    name: str = MISSING

@dataclass
class AAMConfig(LossConfig):
    margin: float = MISSING
    s: float = MISSING
    easy_margin: bool = False

@dataclass
class ModelConfig:
    extractor: str = MISSING
    processor: str = MISSING
    classifier: str = MISSING
    loss: LossConfig = MISSING

@dataclass
class EnvironmentConfig:
    local: bool = False
    metacentrum: bool = False
    sge: bool = False

    batch_size: int = MISSING
    lstm_batch_size: int = MISSING

    rir_root: str = MISSING
    musan_root: str = MISSING
    data_dir: str = MISSING
    sltsstc: dict = MISSING

@dataclass
class TrainingConfig:
    dataset: str = MISSING
    checkpoint: str | None = None
    start_epoch: int = MISSING
    num_epochs: int = MISSING
    save_embeddings: bool = False
    augment: bool = False

defaults = [
    "_self_",
    "model/wavlm_mhfa_aam",
    "environment/local",
    "training/wavlm_mhfa_aam_augment",
]

@dataclass
class Config:
    # this is unfortunately verbose due to @dataclass limitations
    defaults: List[Any] = field(default_factory=lambda: defaults)

    model: ModelConfig = MISSING
    environment: EnvironmentConfig = MISSING
    training: TrainingConfig = MISSING

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="training", name="base_training", node=TrainingConfig)
cs.store(group="environment", name="base_environment", node=EnvironmentConfig)
cs.store(group="model", name="base_model", node=ModelConfig)
cs.store(group="model/loss", name="base_loss", node=LossConfig) # general
cs.store(group="model/loss", name="aam_loss", node=AAMConfig)   # specialized