from dataclasses import dataclass, field
from typing import Any, List

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

@dataclass
class LossConfig:
    name: str = MISSING
    type: str = MISSING

@dataclass
class AAMConfig(LossConfig):
    margin: float = MISSING
    s: float = MISSING
    easy_margin: bool = False

@dataclass
class ModelConfig:
    extractor: str = MISSING
    feature_transform: str = MISSING
    processor: str = MISSING
    classifier: str = MISSING
    loss: LossConfig = MISSING
    trainer: str = MISSING

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

@dataclass
class Config:
    model: ModelConfig = MISSING
    environment: EnvironmentConfig = MISSING
    training: TrainingConfig = MISSING
    save_path: str = MISSING

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="training", name="base_training", node=TrainingConfig)
cs.store(group="environment", name="base_environment", node=EnvironmentConfig)
cs.store(group="model", name="base_model", node=ModelConfig)
cs.store(group="model/loss", name="aam-loss", node=AAMConfig)   # specialized
