"""Module for load configuration from yaml file."""
from typing import List

from omegaconf import OmegaConf
from pydantic import BaseModel


class LossConfig(BaseModel):
    """Config for losses"""
    name: str
    weight: float
    loss_fn: str
    loss_kwargs: dict


class DataConfig(BaseModel):
    """Config for data"""
    batch_size: int
    num_iterations: int
    n_workers: int
    width: int
    height: int
    vocab: str  # str
    text_size: int


class Config(BaseModel):
    """Config for model"""
    project_name: str
    experiment_name: str
    data_config: DataConfig
    n_epochs: int
    num_classes: int
    accelerator: str
    device: int
    monitor_metric: str
    monitor_mode: str
    model_kwargs: dict
    optimizer: str
    optimizer_kwargs: dict
    scheduler: str
    scheduler_kwargs: dict
    losses: List[LossConfig]

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """
        Load config from yaml file
        :param path: path to yaml file
        :return: Config instance
        """
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
