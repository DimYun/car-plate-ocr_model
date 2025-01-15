"""Main training script."""

import argparse
import os
from os import path as osp
from typing import Any

import pytorch_lightning as pl
from clearml import Task
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         RichProgressBar)

from src.config import Config
from src.constants import EXPERIMENTS_PATH
from src.datamodule import OCRDM
from src.lightning_module import OCRModule


def arg_parse() -> Any:
    """
    Parse command line
    :return: dictionary like structure
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file")
    return parser.parse_args()


def train(config: Config):
    """
    Main function to strart the training
    :param config: python config module
    :return:
    """
    Task.init(project_name="hw-02-ocr", task_name="pre_exp")

    datamodule = OCRDM(config.data_config)
    model = OCRModule(config)

    experiment_save_path = osp.join(EXPERIMENTS_PATH, config.experiment_name)
    os.makedirs(experiment_save_path, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator=config.accelerator,
        devices=[config.device],
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            RichProgressBar(),
            ModelCheckpoint(
                experiment_save_path,
                save_top_k=1,
                monitor="valid_string_match",
                mode="max",
                every_n_epochs=3,
                filename="epoch_{{epoch:02d}}-{{valid_string_match:.3f}}",
            ),
        ],
        log_every_n_steps=10,
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    args = arg_parse()
    seed_everything(seed=42, workers=True)
    config = Config.from_yaml(args.config_file)
    train(config)
