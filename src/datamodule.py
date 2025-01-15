"""Lightning module for data preprocessing."""
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, RandomSampler

from src.config import DataConfig
from src.constants import DATA_PATH
from src.dataset import PlatesCodeDataset
from src.transforms import get_transforms


class OCRDM(LightningDataModule):
    """Lightning module for data preprocessing."""

    def __init__(self, config: DataConfig):
        super().__init__()
        self._config = config
        self._train_transforms = get_transforms(
            width=config.width,
            height=config.height,
            vocab=config.vocab,
            text_size=config.text_size,
        )
        self._valid_transforms = get_transforms(
            width=config.width,
            height=config.height,
            vocab=config.vocab,
            text_size=config.text_size,
            augmentations=False,
        )
        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.train_sampler: Optional[RandomSampler] = None

    def setup(self, stage: Optional[str] = None):
        """Set up the data.

        :param stage: name of training stage
        :return: None
        """
        self.train_dataset = PlatesCodeDataset(
            phase="train",
            data_folder=DATA_PATH,
            reset_flag=False,
            transforms=self._train_transforms,
        )
        self.valid_dataset = PlatesCodeDataset(
            phase="test",
            data_folder=DATA_PATH,
            reset_flag=False,
            transforms=self._valid_transforms,
        )
        if self._config.num_iterations != -1:
            self.train_sampler = RandomSampler(
                data_source=self.train_dataset,
                num_samples=self._config.num_iterations * self._config.batch_size,
            )

    def train_dataloader(self) -> DataLoader:
        """Load the training dataloader.

        :return: torch.utils.data.DataLoader
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self._config.batch_size,
            num_workers=self._config.n_workers,
            shuffle=True,  # False if self.train_sampler else True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Load the validation dataloader.

        :return: torch.utils.data.DataLoader
        """
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self._config.batch_size,
            num_workers=self._config.n_workers,
            shuffle=False,
            pin_memory=True,
        )
