import os
from typing import Any, Dict, Optional

import pandas as pd
import torch
from lightning import LightningDataModule
from pytorch_multilabel_balanced_sampler import ClassCycleSampler
from rdkit.Chem.rdmolfiles import MolFromSmarts
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler

from src.data.components.global_dicts import TASK_DICT
from src.data.datasets import FGRDataset, FGRPretrainDataset


class FGRDataModule(LightningDataModule):
    """`LightningDataModule` for the FGR dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/processed",
        dataset: str = "BBBP",
        method: str = "FG",
        descriptors: bool = False,
        tokenize_dataset: str = "pubchem",
        frequency: int = 500,
        split_type: str = "scaffold",
        fold_idx: int = 0,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `FGRDataModule`.

        :param data_dir: Data directory, defaults to "data/processed"
        :param dataset: Dataset to train, defaults to "BBBP"
        :param method: Method of representation, defaults to "FG"
        :param descriptors: Whether to use descriptors, defaults to False
        :param tokenize_dataset: Tokenization dataset, defaults to "pubchem"
        :param frequency: Frequency for tokenization, defaults to 500
        :param split_type: Dataset split type, defaults to "scaffold"
        :param fold_idx: Fold index, defaults to 0
        :param batch_size: Batch size, defaults to 16
        :param num_workers: Number of workers, defaults to 4
        :param pin_memory: Whether to pin memory, defaults to False
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of classes.
        """
        return TASK_DICT[self.hparams["dataset"]][0]

    @property
    def is_regression(self) -> bool:
        """Check if the task is a regression task.

        :return: Whether the task is a regression task.
        """
        return TASK_DICT[self.hparams["dataset"]][2]

    def _get_dataset_(
        self,
        split_name: str,
    ) -> Dataset:
        """Get a dataset.

        :param split_name: Split name, either "train", "val", or "test".
        :return: The dataset.
        """

        # Read data
        data = pd.read_parquet(
            os.path.join(
                self.hparams["data_dir"],
                "tasks",
                self.hparams["dataset"],
                "splits",
                self.hparams["split_type"],
                f"fold_{self.hparams['fold_idx']}",
                f"{split_name}.parquet",
            )
        )

        # Get SMILES and labels
        smiles = data["SMILES"].astype(str).tolist()
        labels = data.drop(columns=["SMILES"]).values

        # Get functional groups
        fgroups = pd.read_parquet(os.path.join(self.hparams["data_dir"], "training", "fg"))[
            "SMARTS"
        ].tolist()
        fgroups_list = [MolFromSmarts(x) for x in fgroups]

        # Get tokenizer
        tokenizer = Tokenizer.from_file(
            os.path.join(
                self.hparams["data_dir"],
                "training",
                "tokenizers",
                f"BPE_{self.hparams['tokenize_dataset']}_{self.hparams['frequency']}.json",
            )
        )

        # Create dataset
        dataset = FGRDataset(
            smiles,
            labels,
            fgroups_list,
            tokenizer,
            self.hparams["descriptors"],
            self.hparams["method"],
        )
        return dataset

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """

        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams["batch_size"] % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams['batch_size']}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams["batch_size"] // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = self._get_dataset_("train")
            self.data_val = self._get_dataset_("val")
            self.data_test = self._get_dataset_("test")

        if not self.is_regression:
            labels = self.data_train.labels.int()  # type: ignore
            if self.num_classes > 1:
                self.sampler = ClassCycleSampler(labels=labels)  # type: ignore
            else:
                class_weights = 1.0 / torch.tensor(
                    [len(labels[labels == i]) for i in torch.unique(labels)],
                    dtype=torch.float,
                )
                samples_weight = torch.tensor([class_weights[t] for t in labels.int()])
                # Define the sampler
                self.sampler = WeightedRandomSampler(
                    weights=samples_weight, num_samples=len(samples_weight)  # type: ignore
                )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        if not self.is_regression:
            return DataLoader(
                dataset=self.data_train,  # type: ignore[assignment]
                batch_size=self.batch_size_per_device,
                num_workers=self.hparams["num_workers"],
                pin_memory=self.hparams["pin_memory"],
                sampler=self.sampler,
                shuffle=False,
                drop_last=True,
                multiprocessing_context="fork",
            )
        else:
            return DataLoader(
                dataset=self.data_train,  # type: ignore[assignment]
                batch_size=self.batch_size_per_device,
                num_workers=self.hparams["num_workers"],
                pin_memory=self.hparams["pin_memory"],
                shuffle=True,
                drop_last=True,
                multiprocessing_context="fork",
            )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,  # type: ignore[assignment]
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
            multiprocessing_context="fork",
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,  # type: ignore[assignment]
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
            multiprocessing_context="fork",
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


class FGRPretrainDataModule(LightningDataModule):
    """`LightningDataModule` for the FGRPretrain dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/processed",
        dataset: str = "BBBP",
        method: str = "FG",
        frequency: int = 500,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `FGRPretrainDataModule`.

        :param data_dir: Data directory, defaults to "data/processed"
        :param dataset: Dataset to train, defaults to "BBBP"
        :param method: Method of representation, defaults to "FG"
        :param frequency: Frequency for tokenization, defaults to 500
        :param batch_size: Batch size, defaults to 16
        :param num_workers: Number of workers, defaults to 4
        :param pin_memory: Whether to pin memory, defaults to False
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def setup(self, stage=None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """

        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams["batch_size"] % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams['batch_size']}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams["batch_size"] // self.trainer.world_size

        # Get training data
        df = pd.read_parquet(
            os.path.join(
                self.hparams["data_dir"],
                "training",
                self.hparams["dataset"],
            )
        )["SMILES"].tolist()

        # Split data
        train, valid = train_test_split(df, test_size=0.1, random_state=123)

        # Get functional groups
        fgroups = pd.read_parquet(os.path.join(self.hparams["data_dir"], "training", "fg"))[
            "SMARTS"
        ].tolist()
        fgroups_list = [MolFromSmarts(x) for x in fgroups]

        # Get tokenizer
        tokenizer = Tokenizer.from_file(
            os.path.join(
                self.hparams["data_dir"],
                "training",
                "tokenizers",
                f"BPE_{self.hparams['dataset']}_{self.hparams['frequency']}.json",
            )
        )

        # Create datasets
        self.data_train = FGRPretrainDataset(
            train, fgroups_list, tokenizer, self.hparams["method"]
        )
        self.data_val = FGRPretrainDataset(valid, fgroups_list, tokenizer, self.hparams["method"])

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,  # type: ignore[assignment]
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=True,
            drop_last=True,
            multiprocessing_context="fork",
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,  # type: ignore[assignment]
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
            multiprocessing_context="fork",
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = FGRDataModule()
    _ = FGRPretrainDataModule()
