import os

import pytest
import rootutils
import torch

from src.data.datamodules import FGRDataModule, FGRPretrainDataModule

root = rootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


@pytest.mark.parametrize("method", ["FG", "MFG", "FGR"])  # Parametrize the method
@pytest.mark.parametrize("descriptors", [True, False])  # Parametrize the descriptors
@pytest.mark.parametrize(
    "dataset", ["BBBP", "Tox21", "BACE", "ESOL", "qm7"]
)  # Parametrize the dataset
@pytest.mark.parametrize("fold_idx", [0, 1, 2, 3, 4])  # Parametrize the dataset
@pytest.mark.parametrize(
    "split_type", ["scaffold", "random", "butina"]
)  # Parametrize the dataset split type
def test_fgr_datamodule(
    method: str, descriptors: bool, dataset: str, fold_idx: int, split_type: str
):
    """Tests `FGRDataModule` to verify that it can be downloaded correctly, that the necessary
    functions were created.

    :param method: Method of representation.
    :param descriptors: Whether to use descriptors.
    :param dataset: Dataset to train.
    :param fold_idx: Fold index.
    :param split_type: Dataset split type.
    """
    data_dir = "data/processed"
    batch_size = 16

    dm = FGRDataModule(
        data_dir=os.path.join(root, data_dir),
        dataset=dataset,
        descriptors=descriptors,
        method=method,
        fold_idx=fold_idx,
        split_type=split_type,
    )

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    if descriptors:
        x, num_feat, y = next(iter(dm.train_dataloader()))  # Get the first batch
        assert num_feat.shape[0] == batch_size  # type: ignore  # Check the batch size
        assert num_feat.shape[1] == 211  # type: ignore  # Check the feature size
        assert num_feat.dtype == torch.float32  # Check the dtype
    else:
        x, y = next(iter(dm.train_dataloader()))  # Get the first batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


@pytest.mark.parametrize("method", ["FG", "MFG", "FGR"])  # Parametrize the method
@pytest.mark.parametrize("dataset", ["pubchem", "chembl"])  # Parametrize the dataset
def test_fgrpretrain_datamodule(method: str, dataset: str):
    """Tests `FGRPretrainDataModule` to verify that it can be downloaded correctly, that the
    necessary functions were created.

    :param method: Method of representation.
    :param dataset: Dataset to train.
    """
    data_dir = "data/processed"
    batch_size = 16

    dm = FGRPretrainDataModule(
        data_dir=os.path.join(root, data_dir),
        dataset=dataset,
        method=method,
    )

    dm.setup()
    assert dm.data_train and dm.data_val
    assert dm.train_dataloader() and dm.val_dataloader()

    x = next(iter(dm.train_dataloader()))  # Get the first batch
    assert len(x) == batch_size
    assert x.dtype == torch.float32
