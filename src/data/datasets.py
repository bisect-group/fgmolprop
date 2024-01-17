from typing import List

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from rdkit.Chem.rdmolfiles import MolFromSmarts
from tokenizers import Tokenizer

from src.data.components.dataset import BaseDataset
from src.data.components.utils import get_descriptors

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


class FGRDataset(BaseDataset):
    """Pytorch dataset for training FGR."""

    def __init__(
        self,
        smiles: List[str],
        labels: np.ndarray,
        fgroups_list: List[MolFromSmarts],
        tokenizer: Tokenizer,
        descriptors: bool,
        method: str,
    ) -> None:
        """Initialize 'FGRDataset'.

        :param smiles: List of SMILES
        :param labels: Label array
        :param fgroups_list: List of functional groups
        :param tokenizer: Pretrained tokenizer
        :param descriptors: Whether to use descriptors
        :param method: Method for training
        """
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.smiles = smiles
        self.fgroups_list = fgroups_list
        self.tokenizer = tokenizer
        self.descriptors = descriptors
        self.method = method

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        smi = self.smiles[idx]  # Get SMILES
        target = self.labels[idx]  # Get label
        x = self._process_smi_(smi)  # Get feature vector
        if self.descriptors:
            descriptors = get_descriptors(smi)  # Get descriptors
            return (
                x,
                descriptors,
                target,
            )  # Return feature vector, descriptors and label
        else:
            return x, target  # Return feature vector and label


class FGRPretrainDataset(BaseDataset):
    """Pytorch dataset for pretraining FGR."""

    def __init__(
        self,
        df: pd.DataFrame,
        method: str,
    ) -> None:
        """Initialize 'FGRPretrainDataset'.

        :param df: Dataframe with SMILES
        :param method: Method for training
        """
        self.df = df
        self.method = method

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fg_idx, mfg_idx = self.df.loc[idx, ["fg", "mfg"]]  # Get SMILES
        if self.method == "FG":
            x = np.zeros(2672, dtype=np.float32)
            x[fg_idx] = 1
        elif self.method == "MFG":
            x = np.zeros(30000, dtype=np.float32)
            x[mfg_idx] = 1
        elif self.method == "FGR":
            f_g = np.zeros(2672, dtype=np.float32)
            f_g[fg_idx] = 1
            mfg = np.zeros(30000, dtype=np.float32)
            mfg[mfg_idx] = 1
            x = np.concatenate((f_g, mfg))  # Concatenate both vectors
        else:
            raise ValueError("Method not supported")  # Raise error if method not supported
        return x
