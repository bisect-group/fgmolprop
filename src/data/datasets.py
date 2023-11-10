from typing import List

import numpy as np
import torch
from rdkit import RDLogger
from rdkit.Chem.rdmolfiles import MolFromSmarts
from tokenizers import Tokenizer

from src.data.components.dataset import BaseDataset

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
            descriptors = self._get_descriptors_(smi)  # Get descriptors
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
        smiles: List[str],
        fgroups_list: List[MolFromSmarts],
        tokenizer: Tokenizer,
        method: str,
    ) -> None:
        """Initialize 'FGRPretrainDataset'.

        :param smiles: List of SMILES
        :param fgroups_list: List of functional groups
        :param tokenizer: Pretrained tokenizer
        :param method: Method for training
        """
        self.smiles = smiles
        self.fgroups_list = fgroups_list
        self.tokenizer = tokenizer
        self.method = method

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]  # Get SMILES
        x = self._process_smi_(smi)  # Get feature vector
        return x
