from typing import Any, List

import numpy as np
from rdkit.Chem import Descriptors
from rdkit.Chem.rdmolfiles import MolFromSmarts, MolFromSmiles
from tokenizers import Tokenizer
from torch.utils.data import Dataset

from src.data.components.utils import smiles2vector_fg, smiles2vector_mfg


class BaseDataset(Dataset):
    """Base dataset class for creating custom datasets."""

    def __init__(
        self,
        method: str,
        fgroups_list: List[MolFromSmarts],
        tokenizer: Tokenizer,
    ) -> None:
        """Initialize 'BaseDataset'.

        :param method: Method for training
        :param fgroups_list: List of functional groups
        :param tokenizer: Pretrained tokenizer
        """

        self.method = method
        self.fgroups_list = fgroups_list
        self.tokenizer = tokenizer

    def _process_smi_(self, smi: str) -> np.ndarray:
        """Process SMILES string.

        :param smi: SMILES string
        :raises ValueError: If method not supported
        :return: Feature vector
        """
        if self.method == "FG":
            x = smiles2vector_fg(smi, self.fgroups_list)  # Get functional group vector
        elif self.method == "MFG":
            x = smiles2vector_mfg(smi, self.tokenizer)  # Get mined functional group vector
        elif self.method == "FGR":
            f_g = smiles2vector_fg(smi, self.fgroups_list)  # Get functional group vector
            mfg = smiles2vector_mfg(smi, self.tokenizer)  # Get mined functional group vector
            x = np.concatenate((f_g, mfg))  # Concatenate both vectors
        else:
            raise ValueError("Method not supported")  # Raise error if method not supported
        return x

    def _get_descriptors_(self, smi: str) -> np.ndarray:
        """Get descriptors from SMILES string.

        :param smi: SMILES string
        :return: Descriptor vector
        """
        mol = MolFromSmiles(smi)  # Get molecule from SMILES string

        # Get descriptors
        desc_list = []
        for _, func in Descriptors._descList:
            try:
                desc_list.append(func(mol))
            except BaseException:
                desc_list.append(0)
        descriptors = np.asarray(desc_list)
        descriptors = np.nan_to_num(
            descriptors, nan=0.0, posinf=0.0, neginf=0.0
        )  # Replace NaNs with 0
        descriptors = descriptors / np.linalg.norm(descriptors)  # Normalize
        return descriptors.astype(np.float32)

    def __getitem__(self, index: int) -> Any:
        """Get item from dataset.

        :param index: Index of item
        :raises NotImplementedError: If method not implemented.
        :return: Item
        """
        raise NotImplementedError()

    def __len__(self) -> int:
        """Get length of dataset.

        :raises NotImplementedError: If method not implemented.
        :return: Length of dataset
        """
        raise NotImplementedError()
