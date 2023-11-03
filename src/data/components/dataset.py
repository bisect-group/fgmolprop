from typing import Any, List

import numpy as np
from rdkit.Chem import Descriptors
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import MolFromSmiles
from tokenizers import Tokenizer
from torch.utils.data import Dataset

from src.data.components.utils import smiles2vector_fg, smiles2vector_mfg


class BaseDataset(Dataset):
    """Base dataset class."""

    def __init__(
        self,
        method: str,
        fgroups_list: List[Mol],
        tokenizer: Tokenizer,
    ) -> None:
        """Base dataset class.

        Args:
            method (str): Method for representation learning.
            fgroups_list (List[Mol]): List of functional groups.
            tokenizer (Tokenizer): Pretrained tokenizer.
        """

        self.method = method
        self.fgroups_list = fgroups_list
        self.tokenizer = tokenizer

    def _process_smi_(self, smi: str) -> np.ndarray:
        """Process SMILES string.

        Args:
            smi (str): SMILES string.

        Raises:
            ValueError: If method not supported.

        Returns:
            np.ndarray: Feature vector.
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

        Args:
            smi (str): SMILES string.

        Returns:
            np.ndarray: Descriptor vector.
        """
        mol = MolFromSmiles(smi)  # Get molecule from SMILES string
        descriptors = np.asarray([func(mol) for _, func in Descriptors._descList])
        descriptors = np.nan_to_num(
            descriptors, nan=0.0, posinf=0.0, neginf=0.0
        )  # Replace NaNs with 0
        descriptors = descriptors / np.linalg.norm(descriptors)  # Normalize
        descriptors = descriptors.astype(np.float32)  # Convert to float32
        return descriptors

    def __getitem__(self, index: int) -> Any:
        """Get item from dataset.

        Args:
            index (int): Index of item.

        Raises:
            NotImplementedError: If method not implemented.

        Returns:
            Any: Item.
        """
        raise NotImplementedError()

    def __len__(self) -> int:
        """Get length of dataset.

        Raises:
            NotImplementedError: If method not implemented.

        Returns:
            int: Length of dataset.
        """
        raise NotImplementedError()
