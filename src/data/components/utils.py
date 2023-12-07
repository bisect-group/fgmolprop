from typing import List

import numpy as np
import pandas as pd
import tokenizers
from molvs import standardize_smiles
from rdkit import RDLogger
from rdkit.Chem.rdmolfiles import MolFromSmarts, MolFromSmiles, MolToSmarts

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def std_smiles(smiles: str) -> str | None:
    """Standardizes a SMILES string.

    :param smiles: SMILES string
    :return: Standardized SMILES string
    """
    try:
        standard = standardize_smiles(smiles)
        MolFromSmiles(standard)
        return standard
    except BaseException:
        return None


def std_smarts(smarts: str) -> str | None:
    """Standardizes a SMARTS string.

    :param smarts: SMARTS string
    :return: Standardized SMARTS string
    """
    try:
        mol = MolFromSmarts(smarts)
        return MolToSmarts(mol)
    except BaseException:
        return None


def smiles2vector_fg(smi: str, fgroups_list: List[MolFromSmarts]) -> np.ndarray:
    """Converts a SMILES string to a functional group vector.

    :param smi: SMILES string
    :param fgroups_list: List of functional groups
    :return: Feature vector
    """
    molecule = MolFromSmiles(smi)
    v_1 = np.zeros(len(fgroups_list), dtype="f")
    for idx, f_g in enumerate(fgroups_list):
        if molecule.HasSubstructMatch(f_g):
            v_1[idx] = 1
    return v_1


def smiles2vector_mfg(smi: str, tokenizer: tokenizers.Tokenizer) -> np.ndarray:
    """Converts a SMILES string to a molecular fingerprint vector.

    :param smi: SMILES string
    :param tokenizer: Pretrained tokenizer
    :return: Feature vector
    """
    idx = tokenizer.encode(smi).ids
    mfg = np.zeros(tokenizer.get_vocab_size(), dtype="f")
    mfg[idx] = 1
    return mfg


def create_butina_split(
    df: pd.DataFrame, seed: int, frac: tuple, entity: str, cutoff: float = 0.4
) -> dict:
    """Creates a Butina split.

    :param df: Dataframe with SMILES
    :param seed: Random seed
    :param frac: Fraction of data for train, val and test
    :param entity: Entity to split on
    :param cutoff: Cutoff for clustering, defaults to 0.4
    :raises ImportError: If rdkit is not installed
    :return: Dictionary with train, val and test dataframes
    """

    try:
        from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
        from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity
        from rdkit.ML.Cluster import Butina

        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        raise ImportError("Please install rdkit by 'conda install -c conda-forge rdkit'! ")

    from random import Random

    from tqdm import tqdm

    random = Random(seed)

    s = df[entity].to_list()

    mols = []
    for smiles in tqdm(s):
        mols.append(MolFromSmiles(smiles))
    fps = [GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]

    # calcaulate scaffold sets
    # (ytz): this is directly copypasta'd from Greg Landrum's clustering example.
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    scaffold_sets = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    index_sets = sorted(scaffold_sets, key=lambda x: -len(x))

    train, val, test = [], [], []
    train_size = int((len(df)) * frac[0])
    val_size = int((len(df)) * frac[1])
    test_size = (len(df)) - train_size - val_size
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    big_index_sets = []
    small_index_sets = []
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)
    random.seed(seed)
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)
    index_sets = big_index_sets + small_index_sets

    if frac[2] == 0:
        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
                train_scaffold_count += 1
            else:
                val += index_set
                val_scaffold_count += 1
    else:
        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
                train_scaffold_count += 1
            elif len(val) + len(index_set) <= val_size:
                val += index_set
                val_scaffold_count += 1
            else:
                test += index_set
                test_scaffold_count += 1

    return {
        "train": df.iloc[train].reset_index(drop=True),
        "valid": df.iloc[val].reset_index(drop=True),
        "test": df.iloc[test].reset_index(drop=True),
    }
