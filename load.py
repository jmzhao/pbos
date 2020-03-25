import logging
from typing import List

import numpy as np


def load_embedding(filename: str, show_progress=False) -> (List[str], np.ndarray):
    """
    :param filename: a .txt file or a .pkl/.pickle file
    :return: tuple (words, embeddings)
    """
    import os
    if show_progress:
        from utils import file_tqdm
    else:
        from utils import dummy_tqdm as file_tqdm

    _, ext = os.path.splitext(filename)
    if ext in (".txt",):
        vocab, emb = [], []
        with open(filename, "r") as fin:
            for line in file_tqdm(fin):
                ss = line.split()
                vocab.append(ss[0])
                emb.append([float(x) for x in ss[1:]])
        emb = np.array(emb)
    elif ext in (".pickle", ".pkl"):
        import pickle
        vocab, emb = pickle.load(open(filename, 'rb'))
    else:
        raise ValueError(f'Unsupported target vector file extent: {filename}')

    return vocab, emb
