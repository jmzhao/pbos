import logging
from typing import Dict, List

import numpy as np


def load_vocab(filename: str, cutoff=5, min_len=1, max_len=None, boundary=False) -> Dict[str, int]:
    """
    :param filename: a .txt file
    :return: dictionary {word: count}
    """
    import collections
    from .preprocess import get_substrings

    part_count = collections.defaultdict(int)
    with open(filename, "r") as f:
        for line in f:
            rows = line.strip().split(",")
            part = rows[0]
            if boundary:
                part = '<' + part + '>'
            count = int(rows[1]) if len(rows) == 2 else 1
            for part in get_substrings(part, min_len=min_len, max_len=min_len):
                part_count[part] += count

    return {k:v for k, v in part_count.items() if v >= cutoff}


def load_embedding(filename: str) -> (List[str], np.ndarray):
    """
    :param filename: a .txt file or a .pkl/.pickle file
    :return: tuple (words, embeddings)
    """
    import os

    logging.info(f'loading embeddings from {filename} ...')

    _, ext = os.path.splitext(filename)
    if ext in (".txt",):
        vocab, emb = [], []
        with open(filename, "r") as f:
            for line in f:
                ss = line.split()
                vocab.append(ss[0])
                emb.append([float(x) for x in ss[1:]])
        emb = np.array(emb)
    elif ext in (".pickle", ".pkl"):
        import pickle
        vocab, emb = pickle.load(open(filename, 'rb'))
    else:
        raise ValueError(f'Unsupported target vector file extent: {filename}')

    logging.info(f'embeddings loaded with {len(vocab)} words')

    return vocab, emb
