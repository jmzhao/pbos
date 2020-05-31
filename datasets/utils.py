import json
import logging
import os
import pickle

import numpy as np

from load import load_embedding

logger = logging.getLogger(__name__)


def convert_target_dataset(
    input_emb_path,
    *,
    w2v_emb_path=None,
    txt_emb_path=None,
    pkl_emb_path=None,
    word_list_path=None,
    word_freq_path=None,
    raw_count_path=None,
):
    if all(path is None or os.path.exists(path) for path in
           (w2v_emb_path, txt_emb_path, pkl_emb_path, word_list_path, word_freq_path, raw_count_path)):
        return

    vocab, emb = load_embedding(input_emb_path)

    return save_target_dataset(
        vocab,
        emb,
        w2v_emb_path=w2v_emb_path,
        txt_emb_path=txt_emb_path,
        pkl_emb_path=pkl_emb_path,
        word_list_path=word_list_path,
        word_freq_path=word_freq_path,
        raw_count_path=raw_count_path,
    )


def save_target_dataset(
    vocab,
    emb,
    *,
    w2v_emb_path=None,
    txt_emb_path=None,
    pkl_emb_path=None,
    word_list_path=None,
    word_freq_path=None,
    raw_count_path=None,
):
    if w2v_emb_path and not os.path.exists(w2v_emb_path):
        logger.info("generating w2v emb file...")
        with open(w2v_emb_path, "w") as fout:
            print(len(vocab), len(emb[0]), file=fout)
            for v, e in zip(vocab, emb):
                print(v, *e, file=fout)

    if txt_emb_path and not os.path.exists(txt_emb_path):
        logger.info("generating txt emb file...")
        with open(txt_emb_path, "w") as fout:
            for v, e in zip(vocab, emb):
                print(v, *e, file=fout)

    if pkl_emb_path and not os.path.exists(pkl_emb_path):
        logger.info("generating pkl emb file...")
        emb = emb if isinstance(emb, np.ndarray) else np.array(emb)
        with open(pkl_emb_path, "bw") as fout:
            pickle.dump((vocab, emb), fout)

    if word_list_path and not os.path.exists(word_list_path):
        logger.info("generating word list file...")
        with open(word_list_path, "w") as fout:
            for word in vocab:
                print(word, file=fout)

    if word_freq_path and not os.path.exists(word_freq_path):
        logger.info("generating word freq jsonl file...")
        with open(word_freq_path, "w") as fout:
            for word in vocab:
                print(json.dumps((word, 1)), file=fout)

    if raw_count_path and not os.path.exists(raw_count_path):
        logger.info("generating word freq txt file...")
        with open(raw_count_path, "w") as fout:
            for word in vocab:
                print(word, 1, file=fout, sep='\t')


def _is_word(w):
    return w.isalpha() and w.isascii() # and w.islower()


def clean_target_emb(raw_vocab, raw_emb):
    logger.info("normalizing...")

    vocab, emb = [], []
    for w, e in zip(raw_vocab, raw_emb):
        if _is_word(w):
            vocab.append(w)
            emb.append(e)
    return vocab, emb
