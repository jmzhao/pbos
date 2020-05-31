import json
import logging
import os
import pickle

import numpy as np

from load import load_embedding

logger = logging.getLogger(__name__)


def save_raw_count(vocab, raw_count_path):
    logger.info("generating word freq txt file...")
    with open(raw_count_path, "w") as fout:
        for word in vocab:
            print(word, 1, file=fout, sep='\t')


def save_word_freq(vocab, word_freq_path):
    logger.info("generating word freq jsonl file...")
    with open(word_freq_path, "w") as fout:
        for word in vocab:
            print(json.dumps((word, 1)), file=fout)


def save_word_list(vocab, word_list_path):
    logger.info("generating word list file...")
    with open(word_list_path, "w") as fout:
        for word in vocab:
            print(word, file=fout)


def save_pkl_emb(emb, vocab, pkl_emb_path):
    logger.info("generating pkl emb file...")
    emb = emb if isinstance(emb, np.ndarray) else np.array(emb)
    with open(pkl_emb_path, "bw") as fout:
        pickle.dump((vocab, emb), fout)


def save_txt_emb(emb, vocab, txt_emb_path):
    logger.info("generating txt emb file...")
    with open(txt_emb_path, "w") as fout:
        for v, e in zip(vocab, emb):
            print(v, *e, file=fout)


def save_w2v_emb(emb, vocab, w2v_emb_path):
    logger.info("generating w2v emb file...")
    with open(w2v_emb_path, "w") as fout:
        print(len(vocab), len(emb[0]), file=fout)
        for v, e in zip(vocab, emb):
            print(v, *e, file=fout)


def convert_target_dataset(
    vocab,
    emb,
    input_emb_path,
    *,
    w2v_emb_path,
    txt_emb_path,
    pkl_emb_path,
    word_list_path,
    word_freq_path,
    raw_count_path
):
    """
    Convert target dataset, need to provide either `vocab` + `emb` or `input_emb_path`
    """
    if all(path is None or os.path.exists(path) for path in
           (w2v_emb_path, txt_emb_path, pkl_emb_path, word_list_path, word_freq_path, raw_count_path)):
        return

    if vocab is not None and emb is not None:
        if input_emb_path is not None:
            raise ValueError("Should not provide both vocab/emb and  input_emb_path")
    else:
        vocab, emb = load_embedding(input_emb_path)

    if w2v_emb_path and not os.path.exists(w2v_emb_path):
        save_w2v_emb(emb, vocab, w2v_emb_path)

    if txt_emb_path and not os.path.exists(txt_emb_path):
        save_txt_emb(emb, vocab, txt_emb_path)

    if pkl_emb_path and not os.path.exists(pkl_emb_path):
        save_pkl_emb(emb, vocab, pkl_emb_path)

    if word_list_path and not os.path.exists(word_list_path):
        save_word_list(vocab, word_list_path)

    if word_freq_path and not os.path.exists(word_freq_path):
        save_word_freq(vocab, word_freq_path)

    if raw_count_path and not os.path.exists(raw_count_path):
        save_raw_count(vocab, raw_count_path)
