import json
import logging
import os
import pickle

import numpy as np

logger = logging.getLogger(__name__)


def save_emb(vocab, emb, w2v_emb_path=None, txt_emb_path=None, pkl_emb_path=None):
    emb = emb if isinstance(emb, np.ndarray) else np.array(emb)

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
        with open(pkl_emb_path, "bw") as fout:
            pickle.dump((vocab, emb), fout)


def save_words(vocab, word_list_path=None, word_freq_path=None, raw_count_path=None):
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


def is_word(w):
    return w.isalpha() and w.isascii() and w.islower()
