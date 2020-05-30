import logging
import os
import shutil
import subprocess as sp
import tarfile

import numpy as np

from datasets.utils import save_emb, save_words, is_normal
from load import load_embedding
from utils import dotdict

logger = logging.getLogger(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))


def prepare_polyglot_emb_paths(language_code, *, dir_path=dir_path):
    language_dir_path = os.path.join(dir_path, language_code)
    tar_path = os.path.join(language_dir_path, "embeddings.tar.bz2")
    pkl_path = os.path.join(language_dir_path, "embeddings.pkl")
    w2v_path = os.path.join(language_dir_path, "embeddings.w2v")
    word_freq_path = os.path.join(language_dir_path, "word_freq.jsonl")
    txt_emb_path = os.path.join(language_dir_path, "embeddings.txt")

    os.makedirs(language_dir_path, exist_ok=True)

    if not os.path.exists(tar_path):
        logger.info(f"Downloading {tar_path}")
        url = f"http://polyglot.cs.stonybrook.edu/~polyglot/embeddings2/{language_code}/embeddings_pkl.tar.bz2"
        sp.run(f"wget -O {tar_path} {url}".split())

    if not os.path.exists(pkl_path):
        logger.info(f"Unzipping {tar_path}")
        with tarfile.open(tar_path) as tar, open(pkl_path, 'wb+') as dst_file:
            src_file = tar.extractfile("./words_embeddings_32.pkl")
            shutil.copyfileobj(src_file, dst_file)

    vocab, emb = load_embedding(pkl_path)
    save_emb(vocab, emb, w2v_emb_path=w2v_path, txt_emb_path=txt_emb_path)
    save_words(vocab, word_freq_path=word_freq_path)

    return dotdict(
        dir_path=dir_path,
        language_dir_path=language_dir_path,
        tar_path=tar_path,
        pkl_path=pkl_path,
        word_freq_path=word_freq_path,
        w2v_path=w2v_path,
        txt_emb_path=txt_emb_path
    )


def _prepare_polyglot_normalized_en_paths(dir_path=dir_path):
    language_dir_path = os.path.join(dir_path, "en")
    raw_en_emb_paths = prepare_polyglot_emb_paths("en", dir_path=dir_path)
    pkl_path = os.path.join(language_dir_path, "normalized_embeddings.pkl")
    w2v_path = os.path.join(language_dir_path, "normalized_embeddings.w2v")
    word_freq_path = os.path.join(language_dir_path, "normalized_word_freq.jsonl")
    txt_emb_path = os.path.join(language_dir_path, "normalized_embeddings.txt")

    raw_vocab, raw_emb = load_embedding(raw_en_emb_paths.pkl_path)
    vocab, emb = [], []

    for w, e in (raw_vocab, raw_emb):
        if is_normal(w):
            vocab.append(w)
            emb.append(e)
    emb = np.array(emb)

    save_emb(vocab, emb, w2v_emb_path=w2v_path, txt_emb_path=txt_emb_path, pkl_emb_path=pkl_path)
    save_words(vocab, word_freq_path=word_freq_path)

    return dotdict(
        dir_path=dir_path,
        language_dir_path=language_dir_path,
        pkl_path=pkl_path,
        word_freq_path=word_freq_path,
        w2v_path=w2v_path,
        txt_emb_path=txt_emb_path
    )


languages = [
    'ar', 'bg', 'cs', 'da', 'el', 'en', 'es', 'eu', 'fa', 'he', 'hi', 'hu',
    'id', 'it', 'kk', 'lv', 'ro', 'ru', 'sv', 'ta', 'tr', 'vi', 'zh',
]

if __name__ == '__main__':
    for language_code in languages:
        prepare_polyglot_emb_paths(language_code)

    _prepare_polyglot_normalized_en_paths()
