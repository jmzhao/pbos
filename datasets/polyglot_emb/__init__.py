import json
import logging
import os
import shutil
import subprocess as sp
import tarfile

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

    if not os.path.exists(word_freq_path):
        vocab, emb = load_embedding(pkl_path)
        with open(word_freq_path, "w") as fout:
            for word in vocab:
                print(json.dumps((word, 1)), file=fout)

    if not os.path.exists(w2v_path):
        vocab, emb = load_embedding(pkl_path)
        with open(w2v_path, "w") as fout:
            print(len(vocab), len(emb[0]), file=fout)
            for v, e in zip(vocab, emb):
                print(v, *e, file=fout)

    if not os.path.exists(txt_emb_path):
        vocab, emb = load_embedding(pkl_path)
        with open(txt_emb_path, "w") as fout:
            for v, e in zip(vocab, emb):
                print(v, *e, file=fout)

    return dotdict(
        dir_path = dir_path,
        language_dir_path = language_dir_path,
        tar_path = tar_path,
        pkl_path = pkl_path,
        word_freq_path = word_freq_path,
        w2v_path = w2v_path,
        txt_emb_path = txt_emb_path
    )


languages = [
    'ar', 'bg', 'cs', 'da', 'el', 'en', 'es', 'eu', 'fa', 'he', 'hi', 'hu',
    'id', 'it', 'kk', 'lv', 'ro', 'ru', 'sv', 'ta', 'tr', 'vi', 'zh',
]


if __name__ == '__main__':
    for language_code in languages:
        prepare_polyglot_emb_paths(language_code)
