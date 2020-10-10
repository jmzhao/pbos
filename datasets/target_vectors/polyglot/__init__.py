import logging
import os
import shutil
import subprocess as sp
import tarfile

from datasets.target_vectors.utils import convert_target_dataset
from utils import dotdict

logger = logging.getLogger(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))


def prepare_polyglot_emb_paths(lang, *, dir_path=dir_path):
    if lang not in polyglot_languages:
        raise NotImplementedError

    language_dir_path = os.path.join(dir_path, lang)
    tar_path = os.path.join(language_dir_path, "embeddings.tar.bz2")
    raw_pkl_emb_path = os.path.join(language_dir_path, "raw_embeddings.pkl")
    pkl_emb_path = os.path.join(language_dir_path, "embeddings.pkl")
    w2v_emb_path = os.path.join(language_dir_path, "embeddings.w2v")
    word_freq_path = os.path.join(language_dir_path, "word_freq.jsonl")
    txt_emb_path = os.path.join(language_dir_path, "embeddings.txt")

    os.makedirs(language_dir_path, exist_ok=True)

    if not os.path.exists(tar_path):
        logger.info(f"Downloading {tar_path}")
        url = f"http://polyglot.cs.stonybrook.edu/~polyglot/embeddings2/{lang}/embeddings_pkl.tar.bz2"
        sp.run(f"wget -O {tar_path} {url}".split())

    if not os.path.exists(raw_pkl_emb_path):
        logger.info(f"Unzipping {tar_path}")
        with tarfile.open(tar_path) as tar, open(raw_pkl_emb_path, 'wb+') as dst_file:
            src_file = tar.extractfile("./words_embeddings_32.pkl")
            shutil.copyfileobj(src_file, dst_file)

    convert_target_dataset(
        input_emb_path=raw_pkl_emb_path,

        txt_emb_path=txt_emb_path,
        w2v_emb_path=w2v_emb_path,
        pkl_emb_path=pkl_emb_path,
        word_freq_path=word_freq_path,

        # vocab_size_limit=10_000
    )

    return dotdict(
        dir_path=dir_path,
        language_dir_path=language_dir_path,
        tar_path=tar_path,

        pkl_emb_path=pkl_emb_path,
        w2v_emb_path=w2v_emb_path,
        txt_emb_path=txt_emb_path,
        word_freq_path=word_freq_path,
    )


polyglot_languages = [
    'ar', 'bg', 'cs', 'da', 'el', 'en', 'es', 'eu', 'fa', 'he', 'hi', 'hu',
    'id', 'it', 'kk', 'lv', 'ro', 'ru', 'sv', 'ta', 'tr', 'vi', 'zh',
]

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    for language_code in polyglot_languages:
        prepare_polyglot_emb_paths(language_code)
