import logging
import os
import shutil
import subprocess as sp
import tarfile

from datasets.utils import save_target_dataset, clean_target_emb, convert_target_dataset
from load import load_embedding
from utils import dotdict

logger = logging.getLogger(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))


def prepare_polyglot_emb_paths(language_code, *, dir_path=dir_path):
    language_dir_path = os.path.join(dir_path, language_code)
    tar_path = os.path.join(language_dir_path, "embeddings.tar.bz2")
    pkl_emb_path = os.path.join(language_dir_path, "embeddings.pkl")
    w2v_emb_path = os.path.join(language_dir_path, "embeddings.w2v")
    word_freq_path = os.path.join(language_dir_path, "word_freq.jsonl")
    txt_emb_path = os.path.join(language_dir_path, "embeddings.txt")

    os.makedirs(language_dir_path, exist_ok=True)

    if not os.path.exists(tar_path):
        logger.info(f"Downloading {tar_path}")
        url = f"http://polyglot.cs.stonybrook.edu/~polyglot/embeddings2/{language_code}/embeddings_pkl.tar.bz2"
        sp.run(f"wget -O {tar_path} {url}".split())

    if not os.path.exists(pkl_emb_path):
        logger.info(f"Unzipping {tar_path}")
        with tarfile.open(tar_path) as tar, open(pkl_emb_path, 'wb+') as dst_file:
            src_file = tar.extractfile("./words_embeddings_32.pkl")
            shutil.copyfileobj(src_file, dst_file)

    convert_target_dataset(
        input_emb_path=pkl_emb_path,

        txt_emb_path=txt_emb_path,
        w2v_emb_path=w2v_emb_path,

        word_freq_path=word_freq_path,
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


def prepare_polyglot_clear_en_paths(dir_path=dir_path):
    language_dir_path = os.path.join(dir_path, "en")
    raw_en_emb_paths = prepare_polyglot_emb_paths("en", dir_path=dir_path)

    pkl_emb_path = os.path.join(language_dir_path, "embeddings.clear.pkl")
    w2v_emb_path = os.path.join(language_dir_path, "embeddings.clear.w2v")
    txt_emb_path = os.path.join(language_dir_path, "embeddings.clear.txt")
    word_freq_path = os.path.join(language_dir_path, "word_freq.clear.jsonl")

    if not os.path.exists(pkl_emb_path):
        raw_vocab, raw_emb = load_embedding(raw_en_emb_paths.pkl_emb_path)
        vocab, emb = clean_target_emb(raw_vocab, raw_emb)
        save_target_dataset(vocab, emb, pkl_emb_path=pkl_emb_path)

    convert_target_dataset(
        input_emb_path=pkl_emb_path,

        txt_emb_path=txt_emb_path,
        w2v_emb_path=w2v_emb_path,

        word_freq_path=word_freq_path,
    )

    return dotdict(
        dir_path=dir_path,
        language_dir_path=language_dir_path,
        pkl_emb_path=pkl_emb_path,
        w2v_emb_path=w2v_emb_path,
        txt_emb_path=txt_emb_path,
        word_freq_path=word_freq_path,
    )


languages = [
    'ar', 'bg', 'cs', 'da', 'el', 'en', 'es', 'eu', 'fa', 'he', 'hi', 'hu',
    'id', 'it', 'kk', 'lv', 'ro', 'ru', 'sv', 'ta', 'tr', 'vi', 'zh',
]

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    for language_code in languages:
        prepare_polyglot_emb_paths(language_code)

    prepare_polyglot_clear_en_paths()
