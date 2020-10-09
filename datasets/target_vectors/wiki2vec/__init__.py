import logging
import os
import subprocess as sp

from datasets.target_vectors.utils import convert_target_dataset
from utils import dotdict

logger = logging.getLogger(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))


def prepare_wiki2vec_emb_paths(language_code, *, dir_path=dir_path):
    language_dir_path = os.path.join(dir_path, language_code)
    download_path = os.path.join(language_dir_path, "raw_embeddings.w2v.bz2")
    pkl_emb_path = os.path.join(language_dir_path, "embeddings.pkl")
    w2v_emb_path = os.path.join(language_dir_path, "embeddings.w2v")
    word_freq_path = os.path.join(language_dir_path, "word_freq.jsonl")
    txt_emb_path = os.path.join(language_dir_path, "embeddings.txt")

    os.makedirs(language_dir_path, exist_ok=True)

    if not os.path.exists(download_path):
        url = f"http://wikipedia2vec.s3.amazonaws.com/models/{language_code}/2018-04-20/{language_code}wiki_20180420_300d.txt.bz2"
        logger.info(f"Downloading {url} to {download_path}")
        sp.run(f"wget -O {download_path} {url}".split())

    if not os.path.exists(w2v_emb_path):
        logger.info(f"Unzipping {download_path}")
        sp.run(f"bzip2 -dk {download_path}".split())
        os.system(f"head -n 100001 {download_path[:-4]} > {w2v_emb_path}")  # keep 100k tokens and one line of header


    convert_target_dataset(
        input_emb_path=w2v_emb_path,

        txt_emb_path=txt_emb_path,
        pkl_emb_path=pkl_emb_path,

        word_freq_path=word_freq_path,
    )

    return dotdict(
        dir_path=dir_path,
        language_dir_path=language_dir_path,
        download_path=download_path,

        pkl_emb_path=pkl_emb_path,
        w2v_emb_path=w2v_emb_path,
        txt_emb_path=txt_emb_path,
        word_freq_path=word_freq_path,
    )


languages = ["en", "it", "ru", "de"]

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    for language_code in languages:
        prepare_wiki2vec_emb_paths(language_code)
