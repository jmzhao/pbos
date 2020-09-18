import logging
import os
import subprocess as sp

from datasets.utils import convert_target_dataset
from utils import dotdict

logger = logging.getLogger(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))


def prepare_wiki2vec_emb_paths(language_code, *, dir_path=dir_path):
    language_dir_path = os.path.join(dir_path, language_code)
    tar_path = os.path.join(language_dir_path, "embeddings.tar.bz2")
    pkl_emb_path = os.path.join(language_dir_path, "embeddings.pkl")
    w2v_emb_path = os.path.join(language_dir_path, "embeddings.w2v")
    word_freq_path = os.path.join(language_dir_path, "word_freq.jsonl")
    txt_emb_path = os.path.join(language_dir_path, "embeddings.txt")

    os.makedirs(language_dir_path, exist_ok=True)

    if not os.path.exists(tar_path):
        logger.info(f"Downloading {tar_path}")
        url = f"http://wikipedia2vec.s3.amazonaws.com/models/{language_code}/2018-04-20/{language_code}wiki_20180420_300d.pkl.bz2"
        sp.run(f"wget -O {tar_path} {url}".split())

    # if not os.path.exists(pkl_emb_path):
    #     logger.info(f"Unzipping {tar_path}")
    #     with tarfile.open(tar_path) as tar, open(pkl_emb_path, 'wb+') as dst_file:
    #         src_file = tar.extractfile("./words_embeddings_32.pkl")
    #         shutil.copyfileobj(src_file, dst_file)

    convert_target_dataset(
        input_emb_path=pkl_emb_path,

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


languages = ["en", "it", "ru", "de"]

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    for language_code in languages:
        prepare_wiki2vec_emb_paths(language_code)
