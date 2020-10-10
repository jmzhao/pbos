import gzip
import logging
import os
import shutil
import subprocess as sp

import gensim

from datasets.target_vectors.utils import save_target_dataset, clean_target_emb, convert_target_dataset
from utils import dotdict

logger = logging.getLogger(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
gz_path = f"{dir_path}/embedding.bin.gz"
bin_emb_path = f"{dir_path}/embedding.bin"
txt_emb_path = f"{dir_path}/embedding.txt"
pkl_emb_path = f"{dir_path}/embedding.pkl"
w2v_emb_path = f"{dir_path}/embedding.w2v"
word_list_path = f"{dir_path}/word_list.txt"
word_freq_path = f"{dir_path}/word_freq.jsonl"
raw_count_path = f"{dir_path}/word_freq.txt"


def prepare_google_paths(
    dir_path=dir_path,
    gz_path=gz_path,
    bin_emb_path=bin_emb_path,
    txt_emb_path=txt_emb_path,
    pkl_emb_path=pkl_emb_path,
    word_list_path=word_list_path,
    word_freq_path=word_freq_path,
    w2v_emb_path=w2v_emb_path,
    raw_count_path=raw_count_path,
):
    if not os.path.exists(gz_path):
        url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
        sp.run(f"wget -O {gz_path} {url}".split())

    if not os.path.exists(bin_emb_path):
        with gzip.open(gz_path, "rb") as fin, open(bin_emb_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)

    if not os.path.exists(pkl_emb_path):
        logger.info("loading pre-trained google news vectors...")
        model = gensim.models.KeyedVectors.load_word2vec_format(bin_emb_path, binary=True)
        vocab, emb = clean_target_emb(raw_vocab=list(model.vocab), raw_emb=model.vectors)
        save_target_dataset(vocab, emb, pkl_emb_path=pkl_emb_path)

    convert_target_dataset(
        input_emb_path=pkl_emb_path,

        txt_emb_path=txt_emb_path,
        w2v_emb_path=w2v_emb_path,

        word_list_path=word_list_path,
        word_freq_path=word_freq_path,
        raw_count_path=raw_count_path,
    )

    return dotdict(
        dir_path=dir_path,
        gz_path=gz_path,

        bin_emb_path=bin_emb_path,
        txt_emb_path=txt_emb_path,
        pkl_emb_path=pkl_emb_path,
        w2v_emb_path=w2v_emb_path,

        word_list_path=word_list_path,
        word_freq_path=word_freq_path,
        raw_count_path=raw_count_path,
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    prepare_google_paths()
