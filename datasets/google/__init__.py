import gzip
import logging
import os
import shutil
import subprocess as sp

import gensim

from datasets.utils import save_emb, save_words, is_word
from load import load_embedding
from utils import dotdict

logger = logging.getLogger(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
gz_path = f"{dir_path}/embedding.bin.gz"
bin_emb_path = f"{dir_path}/embedding.bin"
txt_emb_path = f"{dir_path}/embedding.txt"
w2v_path = f"{dir_path}/embedding.w2v"
word_list_path = f"{dir_path}/word_list.txt"
word_freq_path = f"{dir_path}/word_freq.jsonl"
raw_count_path = f"{dir_path}/word_freq.txt"


def prepare_google_paths(
    dir_path=dir_path,
    gz_path=gz_path,
    bin_emb_path=bin_emb_path,
    txt_emb_path=txt_emb_path,
    word_list_path=word_list_path,
    word_freq_path=word_freq_path,
    w2v_path=w2v_path,
    raw_count_path=raw_count_path,
):
    if not os.path.exists(gz_path):
        url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
        sp.run(f"wget -O {gz_path} {url}".split())

    if not os.path.exists(bin_emb_path):
        with gzip.open(gz_path, "rb") as fin, open(bin_emb_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)

    if not os.path.exists(txt_emb_path):
        # Load Google's pre-trained Word2Vec model.
        logging.info("loading...")
        model = gensim.models.KeyedVectors.load_word2vec_format(bin_emb_path, binary=True)

        logging.info("normalizing...")
        vocab, emb = [], []
        for w in model.vocab:
            if is_word(w):
                vocab.append(w)
                emb.append(model[w])

    vocab, emb = load_embedding(txt_emb_path)
    save_emb(vocab, emb, w2v_emb_path=w2v_path)
    save_words(vocab, word_list_path=word_list_path, word_freq_path=word_freq_path, raw_count_path=raw_count_path)

    return dotdict(
        dir_path=dir_path,
        gz_path=gz_path,
        bin_emb_path=bin_emb_path,
        txt_emb_path=txt_emb_path,
        word_list_path=word_list_path,
        word_freq_path=word_freq_path,
        w2v_path=w2v_path,
        raw_count_path=raw_count_path,
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    prepare_google_paths()
