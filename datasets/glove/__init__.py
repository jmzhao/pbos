import json
import logging
import zipfile
import subprocess as sp
import os

from utils import dotdict
from load import load_embedding

logger = logging.getLogger(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
zip_path = f"{dir_path}/glove.840B.300d.zip"
txt_emb_path = f"{dir_path}/glove.840B.300d.txt"
w2v_path = f"{dir_path}/glove.840B.300d.w2v"
word_freq_path = f"{dir_path}/word_freq.jsonl"
raw_count_path = f"{dir_path}/word_freq.txt"


def prepare_glove_paths(
    dir_path=dir_path,
    zip_path=zip_path,
    txt_emb_path=txt_emb_path,
    word_freq_path=word_freq_path,
    w2v_path=w2v_path,
    raw_count_path=raw_count_path,
):
    if not os.path.exists(zip_path):
        logger.info("downloading zip file...")
        url = "http://nlp.stanford.edu/data/glove.840B.300d.zip"
        sp.run(f"wget -O {zip_path} {url}".split())

    if not os.path.exists(txt_emb_path):
        logger.info("unzipping...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dir_path)

    if not os.path.exists(w2v_path):
        logger.info("generating w2v file...")
        vocab, emb = load_embedding(txt_emb_path)
        with open(w2v_path, "w") as fout:
            print(len(vocab), len(emb[0]), file=fout)
            for v, e in zip(vocab, emb):
                print(v, *e, file=fout)

    if not os.path.exists(word_freq_path):
        logger.info("generating word freq jsonl file...")
        with open(txt_emb_path) as fin, open(word_freq_path, "w") as fout:
            for line in fin:
                print(json.dumps((line.split()[0], 1)), file=fout)

    if not os.path.exists(raw_count_path):
        logger.info("generating word freq txt file...")
        with open(txt_emb_path) as fin, open(raw_count_path, "w") as fout:
            for line in fin:
                print(line.split()[0], 1, file=fout, sep='\t')

    return dotdict(
        dir_path=dir_path,
        txt_emb_path=txt_emb_path,
        w2v_path=w2v_path,
        word_freq_path=word_freq_path,
        raw_count_path=raw_count_path,
    )


if __name__ == '__main__':
    prepare_glove_paths()
