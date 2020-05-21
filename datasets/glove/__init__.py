import json
import logging
import unicodedata
import zipfile
import subprocess as sp
import os

from utils import dotdict, file_tqdm

logger = logging.getLogger(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
zip_path = f"{dir_path}/glove.840B.300d.zip"
raw_emb_path = f"{dir_path}/glove.840B.300d.txt"
txt_emb_path = f"{dir_path}/glove.840B.300d.processed.txt"
w2v_path = f"{dir_path}/glove.840B.300d.processed.w2v"
word_freq_path = f"{dir_path}/word_freq.jsonl"
raw_count_path = f"{dir_path}/word_freq.txt"

emb_dim = 300


def prepare_glove_paths(
    dir_path=dir_path,
    zip_path=zip_path,
    raw_emb_path=raw_emb_path,
    txt_emb_path=txt_emb_path,
    word_freq_path=word_freq_path,
    w2v_path=w2v_path,
    raw_count_path=raw_count_path,
):
    if not os.path.exists(zip_path):
        logger.info("downloading zip file...")
        url = "http://nlp.stanford.edu/data/glove.840B.300d.zip"
        sp.run(f"wget -O {zip_path} {url}".split())

    if not os.path.exists(raw_emb_path):
        logger.info("unzipping...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dir_path)

    # glove includes ~20 words with space which is hard to handle in current codebase
    # need to clean them from `raw_emb`
    if not os.path.exists(txt_emb_path):
        logger.info("generating txt emb file...")
        with open(raw_emb_path, "r") as fin, open(txt_emb_path, "w") as fout:
            vocab_len = 0
            for line in file_tqdm(fin):
                ss = line.split()
                if len(ss) != emb_dim + 1:
                    logging.critical(f'line "{line[:30]}"... might include word with space, skipped')
                    continue

                w = ss[0]

                # copied from `datasets/google/converter.py`
                aw = unicodedata.normalize("NFKD", w).encode("ASCII", "ignore")
                if 20 > len(aw) > 1 and not any(c in w for c in " _./") and aw.islower():
                    vocab_len += 1
                    fout.write(line)

    if not os.path.exists(w2v_path):
        logger.info("generating w2v emb file...")
        with open(txt_emb_path) as fin, open(w2v_path, "w") as fout:
            print(vocab_len, emb_dim, file=fout)
            for line in file_tqdm(fin):
                fout.write(line)

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
        raw_emb_path=raw_emb_path,
        txt_emb_path=txt_emb_path,
        w2v_path=w2v_path,
        word_freq_path=word_freq_path,
        raw_count_path=raw_count_path,
    )


if __name__ == '__main__':
    prepare_glove_paths()
