import gzip
import json
import logging
import shutil
import subprocess as sp
import os

from utils import dotdict
from load import load_embedding


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
    dir_path = dir_path,
    gz_path = gz_path,
    bin_emb_path = bin_emb_path,
    txt_emb_path = txt_emb_path,
    word_list_path = word_list_path,
    word_freq_path = word_freq_path,
    w2v_path = w2v_path,
    raw_count_path = raw_count_path,
):

    if not os.path.exists(gz_path):
        url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
        sp.run(f"wget -O {gz_path} {url}".split())

    if not os.path.exists(bin_emb_path):
        with gzip.open(gz_path, "rb") as fin, open(bin_emb_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)

    if not os.path.exists(txt_emb_path):
        sp.run(
            f"python {dir_path}/converter.py --input {bin_emb_path} --output {txt_emb_path}".split()
        )

    if not os.path.exists(w2v_path):
        vocab, emb = load_embedding(txt_emb_path)
        with open(w2v_path, "w") as fout:
            print(len(vocab), len(emb[0]), file=fout)
            for v, e in zip(vocab, emb):
                print(v, *e, file=fout)

    if not os.path.exists(word_list_path):
        with open(txt_emb_path) as fin, open(word_list_path, "w") as fout:
            for line in fin:
                print(line.split()[0], file=fout)

    if not os.path.exists(word_freq_path):
        with open(txt_emb_path) as fin, open(word_freq_path, "w") as fout:
            for line in fin:
                print(json.dumps((line.split()[0], 1)), file=fout)

    if not os.path.exists(raw_count_path):
            with open(txt_emb_path) as fin, open(raw_count_path, "w") as fout:
                for line in fin:
                    print(line.split()[0], 1, file=fout, sep='\t')

    return dotdict(
        dir_path=dir_path,
        gz_path = gz_path,
        bin_emb_path = bin_emb_path,
        txt_emb_path = txt_emb_path,
        word_list_path = word_list_path,
        word_freq_path = word_freq_path,
        w2v_path = w2v_path,
        raw_count_path = raw_count_path,
    )