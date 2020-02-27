import os
import subprocess as sp
import gzip
import shutil

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_google_news_paths():
    gz_path = f"{dir_path}/embedding.bin.gz"
    bin_emb_path = f"{dir_path}/embedding.bin"
    txt_emb_path = f"{dir_path}/embedding.txt"
    wordlist_path = f"{dir_path}/wordlist.txt"

    if not os.path.exists(gz_path):
        url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
        sp.run(f"wget -O {gz_path} {url}".split())

    if not os.path.exists(bin_emb_path):
        with gzip.open(gz_path, "rb") as f_in, open(bin_emb_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    if not os.path.exists(txt_emb_path):
        sp.run(
            f"python {dir_path}/converter.py --input {bin_emb_path} --output {txt_emb_path}".split()
        )

    if not os.path.exists(wordlist_path):
        with open(txt_emb_path) as f, open(wordlist_path, "w") as fout:
            for line in f:
                print(line.split()[0], file=fout)

    return txt_emb_path, wordlist_path
