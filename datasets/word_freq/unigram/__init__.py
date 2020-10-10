import json
import logging
import subprocess as sp
import os

from utils import dotdict

logger = logging.getLogger(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
csv_path = f"{dir_path}/unigram_freq.csv"
word_freq_path = f"{dir_path}/word_freq.jsonl"


def prepare_unigram_freq_paths(
        dir_path=dir_path,
        csv_path=csv_path,
        word_freq_path=word_freq_path,
):
    if not os.path.exists(csv_path):
        url = "https://raw.githubusercontent.com/jai-dewani/Word-completion/master/unigram_freq.csv"
        sp.run(f"wget -O {csv_path} {url}".split())

    if not os.path.exists(word_freq_path):
        with open(csv_path) as fin, open(word_freq_path, "w") as fout:
            for i, line in enumerate(fin, start=1):
                if i > 1:  # skip the column names
                    word, count = line.split(',')
                    count = int(count)  # make sure count is an int
                    print(json.dumps((word, count)), file=fout)

    return dotdict(
        dir_path=dir_path,
        csv_path=csv_path,
        word_freq_path=word_freq_path,
    )
