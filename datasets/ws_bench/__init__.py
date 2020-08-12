import os
import subprocess as sp

from utils import dotdict

BENCHS = {
    "simlex999-german": {
        "url": "https://raw.githubusercontent.com/nmrksic/eval-multilingual-simlex/master/evaluation/simlex-german.txt",
        "raw_txt_rel_path": "simlex-german.txt",
        "no_zip": True,
        "skip_lines": 1,
    },
    "simlex999-italian": {
        "url": "https://raw.githubusercontent.com/nmrksic/eval-multilingual-simlex/master/evaluation/simlex-italian.txt",
        "raw_txt_rel_path": "simlex-italian.txt",
        "no_zip": True,
        "skip_lines": 1,
    },
    "simlex999-russian": {
        "url": "https://raw.githubusercontent.com/nmrksic/eval-multilingual-simlex/master/evaluation/simlex-russian.txt",
        "raw_txt_rel_path": "simlex-russian.txt",
        "no_zip": True,
        "skip_lines": 1,
    },
    "wordsim353": {
        "url": "https://leviants.com/wp-content/uploads/2020/01/wordsim353.zip",
        "raw_txt_rel_path": "combined.tab",
        "skip_lines": 1,
    },
    "rw": {
        "url": "https://nlp.stanford.edu/~lmthang/morphoNLM/rw.zip",
        "raw_txt_rel_path": "rw/rw.txt",
    },
    "card660": {
        "url": "https://pilehvar.github.io/card-660/dataset.tsv",
        "no_zip": True,
        "raw_txt_rel_path": "dataset.tsv",
    },
}

datasets_dir = os.path.dirname(os.path.realpath(__file__))


def prepare_bench_paths(name):
    binfo = BENCHS[name]

    raw_txt_path = f"{datasets_dir}/{name}/{binfo['raw_txt_rel_path']}"
    txt_path = f"{datasets_dir}/{name}/{name}.txt"
    query_path = f"{datasets_dir}/{name}/queries.txt"

    if not os.path.exists(raw_txt_path):
        sp.call(
            f"""
            wget -c {binfo['url']} -P {datasets_dir}/{name}
        """.split()
        )
        if not binfo.get("no_zip"):
            sp.call(
                f"""
                unzip {datasets_dir}/{name}/{name}.zip -d {datasets_dir}/{name}
            """.split()
            )

    if not os.path.exists(txt_path):
        with open(raw_txt_path) as f, open(txt_path, "w") as fout:
            for i, line in enumerate(f):
                # discard head lines
                if i < binfo.get("skip_lines", 0):
                    continue
                # NOTE: in `fastText/eval.py`, golden words get lowercased anyways,
                # but predicted words remain as they are.
                print(line, end="", file=fout)

    if not os.path.exists(query_path):
        words = set()
        with open(txt_path) as f:
            for line in f:
                w1, w2 = line.split()[:2]
                words.add(w1)
                words.add(w2)
        with open(query_path, "w") as fout:
            for w in words:
                print(w, file=fout)

    return dotdict(
        txt_path=txt_path,
        query_path=query_path,
    )


if __name__ == '__main__':
    for bname in BENCHS:
        prepare_bench_paths(bname)
