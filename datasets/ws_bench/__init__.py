import os
import subprocess as sp

from utils import dotdict

BENCHS = {
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
    query_lower_path = f"{datasets_dir}/{name}/queries.lower.txt"

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

    if not os.path.exists(query_path) or not os.path.exists(query_lower_path):
        def process(query_path, lower):
            words = set()
            with open(txt_path) as f:
                for line in f:
                    if lower:
                        line = line.lower()
                    w1, w2 = line.split()[:2]
                    words.add(w1)
                    words.add(w2)
            with open(query_path, "w") as fout:
                for w in words:
                    print(w, file=fout)

        process(query_path, lower=False)
        process(query_lower_path, lower=True)

    return dotdict(
        txt_path=txt_path,
        query_path=query_path,
        query_lower_path=query_lower_path
    )