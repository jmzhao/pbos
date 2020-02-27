#!/usr/bin/python3
import argparse
import logging
import os
import subprocess as sp
import sys

from datasets.google_news import get_google_news_paths

import pbos_train

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
pbos_train.add_args(parser)
for action in parser._actions:
    if action.dest == "target_vectors":
        action.required = False
    elif action.dest == "model_path":
        action.required = False
        action.default = "./results/pbos/demo/model.pbos"
        action.help = "The path to the model to be evaluated. "
        "If the model is not there, a new model will be trained and saved."
args = parser.parse_args()


numeric_level = getattr(logging, args.loglevel.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError("Invalid log level: %s" % args.loglevel)
logging.basicConfig(level=numeric_level)
logging.info(args)

datasets_dir = "./datasets"
results_dir, _ = os.path.split(args.model_path)

os.makedirs(results_dir, exist_ok=True)
os.makedirs(datasets_dir, exist_ok=True)

pretrained_processed_path, wordlist_path = get_google_news_paths()

if not os.path.exists(args.model_path):
    args.target_vectors = pretrained_processed_path
    args.word_list = wordlist_path
    args.word_list_has_freq = False
    pbos_train.main(args)

BENCHS = {
    "rw": {
        "url": "https://nlp.stanford.edu/~lmthang/morphoNLM/rw.zip",
        "raw_txt_rel_path": "rw/rw.txt",
    },
    "wordsim353": {
        "url": "https://leviants.com/wp-content/uploads/2020/01/wordsim353.zip",
        "raw_txt_rel_path": "combined.tab",
        "skip_lines": 1,
    },
    "card—660": {"raw_txt_rel_path": "rare_word/card-660.txt"},
}

for bname, binfo in BENCHS.items():
    raw_txt_rel_path = binfo["raw_txt_rel_path"]
    raw_txt_path = f"{datasets_dir}/{bname}/{raw_txt_rel_path}"
    if not os.path.exists(raw_txt_path):
        sp.call(
            f"""
            wget -c {binfo['url']} -P {datasets_dir}
        """.split()
        )
        sp.call(
            f"""
            unzip {datasets_dir}/{bname}.zip -d {datasets_dir}/{bname}
        """.split()
        )
    btxt_path = f"{datasets_dir}/{bname}/{bname}.txt"
    if not os.path.exists(btxt_path):
        with open(raw_txt_path) as f, open(btxt_path, "w") as fout:
            for i, line in enumerate(f):
                ## discard head lines
                if i < binfo.get("skip_lines", 0):
                    continue
                ## NOTE: in `fastText/eval.py`, golden words get lowercased anyways,
                ## but predicted words remain as they are.
                print(line, end="", file=fout)
    bquery_path = f"{datasets_dir}/{bname}/queries.txt"
    bquery_lower_path = f"{datasets_dir}/{bname}/queries.lower.txt"
    if not os.path.exists(bquery_path) or not os.path.exists(bquery_lower_path):

        def process(query_path, lower):
            words = set()
            with open(btxt_path) as f:
                for line in f:
                    if lower:
                        line = line.lower()
                    w1, w2 = line.split()[:2]
                    words.add(w1)
                    words.add(w2)
            with open(query_path, "w") as fout:
                for w in words:
                    print(w, file=fout)

        process(bquery_path, lower=False)
        process(bquery_lower_path, lower=True)

    bpred_path = f"{results_dir}/{bname}_vectors.txt"
    ## eval on original benchmark
    sp.call(
        f"""
        python pbos_pred.py \
          --queries {bquery_path} \
          --save {bpred_path} \
          --model {args.model_path}
    """.split()
    )
    sp.call(
        f"""
        python ./fastText/eval.py \
          --data {btxt_path} \
          --model {bpred_path}
    """.split()
    )
    ## eval on lowercased benchmark
    sp.call(
        f"""
        python pbos_pred.py \
          --queries {bquery_lower_path} \
          --save {bpred_path} \
          --model {args.model_path}
    """.split()
    )
    sp.call(
        f"""
        python ./fastText/eval.py \
          --data {btxt_path} \
          --model {bpred_path}
    """.split()
    )
