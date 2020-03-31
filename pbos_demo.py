#!/usr/bin/python3
import argparse
from collections import ChainMap
import json
import logging
import os
import subprocess as sp
import sys

from datasets.google_news import prepare_google_news_paths
from datasets.unigram_freq import prepare_unigram_freq_paths
import pbos_train
import subwords
from utils import dotdict
from utils.args import add_logging_args, logging_config


# default arguments, if not otherwise overwritten by command line arguments.
demo_config = dict(
    subword_min_count = 1,
    subword_prob_min_prob = 1e-6,
    word_boundary = True,
)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--target_vectors', default='google_news',
    choices=['google_news'],
    help="target word vectors")
parser.add_argument('--model_path', default="./results/pbos/demo/model.pbos",
    help="The path to the model to be evaluated. "
    "If the model is not there, a new model will be trained and saved.")
parser.add_argument('--model_type', default="pbos",
    choices=["bos", "pbos"],
    help="model type")
add_logging_args(parser)
pbos_train.add_training_args(parser)
pbos_train.add_model_args(parser)
subwords.add_subword_args(parser)
subwords.add_subword_prob_args(parser)
subwords.add_subword_vocab_args(parser)
parser_action_lookup = {
    action.dest : action
    for action in parser._actions
    # the following test is needed, otherwise `--no_<flag>` option will
    # overwrite `--<flag>` option.
    if not any(s.startswith('--no_') for s in action.option_strings)
}
parser_action_lookup["subword_vocab"].required = False
for dest, value in demo_config.items():
    parser_action_lookup[dest].default = value
args = dotdict(vars(parser.parse_args()))


logging_config(args)
logging.info(json.dumps(args, indent=2))

datasets_dir = "./datasets"
results_dir, _ = os.path.split(args.model_path)

os.makedirs(results_dir, exist_ok=True)
os.makedirs(datasets_dir, exist_ok=True)


"""
python subwords.py build_vocab --word_freq datasets/google_news/word_freq.jsonl --output datasets/google_news/subword_vocab.jsonl -wb --subword_min_count 5;
python subwords.py build_prob --word_freq datasets/google_news/word_freq.jsonl --output datasets/google_news/subword_prob.jsonl -wb --subword_prob_min_prob 1e-6 --subword_prob_take_root;

RDIR=results/trials/pbos/unigram_freq/no_take_root; mkdir -p ${RDIR};
python pbos_train.py --target_vectors datasets/google_news/embedding.txt --model_path ${RDIR}/model.pbos --subword_vocab datasets/google_news/subword_vocab.jsonl --subword_prob datasets/unigram_freq/subword_prob.jsonl 2> >(tee -a ${RDIR}/train.log);

RDIR=results/trials/pbos/unigram_freq/take_root; mkdir -p ${RDIR};
python pbos_train.py --target_vectors datasets/google_news/embedding.txt --model_path ${RDIR}/model.pbos --subword_vocab datasets/google_news/subword_vocab.jsonl --subword_prob datasets/unigram_freq/subword_prob.jsonl --subword_prob_take_root 2> >(tee -a ${RDIR}/train.log);
"""

if not os.path.exists(args.model_path):
    # model does not exist, need to train a model.
    if args.target_vectors.lower() == "google_news": # default, use google vectors
        google_news_paths = prepare_google_news_paths()
        args.target_vectors = google_news_paths.txt_emb_path


        subword_vocab_path = os.path.join(google_news_paths.dir_path, "subword_vocab.jsonl")
        if not os.path.exists(subword_vocab_path):
            subword_vocab_args = dotdict(ChainMap(
                dict(
                    command = "build_vocab",
                    word_freq = google_news_paths.word_freq_path,
                    output = subword_vocab_path,
                ),
                args,
            ))
            subwords.build_subword_vocab_cli(subword_vocab_args)
        args.subword_vocab = subword_vocab_path
    else:
        raise NotImplementedError

    if args.model_type.lower() == 'pbos':
        unigram_freq_paths = prepare_unigram_freq_paths()

        subword_prob_path = os.path.join(unigram_freq_paths.dir_path, "subword_prob.jsonl")
        if not os.path.exists(subword_prob_path):
            subword_prob_args = dotdict(ChainMap(
                dict(
                    command = "build_prob",
                    word_freq = unigram_freq_paths.word_freq_path,
                    output = subword_prob_path,
                    subword_prob_take_root = False,
                ),
                args,
            ))
            subwords.build_subword_prob_cli(subword_prob_args)
        args.subword_prob = subword_prob_path
    elif args.model_type.lower() == 'bos':
        args.subword_prob = None
    else:
        raise NotImplementedError

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
    "card660": {
        "url": "https://pilehvar.github.io/card-660/dataset.tsv",
        "no_zip": True,
        "raw_txt_rel_path": "dataset.tsv",
    },
}

for bname, binfo in BENCHS.items():
    raw_txt_rel_path = binfo["raw_txt_rel_path"]
    raw_txt_path = f"{datasets_dir}/{bname}/{raw_txt_rel_path}"
    if not os.path.exists(raw_txt_path):
        sp.call(
            f"""
            wget -c {binfo['url']} -P {datasets_dir}/{bname}
        """.split()
        )
        if not binfo.get("no_zip"):
            sp.call(
                f"""
                unzip {datasets_dir}/{bname}/{bname}.zip -d {datasets_dir}/{bname}
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
          --model {args.model_path} \
          {'--' if args.word_boundary else '--no_'}word_boundary
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
          --model {args.model_path} \
          {'--' if args.word_boundary else '--no_'}word_boundary
    """.split()
    )
    sp.call(
        f"""
        python ./fastText/eval.py \
          --data {btxt_path} \
          --model {bpred_path}
    """.split()
    )
