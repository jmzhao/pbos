import argparse
import contextlib
import logging
import multiprocessing as mp
import os
import subprocess as sp
from collections import ChainMap

import pbos_train
import subwords
from datasets.google import prepare_google_paths
from datasets.polyglot_emb import prepare_polyglot_emb_paths
from datasets.unigram_freq import prepare_unigram_freq_paths
from datasets.ws_bench import prepare_bench_paths, BENCHS
from datasets import prepare_combined_query_path
from utils import dotdict
from utils.args import add_logging_args, dump_args
from ws_eval import eval_ws


def train(args):
    subwords.build_subword_vocab_cli(dotdict(ChainMap(
        dict(word_freq=args.subword_vocab_word_freq, output=args.subword_vocab), args,
    )))

    if args.subword_prob:
        subwords.build_subword_prob_cli(dotdict(ChainMap(
            dict(word_freq=args.subword_prob_word_freq, output=args.subword_prob), args,
        )))

    pbos_train.main(args)


def predict(args):
    sp.call(f"""
        python pbos_pred.py \
          --queries {args.query_path} \
          --save {args.pred_path} \
          --model {args.model_path} \
          {'--' if args.word_boundary else '--no_'}word_boundary \
    """.split())


def evaluate(args):
    result_file = open(args.eval_result_path, "w")

    for bname in BENCHS:
        bench_paths = prepare_bench_paths(bname)
        for lower in (True, False):
            print(eval_ws(args.pred_path, bench_paths.txt_path, lower=lower, oov_handling='zero'), file=result_file)

    sp.call(f"python affix_eval.py --embeddings {args.pred_path}".split(), stdout=result_file)


def get_target_vector_paths(target_vector_name):
    if target_vector_name.lower() == "google":
        return prepare_google_paths()
    if target_vector_name.lower() == "polyglot":
        return prepare_polyglot_emb_paths("en")
    raise NotImplementedError


def exp(model_type, target_vector_name):
    target_vector_paths = get_target_vector_paths(target_vector_name)
    args = dotdict()

    # misc
    args.results_dir = f"results/best_ws_affix/{target_vector_name}_{model_type}"
    args.model_type = model_type
    args.log_level = "INFO"

    # subword
    args.word_boundary = False
    args.subword_min_count = None
    args.subword_uniq_factor = None  # or shall we ?
    if model_type == 'bos':
        args.subword_min_len = 3
        args.subword_max_len = 6
    elif model_type == 'pbos':
        args.subword_min_len = 1
        args.subword_max_len = None

    # subword vocab
    args.subword_vocab_max_size = None
    args.subword_vocab_word_freq = target_vector_paths.word_freq_path
    args.subword_vocab = f"{args.results_dir}/subword_vocab.jsonl"

    # subword prob
    if model_type == 'bos':
        args.subword_prob = None
    elif model_type == 'pbos':
        args.subword_prob_take_root = False
        args.subword_prob_min_prob = 0
        args.subword_prob_word_freq = prepare_unigram_freq_paths().word_freq_path
        args.subword_prob = f"{args.results_dir}/subword_prob.jsonl"

    # training
    args.target_vectors = target_vector_paths.txt_emb_path
    args.model_path = f"{args.results_dir}/model.pkl"
    args.epochs = 50
    args.lr = 1.0
    args.lr_decay = True
    args.random_seed = 42
    args.subword_prob_eps = 0.01
    args.subword_weight_threshold = None

    # prediction & evaluation
    args.pred_path = f"{args.results_dir}/vectors.txt"
    args.query_path = prepare_combined_query_path()
    args.eval_result_path = f"{args.results_dir}/result.txt"
    os.makedirs(args.results_dir, exist_ok=True)

    # redirect log output
    log_file = open(f"{args.results_dir}/log.txt", "w+")
    logging.basicConfig(level=logging.INFO, stream=log_file)
    dump_args(args)

    with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
        train(args)
        predict(args)
        evaluate(args)


with mp.Pool() as pool:
    model_types = ('pbos', 'bos')
    target_vector_names = ("polyglot", "google")

    results = [
        pool.apply_async(exp, (model_type, target_vector_name))
        for model_type in model_types
        for target_vector_name in target_vector_names
    ]

    for r in results:
        r.get()
