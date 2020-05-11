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
from datasets.ws_bench import prepare_bench_paths, BENCHS, prepare_combined_query_path
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


def evaluate(args):
    sp.call(f"""
        python pbos_pred.py \
          --queries {args.query_path} \
          --save {args.pred_path} \
          --model {args.model_path} \
          {'--' if args.word_boundary else '--no_'}word_boundary \
    """.split())

    for bname in BENCHS:
        bench_paths = prepare_bench_paths(bname)
        for lower in (True, False):
            print(eval_ws(args.pred_path, bench_paths.txt_path, lower=lower, oov_handling='zero'))


def get_default_args():
    parser = argparse.ArgumentParser()
    add_logging_args(parser)
    pbos_train.add_training_args(parser)
    pbos_train.add_model_args(parser)
    subwords.add_subword_args(parser)
    subwords.add_subword_prob_args(parser)
    subwords.add_subword_vocab_args(parser)

    parser_action_lookup = {
        action.dest: action
        for action in parser._actions
        # the following test is needed, otherwise `--no_<flag>` option will
        # overwrite `--<flag>` option.
        if not any(s.startswith('--no_') for s in action.option_strings)
    }
    parser_action_lookup["subword_vocab"].required = False

    return dotdict(vars(parser.parse_args()))


def get_target_vector_paths(target_vector_name):
    if target_vector_name.lower() == "google":
        return prepare_google_paths()
    if target_vector_name.lower() == "polyglot":
        return prepare_polyglot_emb_paths("en")
    raise NotImplementedError


def exp(model_type, target_vector_name):
    target_vector_paths = get_target_vector_paths(target_vector_name)
    args = get_default_args()

    # setup parameters
    args.model_type = model_type
    args.epochs = 50
    if model_type == 'bos':
        args.subword_min_len = 3
        args.subword_max_len = 6

    # setup paths
    args.results_dir = f"results/ws_{target_vector_name}_{model_type}"
    args.target_vectors = target_vector_paths.txt_emb_path
    args.subword_vocab_word_freq = target_vector_paths.word_freq_path
    args.subword_prob_word_freq = prepare_unigram_freq_paths().word_freq_path
    args.subword_vocab = f"{args.results_dir}/subword_vocab.jsonl"
    args.subword_prob = f"{args.results_dir}/subword_prob.jsonl" if args.model_type == 'pbos' else None
    args.model_path = f"{args.results_dir}/model.pkl"
    args.pred_path = f"{args.results_dir}/vectors.txt"
    args.query_path = prepare_combined_query_path()
    os.makedirs(args.results_dir, exist_ok=True)

    # redirect log output
    log_file = open(f"{args.results_dir}/log.txt", "w+")
    logging.basicConfig(level=logging.INFO, stream=log_file)
    dump_args(args)

    with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
        train(args)

    eval_file = open(f"{args.results_dir}/result.txt", "w+")
    with contextlib.redirect_stdout(eval_file), contextlib.redirect_stderr(eval_file):
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
