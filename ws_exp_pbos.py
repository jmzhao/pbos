import contextlib
import os
import subprocess as sp
import multiprocessing as mp
import argparse
from collections import ChainMap
from datasets.google import prepare_google_paths
from datasets.polyglot_emb import prepare_polyglot_emb_paths
from datasets.unigram_freq import prepare_unigram_freq_paths
import pbos_train
import subwords
from datasets.ws_bench import prepare_bench_paths, BENCHS, prepare_combined_query_path
from utils import dotdict
from utils.args import add_logging_args, set_logging_config, dump_args
from ws_eval import eval_ws


def train(args):
    subwords.build_subword_vocab_cli(dotdict(ChainMap(
        dict(
            command='build_vocab',
            output=args.subword_vocab
        ), args,
    )))

    if args.subword_prob:
        subwords.build_subword_prob_cli(dotdict(ChainMap(
            dict(
                command='build_prob',
                word_freq=prepare_unigram_freq_paths().word_freq_path,
                output=args.subword_prob
            ), args,
        )))

    pbos_train.main(args)


def evaluate(args):
    query_path = prepare_combined_query_path()
    pred_path = f"{args.results_dir}/vectors.txt"
    sp.call(f"""
        python pbos_pred.py \
          --queries {query_path} \
          --save {pred_path} \
          --model {args.model_path} \
          {'--' if args.word_boundary else '--no_'}word_boundary \
    """.split())

    for bname in BENCHS:
        bench_paths = prepare_bench_paths(bname)
        for lower in (True, False):
            print(eval_ws(pred_path, bench_paths.txt_path, lower=lower, oov_handling='zero'))


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
    if target_vector_name.lower() == "google_news":
        return prepare_google_paths()
    elif target_vector_name.lower() == "polyglot":
        return prepare_polyglot_emb_paths("en")
    raise NotImplementedError


def exp(model_type, target_vector_name, subword_prob_min_prob, word_boundary):
    args = get_default_args()

    args.results_dir = f"results/ws_{target_vector_name}_{model_type}"

    # additional args
    args.results_dir += f"_minprob{subword_prob_min_prob}_wb{'T' if word_boundary else 'F'}"
    args.subword_prob_min_prob = subword_prob_min_prob
    args.word_boundary = word_boundary

    target_vector_paths = get_target_vector_paths(target_vector_name)
    args.target_vectors = target_vector_paths.txt_emb_path
    args.model_type = model_type
    args.word_freq = target_vector_paths.word_freq_path  # will get overridden for prob
    args.subword_vocab = f"{args.results_dir}/subword_vocab.jsonl"
    args.subword_prob = f"{args.results_dir}/subword_prob.jsonl" if args.model_type == 'pbos' else None
    args.epochs = 50
    args.model_path = f"{args.results_dir}/model.pkl"

    set_logging_config(args)
    dump_args(args)

    os.makedirs(args.results_dir, exist_ok=True)
    with contextlib.redirect_stdout(open(f"{args.results_dir}/train.out", 'w+')), \
         contextlib.redirect_stderr(open(f"{args.results_dir}/train.err", 'w+')):
        train(args)
    with contextlib.redirect_stdout(open(f"{args.results_dir}/train.out", 'w+')), \
         contextlib.redirect_stderr(open(f"{args.results_dir}/train.err", 'w+')):
        evaluate(args)


with mp.Pool() as pool:
    model_types = ('pbos',)
    target_vector_names = ("polyglot",)
    subword_prob_min_probs = (0, 1e-6)
    wbs = (True, False)

    results = [
        pool.apply_async(exp, (model_type, target_vector_name, subword_prob_min_prob, wb))
        for model_type in model_types
        for target_vector_name in target_vector_names
        for subword_prob_min_prob in subword_prob_min_probs
        for wb in wbs
    ]

    for r in results:
        r.get()
