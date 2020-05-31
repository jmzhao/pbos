import contextlib
import logging
import multiprocessing as mp
import os
import subprocess as sp
from collections import ChainMap

import pbos_train
import subwords
from datasets import prepare_combined_query_path, prepare_en_target_vector_paths
from datasets.unigram_freq import prepare_unigram_freq_paths
from datasets.ws_bench import prepare_bench_paths, BENCHS
from utils import dotdict
from utils.args import dump_args
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


def evaluate_ws_affix(args):
    with open(args.eval_result_path, "w") as fout:
        for bname in BENCHS:
            bench_path = prepare_bench_paths(bname).txt_path
            for lower in (True, False):
                print(eval_ws(args.pred_path, bench_path, lower=lower, oov_handling='zero'), file=fout)

        # sp.call(f"python affix_eval.py --embeddings {args.pred_path} --lower".split(), stdout=fout)


def exp(model_type, target_vector_name):
    target_vector_paths = prepare_en_target_vector_paths(target_vector_name)
    args = dotdict()

    # misc
    args.results_dir = f"results/ws_affix_trial/{target_vector_name}_{model_type}"
    args.model_type = model_type
    args.log_level = "INFO"

    # subword
    args.word_boundary = args.model_type in ('pbosn',)
    args.subword_min_count = None
    args.subword_uniq_factor = None  # TODO: investigate if we need to set this to 0.8
    if model_type == 'bos':
        args.subword_min_len = 3
        args.subword_max_len = 6
    elif model_type in ('pbos', 'pbosn'):
        args.subword_min_len = 1    # TODO: investigate if we need to set this to 3
        args.subword_max_len = None

    # subword vocab
    args.subword_vocab_max_size = None
    args.subword_vocab_word_freq = target_vector_paths.word_freq_path
    args.subword_vocab = f"{args.results_dir}/subword_vocab.jsonl"

    # subword prob
    args.subword_prob_take_root = False
    if model_type == 'bos':
        args.subword_prob = None
    elif model_type in ('pbos', 'pbosn'):
        args.subword_prob_min_prob = 0
        args.subword_prob_word_freq = prepare_unigram_freq_paths().word_freq_path
        args.subword_prob = f"{args.results_dir}/subword_prob.jsonl"

    # training
    args.target_vectors = target_vector_paths.pkl_emb_path
    args.model_path = f"{args.results_dir}/model.pkl"
    args.subword_weight_threshold = None
    args.normalize_semb = args.model_type in ('pbosn',)
    if model_types in ("pbos", ):
        args.subword_weight_normalize = False
    else:
        args.subword_weight_normalize = True
    args.random_seed = 42
    args.subword_prob_eps = 0.01
    args.epochs = 50
    if target_vector_name == "polyglot_clean":
        args.lr = 0.1
    else:
        args.lr = 1
    args.lr_decay = True

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
        evaluate_ws_affix(args)


if __name__ == '__main__':
    model_types = ('bos', 'pbos', )
    target_vector_names = ("polyglot_clean", "polyglot")  # "google",)  # "glove")

    for target_vector_name in target_vector_names:  # avoid race condition
        prepare_en_target_vector_paths(target_vector_name)

    with mp.Pool() as pool:
        results = [
            pool.apply_async(exp, (model_type, target_vector_name))
            for model_type in model_types
            for target_vector_name in target_vector_names
        ]

        for r in results:
            r.get()
