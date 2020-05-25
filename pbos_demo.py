#!/usr/bin/python3
import argparse
from collections import ChainMap
import os
import subprocess as sp

# default arguments, if not otherwise overwritten by command line arguments.
from configs import demo_config
from datasets.google import prepare_google_paths
from datasets.polyglot_emb import prepare_polyglot_emb_paths
from datasets.unigram_freq import prepare_unigram_freq_paths
import pbos_train
import subwords
from datasets.ws_bench import prepare_bench_paths, BENCHS
from datasets import prepare_combined_query_path
from utils import dotdict
from utils.args import add_logging_args, set_logging_config, dump_args
from ws_eval import eval_ws

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--target_vectors', default='google_news',
    choices=['google_news', 'polyglot'],
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


set_logging_config(args)
dump_args(args)

results_dir, _ = os.path.split(args.model_path)
os.makedirs(results_dir, exist_ok=True)


"""
python subwords.py build_vocab --word_freq datasets/google_news/word_freq.jsonl --output datasets/google_news/subword_vocab.jsonl -wb --subword_min_count 5;
python subwords.py build_prob --word_freq datasets/google_news/word_freq.jsonl --output datasets/google_news/subword_prob.jsonl -wb --subword_prob_min_prob 1e-6 --subword_prob_take_root;

RDIR=results/trials/pbos/unigram_freq/no_take_root; mkdir -p ${RDIR};
python pbos_train.py --target_vectors datasets/google_news/embedding.txt --model_path ${RDIR}/model.pbos --subword_vocab datasets/google_news/subword_vocab.jsonl --subword_prob datasets/unigram_freq/subword_prob.jsonl 2> >(tee -a ${RDIR}/train.log);

RDIR=results/trials/pbos/unigram_freq/take_root; mkdir -p ${RDIR};
python pbos_train.py --target_vectors datasets/google_news/embedding.txt --model_path ${RDIR}/model.pbos --subword_vocab datasets/google_news/subword_vocab.jsonl --subword_prob datasets/unigram_freq/subword_prob.jsonl --subword_prob_take_root 2> >(tee -a ${RDIR}/train.log);
"""


if not os.path.exists(args.model_path):  # model does not exist, need to train a model.
    if args.target_vectors.lower() == "google_news":
        target_vector_paths = prepare_google_paths()
    elif args.target_vectors.lower() == "polyglot":
        target_vector_paths = prepare_polyglot_emb_paths("en")
    else:
        raise NotImplementedError

    args.target_vectors = target_vector_paths.txt_emb_path
    args.subword_vocab = subwords.build_subword_vocab_cli(dotdict(ChainMap(
        dict(
            command='build_vocab',
            word_freq=target_vector_paths.word_freq_path
        ), args,
    )))

    if args.model_type.lower() == 'pbos':
        unigram_freq_path = prepare_unigram_freq_paths().word_freq_path
        args.subword_prob = subwords.build_subword_prob_cli(dotdict(ChainMap(
            dict(
                command='build_prob',
                word_freq=unigram_freq_path
            ), args,
        )))
    elif args.model_type.lower() == 'bos':
        args.subword_prob = None
    else:
        raise NotImplementedError

    pbos_train.main(args)


bquery_path = prepare_combined_query_path()
bpred_path = f"{results_dir}/vectors.txt"
## collect and predict all query words to save model load time.
sp.call(f"""
    python pbos_pred.py \
      --queries {bquery_path} \
      --save {bpred_path} \
      --model {args.model_path} \
      {'--' if args.word_boundary else '--no_'}word_boundary \
""".split()) ## use `word_boundary` consistent with training
      # {'--' if args.word_boundary else '--no_'}word_boundary \ ## use `word_boundary` consistent with training
      # --no_word_boundary \ ## always use `--no_word_boundary` when pred

for bname in BENCHS:
    bench_paths = prepare_bench_paths(bname)
    for lower in (True, False):
        result = eval_ws(bpred_path, bench_paths.txt_path, lower=lower, oov_handling='zero')
        with open(f"{results_dir}/ws_result.txt", "a+") as fout:
            print(result, file=fout)
