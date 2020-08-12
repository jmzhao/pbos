import argparse
import json
import math
from importlib import import_module
from itertools import islice

from datasets import prepare_target_vector_paths, target_vector_names
from nshortest import nshortest
from pbos import *
from subwords import (
    add_subword_args,
    add_subword_prob_args,
    add_subword_vocab_args,
    build_subword_counter,
    build_subword_prob,
)
from utils import file_tqdm, normalize_prob
from utils.args import add_logging_args, set_logging_config, dump_args

parser = argparse.ArgumentParser("PboS segmenter and subword weigher.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--prob_word_freq', default="unigram_freq",
                    choices=["unigram_freq"],
                    help="list of words to create subword prob")
parser.add_argument('--vocab_word_freq',
                    choices=target_vector_names,
                    help="list of words to create subword vocab")
parser.add_argument('--n_largest', '-n', type=int, default=20,
                    help="the number of segmentations to show")
parser.add_argument('--subword_prob_eps', '-spe', type=float, default=1e-2,
                    help="the infinitesimal prob for unseen subwords")
parser.add_argument('--subword_weight_threshold', '-swt', type=float,
                    help="the minimum weight of a subword to be considered")
parser.add_argument('--interactive', '-i', action='store_true',
                    help="interactive mode")
parser.add_argument('--latex', action='store_true',
                    help="output latex")
add_subword_args(parser)
add_subword_prob_args(parser)
add_subword_vocab_args(parser)
add_logging_args(parser)
args = parser.parse_args()

set_logging_config(args)
dump_args(args)

logger.info(f"building subword prob from `{args.prob_word_freq}`...")
if args.prob_word_freq.lower().startswith("unigram_freq"):
    word_freq_path = import_module("datasets.unigram_freq")\
        .prepare_unigram_freq_paths().word_freq_path
else:
    raise ValueError(f"args.prob_word_freq=`{args.prob_word_freq}` not supported.")
with open(word_freq_path) as fin:
    word_count_iter = (json.loads(line) for line in file_tqdm(fin))
    subword_counter = build_subword_counter(
        word_count_iter,
        min_count=args.subword_min_count,
        min_len=args.subword_min_len,
        max_len=args.subword_max_len,
        word_boundary=args.word_boundary,
        uniq_factor=args.subword_uniq_factor,
    )
subword_prob = build_subword_prob(
    subword_counter,
    normalize_prob=normalize_prob,
    min_prob=args.subword_prob_min_prob,
    take_root=args.subword_prob_take_root,
)
logger.info(f"subword prob size: {len(subword_prob)}")

logger.info(f"building subword vocab from `{args.vocab_word_freq}`...")
if args.vocab_word_freq is None:
    subword_vocab = set(subword_prob)
else:
    word_freq_path = prepare_target_vector_paths(args.vocab_word_freq).word_freq_path
    with open(word_freq_path) as fin:
        word_count_iter = (json.loads(line) for line in file_tqdm(fin))
        subword_counter = build_subword_counter(
            word_count_iter,
            max_size=args.subword_vocab_max_size,
            min_count=args.subword_min_count,
            min_len=args.subword_min_len,
            max_len=args.subword_max_len,
            word_boundary=args.word_boundary,
            uniq_factor=args.subword_uniq_factor,
        )
    subword_vocab = set(subword_counter)
subword_vocab -= set('<>')
logger.info(f"subword vocab size: {len(subword_vocab)}")



test_words = [
    "farmland",
    "higher",
    "penpineapplepie",
    "paradichlorobenzene",
    "bisimulation",
]

get_subword_prob=partial(
    get_subword_prob,
    subword_prob=subword_prob,
    take_root=args.subword_prob_take_root,
    eps=args.subword_prob_eps,
)


def word_segs(w):
    if args.word_boundary:
        w = '<' + w + '>'

    p_prefix = calc_prefix_prob(w, get_subword_prob)
    p_suffix = calc_prefix_prob(w, get_subword_prob, backward=True)

    adjmat = [[None for __ in range(len(w) + 1)] for _ in range(len(w) + 1)]
    for i in range(len(w)):
        for j in range(i + 1, len(w) + 1):
            adjmat[i][j] = - math.log(max(1e-100, get_subword_prob(w[i:j])))
    segs = nshortest(adjmat, args.n_largest)

    seg_score_dict = {
        '/'.join(w[i:j] for i, j in zip(seg, seg[1:])): math.exp(-score) / p_prefix[-1]
        for score, seg in segs
    }

    subword_weights = calc_subword_weights(
        w,
        subword_vocab=subword_vocab,
        get_subword_prob=get_subword_prob,
        weight_threshold=args.subword_weight_threshold,
    )

    sub_weight_dict = {
        sub : weight
        for sub, weight in islice(sorted(subword_weights.items(), key=lambda t: t[1], reverse=True), args.n_largest)
    }

    return p_prefix, p_suffix, seg_score_dict, sub_weight_dict


def test_word(w):
    p_prefix, p_suffix, seg_score_dict, sub_weight_dict = word_segs(w)

    if args.latex:
        top_seg_str = ", ".join(f"{seg} ({score:.3f})" for seg, score in seg_score_dict.items())
        sub_weight_str = ", ".join(f"{sub} ({weight:.3f})" for sub, weight in sub_weight_dict.items())
        print(f"{w} \n& {top_seg_str} \n& {sub_weight_str} \n\\\\\n\n".translate(
            str.maketrans({
                "<": r"{\textless}",
                ">": r"{\textgreater}",
            })
          )
        )

    else:

        print("Word:", w)

        logging.info("p_prefix: " + '\t'.join(f"{x:.5e}" for x in p_prefix))
        logging.info("p_suffix: " + '\t'.join(f"{x:.5e}" for x in p_suffix))

        print("top segmentations:")
        for seg, score in seg_score_dict.items():
            print("{:.5e} : {}".format(score, seg))

        print("top subword weights:")
        for sub, weight in sub_weight_dict.items():
            print("{:.5e} : {}".format(weight, sub))


for w in test_words:
    test_word(w)

if args.interactive:
    while True:
        w = input().strip()
        if not w:
            break
        test_word(w)
