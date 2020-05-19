import argparse
from importlib import import_module
from itertools import islice
import json
import math

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

parser = argparse.ArgumentParser("PboS segmenter and subword weigher.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--prob_word_freq', default="unigram_freq",
                    choices=["unigram_freq"],
                    help="list of words to create subword prob")
parser.add_argument('--vocab_word_freq',
                    choices=["google", "polyglot"],
                    help="list of words to create subword vocab")
parser.add_argument('--n_largest', '-n', type=int, default=20,
                    help="the number of segmentations to show")
parser.add_argument('--subword_prob_eps', '-spe', type=float, default=1e-2,
                    help="the infinitesimal prob for unseen subwords")
parser.add_argument('--subword_weight_threshold', '-swt', type=float,
                    help="the minimum weight of a subword to be considered")
parser.add_argument('--interactive', '-i', action='store_true',
                    help="interactive mode")
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
def revised_normalize(subword_counter):
    with open(word_freq_path) as fin:
        word_count_iter = (json.loads(line) for line in fin)
        total_word_count = sum(count for word, count in word_count_iter)
        return {k : v / total_word_count for k, v in subword_counter.items()}
subword_prob = build_subword_prob(
    subword_counter,
    # normalize_prob=normalize_prob,
    normalize_prob=revised_normalize, ## trial
    min_prob=args.subword_prob_min_prob,
)
logger.info(f"subword prob size: {len(subword_prob)}")

logger.info(f"building subword vocab from `{args.vocab_word_freq}`...")
if args.vocab_word_freq is None:
    subword_vocab = set(subword_prob)
else:
    if args.vocab_word_freq.lower().startswith("google"):
        word_freq_path = import_module("datasets.google")\
            .prepare_google_paths().word_freq_path
    elif args.vocab_word_freq.lower().startswith("polyglot"):
        word_freq_path = import_module("datasets.polyglot_emb")\
            .prepare_polyglot_emb_paths('en').word_freq_path
    else:
        raise ValueError(f"args.vocab_word_freq=`{args.vocab_word_freq}` not supported.")
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
    "neurology",
    "farmland",
    "somnambulists", ## aka "sleep-walkers"
    "penpineappleapplepie",
    "lowest",
    "boring",
    "technical",
    "electronic",
    "synchronic",
    "synchronized",
]

get_subword_prob=partial(
    get_subword_prob,
    subword_prob=subword_prob,
    take_root=args.subword_prob_take_root,
    eps=args.subword_prob_eps,
)

def test_word(w):
    if args.word_boundary:
        w = '<' + w + '>'

    p_prefix = calc_prefix_prob(w, get_subword_prob)
    logging.info("p_prefix: " + '\t'.join(f"{x:.5e}" for x in p_prefix))
    p_suffix = calc_prefix_prob(w[::-1], get_subword_prob)[::-1]
    logging.info("p_suffix: " + '\t'.join(f"{x:.5e}" for x in p_suffix))

    print("top segmentations:")
    adjmat = [[None for __ in range(len(w) + 1)] for _ in range(len(w) + 1)]
    for i in range(len(w)):
        for j in range(i + 1, len(w) + 1):
            adjmat[i][j] = - math.log(max(1e-100, get_subword_prob(w[i:j])))
    segs = nshortest(adjmat, args.n_largest)
    for score, seg in segs:
        print("{:.5e} : {}".format(math.exp(-score), '/'.join(w[i:j] for i, j in zip(seg, seg[1:]))))

    print("top subword weights:")
    subword_weights = calc_subword_weights(
        w,
        subword_vocab=subword_vocab,
        get_subword_prob=get_subword_prob,
        weight_threshold=args.subword_weight_threshold,
    )
    for sub, weight in islice(sorted(subword_weights.items(), key=lambda t: t[1], reverse=True), args.n_largest):
        print("{:.5e} : {}".format(weight, sub))

for w in test_words:
    print("Word:", w)
    test_word(w)

if args.interactive:
    while True:
        w = input().strip()
        if not w:
            break
        test_word(w)
