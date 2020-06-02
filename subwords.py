import argparse
import os
from collections import Counter
import json
import logging

from tqdm import tqdm

from utils import file_tqdm, get_substrings, normalize_prob
from utils.args import add_logging_args, set_logging_config, dump_args

logger = logging.getLogger(__name__)


def bound_word(word):
    return '<' + word + '>'


def build_subword_counter(
    word_count_iter,
    max_size=None,
    min_count=None,
    min_len=1,
    max_len=None,
    word_boundary=False,
    uniq_factor=None,
):
    subword_counter = Counter()
    for word, count in iter(word_count_iter):
        if word_boundary:
            word = bound_word(word)
        for subword in get_substrings(word, min_len=min_len, max_len=max_len):
            subword_counter[subword] += count

    if min_count:
        subword_counter = Counter({k : v for k, v in subword_counter.items() if v >= min_count})
    if uniq_factor is not None:
        for sub in tqdm(list(subword_counter)):
            for subsub in get_substrings(sub, min_len=min_len, max_len=max_len):
                if subsub != sub and subsub in subword_counter and subword_counter[subsub] * uniq_factor <= subword_counter[sub]:
                    del subword_counter[subsub]
    if max_size:
        subword_count_pairs = subword_counter.most_common(max_size)
    else:
        subword_count_pairs = subword_counter.items()
    return Counter(dict(subword_counter))



def subword_prob_post_process(subword_prob, min_prob=None, take_root=False):
    if min_prob:
        subword_prob = {k : v for k, v in subword_prob.items() if v >= min_prob}
    if take_root:
        subword_prob = {k : (v ** (1 / len(k))) for k, v in subword_prob.items()}
    return subword_prob

def build_subword_prob(
    subword_counter,
    normalize_prob=normalize_prob,
    min_prob=None,
    take_root=False,
):
    subword_prob = normalize_prob(subword_counter)
    subword_prob = subword_prob_post_process(
        subword_prob,
        min_prob=min_prob,
        take_root=take_root,
    )
    return Counter(subword_prob)


def parse_args():
    parser = argparse.ArgumentParser(description='Subword processing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    return parser.parse_args()


def add_args(parser):
    parser.add_argument('command', choices=['build_vocab', 'build_prob'])
    parser.add_argument('--word_freq', required=True,
        help='word frequencies (.jsonl). '
        'Each line is a pair of word and its count.')
    parser.add_argument('--output',  default='subword.jsonl',
        help='output file (.jsonl). '
        'Each line is a pair of word and count (build_vocab) '
        'or a pair of word and score (build_prob).')
    add_logging_args(parser)
    add_subword_args(parser)
    add_subword_vocab_args(parser)
    add_subword_prob_args(parser)
    return parser


def add_word_args(parser):
    group = parser.add_argument_group('word arguments')
    group.add_argument('--word_boundary', '-wb', action='store_true',
        help="annotate word boundary with '<' and '>'")
    group.add_argument('--no_word_boundary', '-Nwb',
        dest='word_boundary', action='store_false')
    return group


def add_subword_args(parser):
    add_word_args(parser)
    group = parser.add_argument_group('subword arguments')
    group.add_argument('--subword_min_count', type=int,
        help="subword min count for it to be included in vocab")
    group.add_argument('--subword_min_len', type=int, default=1,
        help="subword min length for it to be included in vocab")
    group.add_argument('--subword_max_len', type=int,
        help="subword max length for it to be included in vocab")
    group.add_argument('--subword_uniq_factor', '-suf', type=float,
        help="subword uniqueness factor")
    return group


def add_subword_vocab_args(parser):
    group = parser.add_argument_group('subword vocab arguments')
    group.add_argument('--subword_vocab_max_size', type=int,
        help="maximum size of subword vocab")
    return group


def add_subword_prob_args(parser):
    group = parser.add_argument_group('subword prob arguments')
    group.add_argument('--subword_prob_min_prob', '-spmp', type=float,
        help="minimum prob score of subword vocab")
    group.add_argument('--subword_prob_take_root', '-sptr', action='store_true',
        help="take `** (1 / len(subword))` for prob score")
    group.add_argument('--no_subword_prob_take_root', '-Nsptr',
        dest='subword_prob_take_root', action='store_false')
    return group


def build_subword_vocab_cli(args):
    if os.path.exists(args.output):
        logger.warning(f"{args.output} already exists!")

    logger.info("loading...")
    with open(args.word_freq) as fin:
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
    logger.info("processing...")
    subword_vocab = subword_counter
    logger.info("saving...")
    with open(args.output, 'w') as fout:
        for (subword, count) in tqdm(subword_vocab.most_common()):
            print(json.dumps((subword, count)), file=fout)


def build_subword_prob_cli(args):
    if os.path.exists(args.output):
        logger.warning(f"{args.output} already exists!")

    logger.info("loading...")
    with open(args.word_freq) as fin:
        word_count_iter = (json.loads(line) for line in file_tqdm(fin))
        subword_counter = build_subword_counter(
            word_count_iter,
            min_count=args.subword_min_count,
            min_len=args.subword_min_len,
            max_len=args.subword_max_len,
            word_boundary=args.word_boundary,
            uniq_factor=args.subword_uniq_factor,
        )
    logger.info("processing...")
    if args.subword_prob_take_root:
        logger.warning("`args.subword_prob_take_root = True` ignored at this step.")
    subword_prob = build_subword_prob(
        subword_counter,
        normalize_prob=normalize_prob,
        min_prob=args.subword_prob_min_prob,
        # take_root=args.subword_prob_take_root,
    )
    logger.info("saving...")
    with open(args.output, 'w') as fout:
        for (subword, prob) in tqdm(subword_prob.most_common()):
            print(json.dumps((subword, prob)), file=fout)


def main_cli(args):
    set_logging_config(args)
    if args.command == 'build_vocab':
        build_subword_vocab_cli(args)
    elif args.command == 'build_prob':
        build_subword_prob_cli(args)
    else:
        raise ValueError(f"Unknown command `{args.command}`")


if __name__ == '__main__':
    main_cli(parse_args())
