import argparse
from collections import Counter
import json
import logging

from tqdm import tqdm

from utils import file_tqdm, get_substrings


def build_subword_counter(
    word_count_iter,
    max_size=None,
    min_count=None,
    min_len=1,
    max_len=None,
    word_boundary=False,
):
    subword_counter = Counter()
    for word, count in iter(word_count_iter):
        if word_boundary:
            word = '<' + word + '>'
        for subword in get_substrings(word, min_len=min_len, max_len=max_len):
            subword_counter[subword] += count

    if max_size:
        subword_counter = subword_counter.most_common(max_size)
    else:
        subword_counter = subword_counter.items()
    if min_count:
        subword_counter = ((k, v) for k, v in subword_counter if v >= min_count)
    return Counter(subword_counter)


def build_subword_prob(subword_counter, min_prob=None, take_root=False):
    subword_prob = normalize_prob(subword_counter, take_root=take_root)
    if min_prob:
        subword_prob = {k : v for k, v in subword_prob if v >= min_prob}
    return subword_prob


def build_subword_vocab(subword_counter):
    return list(subword_counter)


def parse_args():
    parser = argparse.ArgumentParser(description='Subword processing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    return parser.parse_args()


def add_args(parser):
    parser.add_argument('command', choices=['build_vocab', 'build_prob'])
    parser.add_argument('--word_freq', required=True,
        help='word frequencies (.jsonl)')
    parser.add_argument('--output', default='subword.json',
        help='output file (.jsonl)')
    parser.add_argument('--loglevel', default='INFO',
        help='log level used by logging module')
    add_subword_args(parser)
    return parser


def add_subword_args(parser):
    group = parser.add_argument_group('subword arguments')
    group.add_argument('--word_boundary', '-wb', action='store_true',
        help="annotate word boundary with '<' and '>'")
    group.add_argument('--no_word_boundary',
        dest='word_boundary', action='store_false')
    group.add_argument('--subword_max_num', type=int,
        help="maximum size of subword vocab")
    group.add_argument('--subword_min_count', type=int,
        help="subword min count for it to be included in vocab")
    group.add_argument('--subword_min_len', type=int, default=1,
        help="subword min length for it to be included in vocab")
    group.add_argument('--subword_max_len', type=int,
        help="subword max length for it to be included in vocab")
    return parser


def add_subword_vocab_args(parser):
    group = parser.add_argument_group('subword vocab arguments')
    group.add_argument('--subword_vocab_max_size', type=int,
        help="maximum size of subword vocab")
    return parser


def add_subword_prob_args(parser):
    group = parser.add_argument_group('subword prob arguments')
    group.add_argument('--subword_prob_min_prob', type=float,
        help="minimum prob score of subword vocab")
    group.add_argument('--subword_prob_take_root', action='store_true',
        help="take `** (1 / len(subword))` for prob score")
    group.add_argument('--no_subword_prob_take_root',
        dest='subword_prob_take_root', action='store_false')
    return parser


def build_subword_vocab_cli(args):
    with open(args.word_freq) as fin:
        word_count_iter = (json.loads(line) for line in file_tqdm(fin))
        subword_counter = build_subword_counter(
            word_count_iter,
            max_size=args.subword_vocab_max_size,
            min_count=args.subword_min_count,
            min_len=args.subword_min_len,
            max_len=args.subword_max_len,
            word_boundary=args.word_boundary,
        )
    subword_vocab = build_subword_vocab(subword_counter)
    with open(args.output, 'w') as fout:
        for subword in tqdm(subword_vocab):
            print(json.dumps(subword), file=fout)


def build_subword_prob_cli(args):
    with open(args.word_freq) as fin:
        word_count_iter = (json.loads(line) for line in file_tqdm(fin))
        subword_counter = build_subword_counter(
            word_count_iter,
            min_count=args.subword_min_count,
            min_len=args.subword_min_len,
            max_len=args.subword_max_len,
            word_boundary=args.word_boundary,
        )
    subword_prob = build_subword_prob(
        subword_counter,
        min_prob=args.subword_prob_min_prob,
        take_root=args.subword_prob_take_root,
    )
    with open(args.output, 'w') as fout:
        for (subword, prob) in tqdm(subword_prob):
            print(json.dumps((subword, prob)), file=fout)


def main_cli(args):
    if args.command == 'build_vocab':
        build_subword_vocab_cli(args)
    elif args.command == 'build_prob':
        build_subword_prob_cli(args)
    else:
        raise ValueError(f"Unkown comman `{args.command}`")


if __name__ == '__main__':
    main_cli(parse_args())
