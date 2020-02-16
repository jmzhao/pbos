#!/usr/bin/env python3
import collections
from itertools import combinations
import logging

from dag import DAG


logging.basicConfig(level=logging.INFO)


def get_substrings(s):
    return [s[x:y] for x, y in combinations(range(len(s) + 1), r=2)]


def normalize_word_prob(part_count):
    total = sum(part_count.values())
    return {k: (v / total) ** (len(k) ** -1) for k, v in part_count.items()}


def load_vocab(filename):
    part_count = collections.defaultdict(int)
    with open(filename, "r") as f:
        for line in f:
            rows = line.strip().split(",")
            part = rows[0]
            count = int(rows[1]) if len(rows) == 2 else 1
            for part in get_substrings(part):
                part_count[part] += count

    return normalize_word_prob(part_count)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_list', '-f', default="../datasets/unigram_freq.csv",
        help="list of words to create subword vocab")
    parser.add_argument('--n_largest', '-n', type=int, default=5,
        help="the number of segmentations to show")
    parser.add_argument('--interactive', '-i', action='store_true',
        help="interactive mode")
    parser.add_argument('--test_file', '-t',
        help="list of words to be segmented")
    args = parser.parse_args()

    logging.info(f"building subword vocab from `{args.word_list}`...")
    part_prob = load_vocab(args.word_list)
    logging.info(f"subword vocab size: {len(part_prob)}")
    dag_segger = DAG(part_prob=part_prob, n_largest=args.n_largest)
    logging.info(f"Ready.")
    if args.interactive:
        while True:
            word = input().strip()
            if not word:
                break
            dag_segger.test([word])
    elif args.test_file:
        dag_segger.test((line.strip() for line in open(args.test_file)))
    else:
        dag_segger.test(
            [
                "ilikeeatingapples",
                "pineappleanapplepie",
                "bioinfomatics",
                "technical",
                "electrical",
                "electronic",
            ]
        )
