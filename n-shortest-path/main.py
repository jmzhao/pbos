#!/usr/bin/env python3
import collections
import math
from itertools import combinations

from dag import DAG


def get_substrings(s):
    return [s[x:y] for x, y in combinations(range(len(s) + 1), r=2)]


def normalize_word_prob(part_count):
    total = sum(part_count.values())
    return {k: math.pow(v / total, 1 / (len(k))) for k, v in part_count.items()}


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
    part_prob = load_vocab("../datasets/20k.txt")
    # part_prob = load_vocab("../datasets/unigram_freq.csv")
    dag_segger = DAG(part_prob=part_prob, n_largest=5)
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
