import collections
from itertools import combinations
from typing import List, Dict


def get_substrings(s: str) -> List[str]:
    """
    :param s: string
    :return: a list of contiguous substrings
    """
    return [s[x:y] for x, y in combinations(range(len(s) + 1), r=2)]


def count_subwords(vocab: List[str]) -> Dict[str, int]:
    """
    :param vocab: a list of string
    :return: count all the substrings in the vocab
    """
    subword_count = collections.defaultdict(int)
    for word in vocab:
        for subword in get_substrings(word):
            subword_count[subword] += 1
    return subword_count


def normalize_prob(subword_count: Dict[str, int]) -> Dict[str, float]:
    """
    :param subword_count: dictionary {word: count}
    :return: normalized probability {word: probability}, the length of word is also normalized
    """
    total = sum(subword_count.values())

    # TODO: is this formula right? Long words will get a much higher score since we are taking the n-th root
    return {k: (v / total) ** (len(k) ** -1) for k, v in subword_count.items()}
