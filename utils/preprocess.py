import collections
from itertools import combinations
from typing import List, Dict


def get_substrings(s: str, min_len=1, max_len=None) -> List[str]:
    """
    :param s: string
    :return: a list of contiguous substrings
    """
    max_len = max_len or len(s)
    for j in range(min_len, len(s)):
        for i in range(max(0, j - max_len), max(0, j - min_len + 1)):
            yield s[i:j]


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


def normalize_prob(subword_count: Dict[str, int], take_root=False) -> Dict[str, float]:
    """
    :param subword_count: dictionary {word: count}
    :return: normalized probability {word: probability}, the length of word is also normalized
    """
    total = sum(subword_count.values())

    if take_root:
        # TODO: is this formula right? Long words will get a much higher score since we are taking the n-th root
        return {k: (v / total) ** (len(k) ** -1) for k, v in subword_count.items()}
    else:
        return {k: (v / total) for k, v in subword_count.items()}
