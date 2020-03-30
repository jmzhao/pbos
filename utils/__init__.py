from typing import List, Dict

from tqdm import tqdm


def get_substrings(s: str, min_len=1, max_len=None) -> List[str]:
    """
    :param s: string
    :return: a list of contiguous substrings
    """
    max_len = max_len or len(s)
    for j in range(min_len, len(s) + 1):
        for i in range(max(0, j - max_len), max(0, j - min_len + 1)):
            yield s[i:j]


def normalize_prob(subword_count: Dict[str, int], take_root=False) -> Dict[str, float]:
    """
    :param subword_count: dictionary {word: count}
    :return: normalized probability {word: probability}, the length of word is also normalized
    """
    total = sum(subword_count.values())

    if take_root:
        # CONCERN: is this formula right? Long words will get a much higher score since we are taking the n-th root
        return {k: (v / total) ** (1 / len(k)) for k, v in subword_count.items()}
    else:
        return {k: (v / total) for k, v in subword_count.items()}


def dummy_tqdm(x, *args, **kwargs):
    return x


def get_number_of_lines(fobj):
    pos = fobj.tell()
    nol = sum(1 for _ in fobj)
    fobj.seek(pos)
    return nol


def file_tqdm(fobj):
    return tqdm(fobj, total=get_number_of_lines(fobj))
