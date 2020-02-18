from collections import defaultdict
from functools import lru_cache, partial
import logging

import numpy as np

from bos import BoS
from utils.preprocess import normalize_prob


logging.basicConfig(level=logging.DEBUG)

EPS = 1e-6

def calc_prefix_prob(w, subword_prob, backward=False):
    w = w[::-1] if backward else w
    p = [1]
    for i in range(1, len(w) + 1):
        p.append(sum(p[j] * subword_prob.get(w[j:i][::-1] if backward else w[j:i], EPS) for j in range(i)))
    return p[::-1] if backward else p

# def calc_prefix_prob(w, subword_prob, backward=False):
#     w = w[::-1] if backward else w
#     dag = {}
#     e_total = [0] * (len(w) + 1)
#     for i in range(1, len(w) + 1):
#         for j in range(i):
#             sub = w[j:i][::-1] if backward else w[j:i]
#             if sub in subword_prob:
#                 # logging.debug(sub)
#                 raw_p = subword_prob[sub]
#                 dag[(j, i)] = raw_p
#                 e_total[j] += raw_p
#     # logging.debug(' '.join(f"{x:.5E}" for x in (e_total[::-1] if backward else e_total)))
#     p = [1]
#     for i in range(1, len(w) + 1):
#         p.append(sum(p[j] * dag.get((j, i), 0) / e_total[j] for j in range(i)))
#     return p[::-1] if backward else p

def calc_subword_weights(w, subword_prob, boundary=False):
    if boundary:
        w = '<' + w + '>'
    p_prefix = calc_prefix_prob(w, subword_prob)
    p_suffix = calc_prefix_prob(w, subword_prob, backward=True)
    subword_weights = {}
    for j in range(1, len(w) + 1):
        for i in range(j):
            sub = w[i:j]
            if sub in subword_prob:
                p_sub = subword_prob[sub] * p_prefix[i] * p_suffix[j]
                subword_weights.setdefault(sub, 0)
                subword_weights[sub] += p_sub
    try:
        return normalize_prob(subword_weights)
    except ZeroDivisionError:
        logging.warning(f"zero weights for '{w}'")
        return {}

def calc_subword_weights_bos(w, subword_prob, boundary=False):
    if boundary:
        w = '<' + w + '>'
    subword_weights = {}
    for j in range(1, len(w) + 1):
        for i in range(j):
            sub = w[i:j]
            if sub in subword_prob:
                subword_weights[sub] = 1
    try:
        return normalize_prob(subword_weights)
    except ZeroDivisionError:
        logging.warning(f"zero weights for '{w}'")
        return {}


class PBoS (BoS):
    def __init__(self, embedding_dim, * , subword_prob, boundary=False):
        self.semb = defaultdict(float)
        self.subword_prob = subword_prob
        self._calc_subword_weights = lru_cache(maxsize=32)(
            partial(calc_subword_weights_bos, subword_prob=subword_prob, boundary=boundary))
        self.config = dict(embedding_dim=embedding_dim, subword_prob=subword_prob, boundary=boundary)
        self._zero_emb = np.zeros(self.config['embedding_dim'])

    def embed(self, w):
        subword_weights = self._calc_subword_weights(w)
        wemb = sum(w * self.semb[sub] for sub, w in subword_weights.items())
        return wemb if isinstance(wemb, np.ndarray) else self._zero_emb

    def step(self, w, d):
        subword_weights = self._calc_subword_weights(w)
        for sub, w in subword_weights.items():
            self.semb[sub] += w * d


if __name__ == '__main__':
    import argparse
    from itertools import islice
    from utils.load import load_vocab
    from utils.preprocess import normalize_prob

    parser = argparse.ArgumentParser()
    parser.add_argument('--word_list', '-f', default="./datasets/unigram_freq.csv",
                        help="list of words to create subword vocab")
    parser.add_argument('--n_largest', '-n', type=int, default=20,
                        help="the number of segmentations to show")
    parser.add_argument('--boundary', '-b', action='store_true',
                        help="annotate word boundary")
    parser.add_argument('--interactive', '-i', action='store_true',
                        help="interactive mode")
    args = parser.parse_args()

    logging.info(f"building subword vocab from `{args.word_list}`...")
    subword_count = load_vocab(args.word_list, boundary=args.boundary)
    subword_prob = normalize_prob(subword_count, take_root=False)
    logging.info(f"subword vocab size: {len(subword_prob)}")

    test_words = [
        "lowest",
        "somnambulists", ## aka "sleep-walkers"
        "technically",
        "electronics",
    ]

    def test_word(w):
        if args.boundary:
            w = '<' + w + '>'
        p_prefix = calc_prefix_prob(w, subword_prob)
        print("p_prefix:", '\t'.join(f"{x:.5E}" for x in p_prefix))
        p_suffix = calc_prefix_prob(w[::-1], subword_prob)[::-1]
        print("p_suffix:", '\t'.join(f"{x:.5E}" for x in p_suffix))
        subword_weights = calc_subword_weights(w, subword_prob)
        print("top subword_weights:")
        format_str = "{:%ds}: {:.5E}" % (len(w))
        for sub, w in islice(sorted(subword_weights.items(), key=lambda t: t[1], reverse=True), args.n_largest):
            print(format_str.format(sub, w))

    for w in test_words:
        test_word(w)

    if args.interactive:
        while True:
            w = input().strip()
            if not w:
                break
            test_word(w)
