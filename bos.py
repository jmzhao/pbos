from collections import defaultdict
from functools import lru_cache, partial
import logging

import numpy as np

from utils.preprocess import normalize_prob


logging.basicConfig(level=logging.DEBUG)

def _substrings(s, lmin, lmax) :
    s = '<' + s + '>'
    for i in range(len(s)) :
        s0 = s[i:]
        for j in range(lmin, 1 + min(lmax, len(s0))) :
            yield s0[:j]
def _cached(f) :
    c = dict()
    def cf(*args) :
        if args in c :
            return c[args]
        else :
            return c.setdefault(args, f(*args))
    return cf

def make_substrings(lmin, lmax, cached, hash) :
    if hash is None :
        def substrings(w) :
            return [s for s in _substrings(w, lmin, lmax)]
    else :
        def substrings(w) :
            return [hash(s) for s in _substrings(w, lmin, lmax)]
    if cached :
        substrings = _cached(substrings)
    return substrings

def FNV_1a(s) :
  h = 2166136261
  for c in s :
    h = h ^ ord(c)
    h = (h * 16777619) & 0xffffffff
  return h

class Hash :
    def __init__(self, max_n) :
        self.max_n = max_n
    def __call__(self, x) :
        return FNV_1a(x) % self.max_n

class BoS :
    def __init__(self, embedding_dim, lmin=3, lmax=6, cached=True, hashed=False, hash_range=None) :
        self.semb = defaultdict(float)
        self.substrings = make_substrings(lmin=lmin, lmax=lmax, cached=cached, hash=Hash(max_n=hash_range) if hashed else None)
        self.config = dict(embedding_dim=embedding_dim, lmin=lmin, lmax=lmax, cached=cached, hashed=hashed, hash_range=hash_range)

    def dump(self, filename) :
        import json, pickle
        json.dump(self.config, open(filename + '.config.json', 'w'))
        with open(filename, 'bw') as bfout :
            pickle.dump((self.config, self.semb), bfout)

    @classmethod
    def load(cls, filename) :
        import pickle
        config, semb = pickle.load(open(filename, 'rb'))
        bos = cls(**config)
        bos.semb = semb
        return bos

    def embed(self, w) :
        ss = self.substrings(w)
        e = sum(self.semb[s] for s in ss) / len(ss)
        return e if isinstance(e, np.ndarray) else np.zeros(self.config['embedding_dim'])

    def step(self, w, d) :
        for s in self.substrings(w) :
            self.semb[s] += d

def calc_prefix_prob(w, subword_prob, backward=False):
    w = w[::-1] if backward else w
    p = [1]
    for i in range(1, len(w) + 1):
        p.append(sum(p[j] * subword_prob.get(w[j:i][::-1] if backward else w[j:i], 0) for j in range(i)))
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


class PBoS (BoS):
    def __init__(self, embedding_dim, * , subword_prob, boundary=False):
        self.semb = defaultdict(float)
        self.subword_prob = subword_prob
        self._calc_subword_weights = lru_cache(maxsize=32)(
            partial(calc_subword_weights, subword_prob=subword_prob, boundary=boundary))
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
