from collections import defaultdict

import numpy as np

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
