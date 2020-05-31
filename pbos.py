import logging
from collections import Counter, defaultdict
from functools import lru_cache, partial

import numpy as np

from utils import normalize_prob

logger = logging.getLogger(__name__)

def get_subword_prob(sub, subword_prob, eps=None, take_root=False):
    prob = subword_prob.get(sub, eps if len(sub) == 1 else 0)
    if take_root:
        prob = prob ** (1 / len(sub))
    return prob

def calc_prefix_prob(w, get_subword_prob, backward=False):
    w = w[::-1] if backward else w
    p = [1]
    for i in range(1, len(w) + 1):
        p.append(sum(
            p[j] * get_subword_prob(w[j:i][::-1] if backward else w[j:i])
            for j in range(i)))
    return p[::-1] if backward else p

def calc_subword_weights(
    w,
    *,
    subword_vocab,
    get_subword_prob=None,
    weight_threshold=None,
    normalize=True,
):
    subword_weights = {}
    if get_subword_prob:
        p_prefix = calc_prefix_prob(w, get_subword_prob)
        p_suffix = calc_prefix_prob(w, get_subword_prob, backward=True)
        for j in range(1, len(w) + 1):
            for i in range(j):
                sub = w[i:j]
                if sub in subword_vocab:
                    p_sub = get_subword_prob(sub) * p_prefix[i] * p_suffix[j]
                    subword_weights.setdefault(sub, 0)
                    subword_weights[sub] += p_sub
        if normalize:
            subword_weights = normalize_prob(subword_weights)
        else:
            for k in subword_weights:
                 subword_weights[k] /= p_prefix[-1]
        if weight_threshold:
            subword_weights = {k : v for k, v in subword_weights.items() if v > weight_threshold}
    else:
        for j in range(1, len(w) + 1):
            for i in range(j):
                sub = w[i:j]
                if sub in subword_vocab:
                    subword_weights.setdefault(sub, 0)
                    subword_weights[sub] += 1
        subword_weights = normalize_prob(subword_weights)

    if len(subword_weights) == 0:
        logger.warning(f"no qualified subwords for '{w}'")
        return {}

    return subword_weights


class PBoS:
    def __init__(
        self,
        subword_embedding=None,
        *,
        subword_vocab,
        embedding_dim=None,
        subword_prob=None,
        weight_threshold=None,
        eps=1e-2,
        take_root=False,
        normalize_semb=False,
        subword_weight_normalize=True
    ):
        """
        Params:
            subword_embedding (default: None) - existing subword embeddings.
                If None, initialize an empty set of embeddings.

            embedding_dim (default: None) - embedding dimensions.
                If None, infer from `subword_embedding`.

            subword_prob (default: None) - subword probabilities.
                Used by probabilistic segmentation to calculate subword weights.
                If None, assume uniform probability, i.e. = BoS.

            subword_vocab - subword vocabulary.
                The set of subwords to maintain subword embeddings.
                OOV subwords will be regarded as having zero vector embedding.

            weight_threshold (default: None) - minimum subword weight to consider.
                Extremely low-weighted subword will be discarded for effiency.
                If None, consider subwords with any weights.

            eps (default: 1e-2) - the default subword probability if it is not
                present in `subword_prob`. This is needed to keep the segmenation
                graph connected.
                Only effective when `subword_prob` is present.

            take_root (default: False) - whether take `** ( 1 / len(sub))` when
                getting subword prob.

            subword_weight_normalize (default: True) - whether to normalize
                all final subword weights (a_{s|w})
        """
        self.semb = subword_embedding or defaultdict(float)
        if embedding_dim is None:
            subword_embedding_entry = next(iter(subword_embedding.values()))
            embedding_dim = len(subword_embedding_entry)
        for w in '<>':
            if w in subword_vocab:
                del subword_vocab[w]
        self._calc_subword_weights = lru_cache(maxsize=32)(partial(
            calc_subword_weights,
            subword_vocab=subword_vocab,
            get_subword_prob=partial(
                get_subword_prob,
                subword_prob=subword_prob,
                eps=eps,
                take_root=take_root,
            ) if subword_prob else None,
            weight_threshold=weight_threshold,
            normalize=subword_weight_normalize
        ))
        self.config = dict(
            embedding_dim=embedding_dim,
            weight_threshold=weight_threshold,
            eps=eps,
            take_root=take_root,
            subword_vocab=subword_vocab,
            subword_prob=subword_prob,
            normalize_semb=normalize_semb,
            subword_weight_normalize=subword_weight_normalize,
        )
        self._zero_emb = np.zeros(self.config['embedding_dim'])

    def dump(self, filename) :
        import json, pickle
        with open(filename + '.config.json', 'w') as fout:
            json.dump(self.config, fout)
        with open(filename, 'bw') as bfout :
            pickle.dump(self.semb, bfout)

    @classmethod
    def load(cls, filename) :
        import json, pickle
        try:
            # backward compatibility
            with open(filename, 'rb') as bfin:
                config, semb = pickle.load(bfin)
        except ValueError:
            with open(filename, 'rb') as bfin:
                semb = pickle.load(bfin)
        with open(filename + '.config.json') as fin:
            config = json.load(fin)
        bos = cls(**config)
        bos.semb = semb
        return bos

    @staticmethod
    def _semb_normalized_contrib(w, emb):
        norm = np.linalg.norm(emb)
        return w * emb / norm if norm > 1e-4 else 0

    def embed(self, w):
        subword_weights = self._calc_subword_weights(w)
        logger.debug(Counter(subword_weights).most_common())
        # Will we have performance issue if we put the if check inside sum?
        if self.config['normalize_semb']:
            wemb = sum(
                self._semb_normalized_contrib(w, self.semb[sub])
                for sub, w in subword_weights.items()
            )
        else:
            wemb = sum(
                w * self.semb[sub]
                for sub, w in subword_weights.items()
            )
        return wemb if isinstance(wemb, np.ndarray) else self._zero_emb

    def step(self, w, d):
        subword_weights = self._calc_subword_weights(w)
        for sub, weight in subword_weights.items():
            self.semb[sub] += weight * d
