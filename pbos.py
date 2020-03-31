from collections import Counter, defaultdict
from functools import lru_cache, partial
import logging

import numpy as np

from utils import normalize_prob
from utils.args import add_logging_args, logging_config


logger = logging.getLogger(__name__)

def calc_prefix_prob(w, subword_prob, backward=False, eps=1e-6):
    w = w[::-1] if backward else w
    p = [1]
    for i in range(1, len(w) + 1):
        p.append(sum(p[j] * subword_prob.get(w[j:i][::-1] if backward else w[j:i], eps) for j in range(i)))
    return p[::-1] if backward else p

def calc_subword_weights(
    w,
    *,
    subword_prob,
    subword_vocab,
    weight_threshold=None,
    eps=1e-6,
):
    subword_weights = {}
    if subword_prob:
        p_prefix = calc_prefix_prob(w, subword_prob, eps=eps)
        p_suffix = calc_prefix_prob(w, subword_prob, eps=eps, backward=True)
        for j in range(1, len(w) + 1):
            for i in range(j):
                sub = w[i:j]
                if sub in subword_vocab:
                    p_sub = subword_prob.get(sub, eps) * p_prefix[i] * p_suffix[j]
                    subword_weights.setdefault(sub, 0)
                    subword_weights[sub] += p_sub
    else:
        for j in range(1, len(w) + 1):
            for i in range(j):
                sub = w[i:j]
                if sub in subword_vocab:
                    subword_weights[sub] = 1
    try:
        if weight_threshold:
            return {k : v for k, v in normalize_prob(subword_weights).items() if v > weight_threshold}
        else:
            return normalize_prob(subword_weights)
    except ZeroDivisionError:
        logger.warning(f"zero weights for '{w}'")
        return {}


class PBoS:
    def __init__(
        self,
        subword_embedding=None,
        *,
        embedding_dim=None,
        subword_prob=None,
        subword_vocab,
        weight_threshold=1e-3,
        eps=1e-6,
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

            weight_threshold (default: 1e-3) - minimum subword weight to consider.
                Extremely low-weighted subword will be discarded for effiency.
                If None, consider subwords with any weights.

            eps (default: 1e-6) - the default subword probability if it is not
                present in `subword_prob`. This is needed to keep the segmenation
                graph connected.
                Only effective when `subword_prob` is present.
        """
        self.semb = subword_embedding or defaultdict(float)
        if embedding_dim is None:
            subword_embedding_entry = next(iter(subword_embedding.values()))
            embedding_dim = len(subword_embedding_entry)
        self._calc_subword_weights = lru_cache(maxsize=32)(partial(
            calc_subword_weights,
            subword_prob=subword_prob,
            subword_vocab=subword_vocab,
            weight_threshold=weight_threshold,
            eps=eps,
        ))
        self.config = dict(
            embedding_dim=embedding_dim,
            weight_threshold=weight_threshold,
            eps=eps,
            subword_prob=subword_prob,
            subword_vocab=subword_vocab,
        )
        self._zero_emb = np.zeros(self.config['embedding_dim'])

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

    def embed(self, w):
        subword_weights = self._calc_subword_weights(w)
        logger.debug(Counter(subword_weights).most_common())
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
    parser.add_argument('--subword_weight_threshold', '-swt', type=float,
                        help="the minimum weight of a subword to be considered")
    parser.add_argument('--subword_freq_take_root', '-sftr', action='store_true',
                        help="take root when normalize subword frequencies into probabilities")
    parser.add_argument('--interactive', '-i', action='store_true',
                        help="interactive mode")
    add_logging_args(parser)
    args = parser.parse_args()

    logging_config(args)

    logger.info(f"building subword vocab from `{args.word_list}`...")
    subword_count = load_vocab(args.word_list, boundary=args.boundary, has_freq=True)
    subword_prob = normalize_prob(subword_count, take_root=args.subword_freq_take_root)
    logger.info(f"subword vocab size: {len(subword_prob)}")

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
        print("p_prefix:", '\t'.join(f"{x:.5e}" for x in p_prefix))
        p_suffix = calc_prefix_prob(w[::-1], subword_prob)[::-1]
        print("p_suffix:", '\t'.join(f"{x:.5e}" for x in p_suffix))
        subword_weights = calc_subword_weights(
            w,
            subword_prob=subword_prob,
            subword_vocab=subword_prob,
            weight_threshold=args.subword_weight_threshold,
        )
        print("top subword_weights:")
        format_str = "{:%ds}: {:.5e}" % (len(w))
        for sub, w in islice(sorted(subword_weights.items(), key=lambda t: t[1], reverse=True), args.n_largest):
            print(format_str.format(sub, w))

    for w in test_words:
        print("Word:", w)
        test_word(w)

    if args.interactive:
        while True:
            w = input().strip()
            if not w:
                break
            test_word(w)
