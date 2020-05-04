from collections import Counter, defaultdict
from functools import lru_cache, partial
import logging

import numpy as np

from utils import normalize_prob
from utils.args import add_logging_args, logging_config


logger = logging.getLogger(__name__)

def get_subword_prob(sub, subword_prob, eps=None, take_root=False):
    prob = subword_prob.get(sub, eps)
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
    else:
        for j in range(1, len(w) + 1):
            for i in range(j):
                sub = w[i:j]
                if sub in subword_vocab:
                    subword_weights.setdefault(sub, 0)
                    subword_weights[sub] += 1

    if len(subword_weights) == 0:
        logger.warning(f"no qualified subwords for '{w}'")
        return {}
    subword_weights = normalize_prob(subword_weights)
    if get_subword_prob and weight_threshold:
        subword_weights = {k : v for k, v in subword_weights.items() if v > weight_threshold}

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
        eps=1e-6,
        take_root=False,
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

            eps (default: 1e-6) - the default subword probability if it is not
                present in `subword_prob`. This is needed to keep the segmenation
                graph connected.
                Only effective when `subword_prob` is present.

            take_root (default: False) - whether take `** ( 1 / len(sub))` when
                getting subword prob.
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
        ))
        self.config = dict(
            embedding_dim=embedding_dim,
            weight_threshold=weight_threshold,
            eps=eps,
            take_root=take_root,
            subword_vocab=subword_vocab,
            subword_prob=subword_prob,
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
    from subwords import build_subword_counter, build_subword_prob, subword_prob_post_process
    from utils import normalize_prob

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
    
    def parse(l):
        k, v = l.split("\t")
        return k, int(v)
    
    with open(args.word_list) as f: 
        subword_count = build_subword_counter((parse(l) for l in f), word_boundary=args.boundary)
    subword_prob = normalize_prob(subword_count)
    
    logger.info(f"subword vocab size: {len(subword_prob)}")

    test_words = [
        "lowest",
        "somnambulists", ## aka "sleep-walkers"
        "technically",
        "electronics",
    ]

    get_subword_prob=partial(
        get_subword_prob,
        subword_prob=subword_prob,
        take_root=args.subword_freq_take_root,
        eps=1e-6
    )

    def test_word(w):
        if args.boundary:
            w = '<' + w + '>'
        p_prefix = calc_prefix_prob(w, get_subword_prob)
        print("p_prefix:", '\t'.join(f"{x:.5e}" for x in p_prefix))
        p_suffix = calc_prefix_prob(w[::-1], get_subword_prob)[::-1]
        print("p_suffix:", '\t'.join(f"{x:.5e}" for x in p_suffix))
        subword_weights = calc_subword_weights(
            w,
            subword_vocab=subword_prob,
            get_subword_prob=get_subword_prob,
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
