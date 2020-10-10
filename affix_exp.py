from functools import partial
from itertools import islice

import numpy as np
import sklearn.metrics

from datasets.affix import prepare_affix_paths
from pbos import PBoS
from utils import get_substrings


def bos_predictor(w, rng, possible_affixes):
    affixes = list(sorted(possible_affixes & set(get_substrings(w)))) ## sort to ensure reproducibility
    return rng.choice(affixes)


def pbos_predictor(w, model, possible_affixes):
    subword_weights = model._calc_subword_weights(w)
    score, affix = max((subword_weights[af], af)
        for af in possible_affixes if af in subword_weights)
    return affix


def print_metrics(true_y, pred_y):
    for average_scheme in ('micro', 'macro'):
        for score_name in ('precision', 'recall', 'f1'):
            print("{} {}:\t{}".format(
                average_scheme,
                score_name,
                getattr(sklearn.metrics, score_name + "_score")(
                    true_y, pred_y,
                    average=average_scheme,
                ),
            ))


def main(args):
    word_affix_pairs = []
    with open(prepare_affix_paths().raw_path) as fin:
        for line in islice(fin, 1, None): ## skip the title row
            ## row fmt: affix	stem	stemPOS	derived	derivedPOS	type	...
            affix, stem, _, derived, _, split = line.split()[:6]
            affix = affix.strip('-')
            if affix != 'y':
                word_affix_pairs.append((derived, affix))

    possible_affixes = set(af for w, af in word_affix_pairs)
    print(f"# interesting possible affixes: {len(possible_affixes)}")

    interesting_word_affix_pairs = [
        (w, af)
        for w, af in word_affix_pairs
        if len(possible_affixes & set(get_substrings(w))) > 1
    ]
    print(f"# interesting words: {len(interesting_word_affix_pairs)}")


    true_affixes = [af for w, af in interesting_word_affix_pairs]
    print("bos affix prediction:")
    bos_predict = partial(
        bos_predictor,
        rng=np.random.RandomState(args.seed),
        possible_affixes=possible_affixes,
    )
    bos_affixes = [bos_predict(w) for w, af in interesting_word_affix_pairs]
    print_metrics(true_affixes, bos_affixes)
    print("pbos affix prediction:")
    pbos_predict = partial(
        pbos_predictor,
        model=PBoS.load(args.pbos),
        possible_affixes=possible_affixes,
    )
    pbos_affixes = [pbos_predict(w) for w, af in interesting_word_affix_pairs]
    print_metrics(true_affixes, pbos_affixes)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PBoS trainer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pbos',
        help="path to pbos model")
    parser.add_argument('--seed', type=int, default=1337,
        help="random seed")
    args = parser.parse_args()
    main(args)
