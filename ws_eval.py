#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""Evaluate word similarity.
Adapted from: `https://github.com/facebookresearch/fastText/blob/316b4c9f499669f0cacc989c32bf2cef23a8f9ac/eval.py`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from scipy import stats
import os
import math


def compat_splitting(line):
    return line.decode('utf8').split()


def similarity(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / n1 / n2

def edit_distence(s1, s2) :
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def editsim(w1, w2):
    return -edit_distence(w1, w2) / max(len(w1), len(w2))


def load_vectors(modelPath):
    vectors = {}
    fin = open(modelPath, 'rb')
    for _, line in enumerate(fin):
        try:
            tab = compat_splitting(line)
            vec = np.array([float(x) for x in tab[1:]], dtype=float)
            word = tab[0]
            if np.linalg.norm(vec) < 1e-6:
                continue
            if not word in vectors:
                vectors[word] = vec
        except ValueError:
            continue
        except UnicodeDecodeError:
            continue
    fin.close()
    return vectors


def eval_ws(modelPath, dataPath, lower, oov_handling="drop"):
    mysim = []
    gold = []
    # words =  []
    drop = 0.0
    nwords = 0.0

    if modelPath != "EditSim":
        vectors = load_vectors(modelPath)

    fin = open(dataPath, 'rb')
    for line in fin:
        tline = compat_splitting(line)
        word1 = tline[0]
        word2 = tline[1]
        golden_score = float(tline[2])

        if lower:
            word1, word2 = word1.lower(), word2.lower()
        nwords = nwords + 1.0

        # words.append((word1, word2))

        if modelPath == "EditSim":
            d = editsim(word1, word2)
        else:
            if (word1 in vectors) and (word2 in vectors):
                v1 = vectors[word1]
                v2 = vectors[word2]
                d = similarity(v1, v2)
            else:
                drop = drop + 1.0
                if oov_handling == "zero":
                    d = 0
                else:
                    continue

        mysim.append(d)
        gold.append(golden_score)
    fin.close()
    # for _, g, m, (w1, w2) in sorted(zip(np.abs(stats.zscore(mysim) - stats.zscore(gold)), gold, mysim, words)):
    #     print(f"{g:.2f} {m:.2f} {w1} {w2}")
    corr = stats.spearmanr(mysim, gold)
    dataset = os.path.basename(dataPath)
    return "{:20s}: {:2.0f}  (OOV: {:2.0f}%, {}, lower={})".format(
        dataset,
        corr[0] * 100,
        math.ceil(drop / nwords * 100.0),
        oov_handling,
        lower
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--model',
        '-m',
        dest='modelPath',
        action='store',
        required=True,
        help='path to model'
    )
    parser.add_argument(
        '--data',
        '-d',
        dest='dataPath',
        action='store',
        required=True,
        help='path to data'
    )
    parser.add_argument('--lower', action='store_true', default=True)
    parser.add_argument('--no_lower', dest='lower', action='store_false')
    parser.add_argument('--oov_handling', default='drop', choices=['drop', 'zero'])
    args = parser.parse_args()

    print(eval_ws(args.modelPath, args.dataPath, args.lower, args.oov_handling))
