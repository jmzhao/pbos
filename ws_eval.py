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
import argparse


def compat_splitting(line):
    return line.decode('utf8').split()


def similarity(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / n1 / n2


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
args = parser.parse_args()

vectors = {}
fin = open(args.modelPath, 'rb')
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

mysim = []
gold = []
drop = 0.0
nwords = 0.0

fin = open(args.dataPath, 'rb')
for line in fin:
    tline = compat_splitting(line)
    word1 = tline[0]
    word2 = tline[1]
    if args.lower:
        word1, word2 = word1.lower(), word2.lower()
    nwords = nwords + 1.0

    if (word1 in vectors) and (word2 in vectors):
        v1 = vectors[word1]
        v2 = vectors[word2]
        d = similarity(v1, v2)
        mysim.append(d)
        gold.append(float(tline[2]))
    else:
        # print("dropped", (word1, word2))
        drop = drop + 1.0
fin.close()

corr = stats.spearmanr(mysim, gold)
dataset = os.path.basename(args.dataPath)
print(
    "{0:20s}: {1:2.0f}  (OOV: {2:2.0f}%)"
    .format(dataset, corr[0] * 100, math.ceil(drop / nwords * 100.0))
)