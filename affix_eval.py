import argparse
import importlib
from itertools import islice
import logging
import pickle
from collections import namedtuple
from itertools import chain, repeat

import numpy as np
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
from tqdm import tqdm

from load import load_embedding
from utils.args import add_logging_args, set_logging_config

parser = argparse.ArgumentParser("Evaluate embedding on affix prediction.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default="affix",
    choices=["affix"],
    help="path to affix prediction dataset")
parser.add_argument('--embeddings',
    help="path to word embeddings")
parser.add_argument('--lower', action='store_true', default=True)
parser.add_argument('--no_lower', dest='lower', action='store_false')
parser.add_argument('--random_seed', type=int, default=42,
    help="random seed for training the classifier")
add_logging_args(parser)
args = parser.parse_args()

set_logging_config(args)


Instance = namedtuple("Instance", "word affix")


## Load affix prediction data
prepare_affix_paths = getattr(
    importlib.import_module(f"datasets.{args.dataset}"),
    f"prepare_{args.dataset}_paths",
)
affix_raw_path = prepare_affix_paths().raw_path
dataset = {"test": [], "train": []}
with open(affix_raw_path) as fin:
    for line in islice(fin, 1, None): ## skip the title row
        ## row fmt: affix	stem	stemPOS	derived	derivedPOS	type	...
        affix, stem, _, derived, _, split = line.split()[:6]
        if args.lower:
            derived = derived.lower()
        dataset[split].append(Instance(word = derived, affix = affix))
all_affixes = set(ins.affix for ins in dataset["train"])
affixes_a2i = {a : i for i, a in enumerate(sorted(all_affixes))}
from collections import Counter
logging.info(Counter(ins.affix for ins in dataset["test"]))


## Load embeddings
vocab, emb = load_embedding(args.embeddings)
emb_w2i = {w : i for i, w in enumerate(vocab)}

## Prepare training and testing data arrays
def make_X(instance):
    X = emb[emb_w2i[instance.word]]
    return X

def make_y(instance):
    return np.array(affixes_a2i[instance.affix])

def make_X_y(instances):
    X = np.stack([make_X(ins) for ins in tqdm(instances)])
    logging.info(f"X.shape = {X.shape}")
    y = np.stack([make_y(ins) for ins in instances])
    logging.info(f"y.shape = {y.shape}")
    return X, y

logging.info("building training instances...")
train_X, train_y = make_X_y(dataset["train"])
logging.info("building test instances...")
test_X,  test_y  = make_X_y(dataset["test"])

## Train a logistic regression classifier and report scores
logging.info("training...")
clsfr = LogisticRegression(random_state=args.random_seed, verbose=False)
clsfr.fit(train_X, train_y)
# print("Train acc: {}".format(clsfr.score(train_X, train_y)))
print("Test acc:  {}".format(clsfr.score( test_X,  test_y)))
pred_y = clsfr.predict(test_X)
for average_scheme in ('micro', 'macro'):
    for score_name in ('precision', 'recall', 'f1'):
        print("{} {}:\t{}".format(
            average_scheme,
            score_name,
            getattr(sklearn.metrics, score_name + "_score")(
                test_y, pred_y,
                average=average_scheme,
            ),
        ))
