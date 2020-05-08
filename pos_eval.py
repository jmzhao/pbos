import argparse
import logging
import pickle
from collections import namedtuple
from itertools import chain, repeat

import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from load import load_embedding
from utils.args import add_logging_args, set_logging_config

parser = argparse.ArgumentParser("Evaluate embedding on POS tagging",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset',
    help="path to processed UD dataset")
parser.add_argument('--embeddings',
    help="path to word embeddings")
parser.add_argument('--random_seed', type=int, default=42,
    help="random seed for training the classifier")
add_logging_args(parser)
args = parser.parse_args()

set_logging_config(args)

Instance = namedtuple("Instance", ["sentence", "tags"])

## Load Mimick-style UD data
with open(args.dataset, 'rb') as f:
    dataset = pickle.load(f)
w2i = dataset["w2i"]
t2is = dataset["t2is"]
c2i = dataset["c2i"]
i2w = {i: w for w, i in list(w2i.items())}
i2ts = {
    att: {i: t
          for t, i in list(t2i.items())}
    for att, t2i in list(t2is.items())
}
i2c = {i: c for c, i in list(c2i.items())}

training_instances = dataset["training_instances"]
training_vocab = dataset["training_vocab"]
dev_instances = dataset["dev_instances"]
dev_vocab = dataset["dev_vocab"]
test_instances = dataset["test_instances"]

## Load embeddings
vocab, emb = load_embedding(args.embeddings)
emb_w2i = {w : i for i, w in enumerate(vocab)}

emb_unk_i = vocab.index('<UNK>')
# assert '<UNK>' == vocab[0], vocab[0]
assert '<UNK>' ==   i2w[0],   i2w[0]

## Prepare training and testing data arrays
def make_X(instance, ipad=0, hws=2):
    i_seq = chain(repeat(ipad, hws), instance.sentence, repeat(ipad, hws))
    emb_i_seq = [emb_w2i.get(i2w[i], emb_unk_i) for i in i_seq]
    len_sen = len(instance.sentence)
    ws = 2 * hws + 1
    emb_i_X = [emb_i_seq[i : i + ws] for i in range(len_sen)]
    X = emb.take(emb_i_X, axis=0) # shape: (len, ws, emb_dim)
    X = X.reshape((len_sen, -1)) # shape: (len, ws * emb_dim)
    return X

def make_y(instance, tag_type='POS'):
    return np.array(instance.tags[tag_type])

def make_X_y(instances):
    X = np.concatenate(list(make_X(ins) for ins in tqdm(instances)))
    y = np.concatenate(list(make_y(ins) for ins in instances))
    logging.info(f"X.shape = {X.shape}")
    logging.info(f"y.shape = {y.shape}")
    return X, y

logging.info("building training instances...")
train_X, train_y = make_X_y(training_instances)
logging.info("building test instances...")
test_X,  test_y  = make_X_y(test_instances)

## Train a logistic regression classifier and report scores
logging.info("training...")
clsfr = LogisticRegression(random_state=args.random_seed, verbose=False)
clsfr.fit(train_X, train_y)
# print("Train acc: {}".format(clsfr.score(train_X, train_y)))
print("Test acc:  {}".format(clsfr.score( test_X,  test_y)))
