from collections import defaultdict, Counter
from random import choice

import numpy as np
from tqdm import tqdm

from pbos import PBoS

import argparse, datetime, json, logging, os

parser = argparse.ArgumentParser(description="Bag of substrings: prediction")
parser.add_argument("--pre_trained",
    help="If this variable is specified, only use the model for OOV, "
    "and use the pre_trainved vectors for query")
parser.add_argument("--model", required=True)
parser.add_argument("--save", required=True)
parser.add_argument("--queries", required=True)
parser.add_argument("--loglevel", default="INFO",
    help="log level used by logging module")
args = parser.parse_args()

numeric_level = getattr(logging, args.loglevel.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError("Invalid log level: %s" % args.loglevel)
logging.basicConfig(level=numeric_level)

if args.pre_trained:
    from load import load_embedding

    pre_trained_vocab, pre_trained_emb = load_embedding(args.pre_trained)

logging.info("loading...")
model = PBoS.load(args.model)
logging.debug(type(model.semb))
logging.info("generating...")
with open(args.queries, "r", encoding="utf-8") as fin:
    queries = [l.strip() for l in fin]
if args.pre_trained:
    pre_trained_vocab_set = set(pre_trained_vocab)
    vectors = [(
        pre_trained_emb[pre_trained_vocab.index(w)]
        if w in pre_trained_vocab_set
        else model.embed(w)
    ) for w in queries]
else:
    vectors = [model.embed(w) for w in queries]
logging.info("saving...")
save_dir = os.path.dirname(args.save)
try:
    os.makedirs(save_dir)
except FileExistsError:
    logging.warning("Things will get overwritten for directory {}".format(save_dir))
with open(args.save, "w", encoding="utf-8") as fout:
    for w, e in zip(queries, vectors):
        print(w, *e, file=fout)
