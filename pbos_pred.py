import argparse, datetime, json, logging, os, sys
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm

from pbos import PBoS
from subwords import add_word_args, bound_word
from utils.args import add_logging_args, set_logging_config


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Bag of substrings: prediction")
parser.add_argument("--pre_trained",
    help="If this variable is specified, only use the model for OOV, "
    "and use the pre_trainved vectors for query")
parser.add_argument("--model", required=True)
parser.add_argument("--save",
    help="If not specified, use stdin.")
parser.add_argument("--queries",
    help="If not specified, use stdout.")
add_logging_args(parser)
add_word_args(parser)
args = parser.parse_args()

set_logging_config(args)

if args.pre_trained:
    from load import load_embedding

    pre_trained_vocab, pre_trained_emb = load_embedding(args.pre_trained)
    pre_trained_emb_lookup = dict(zip(pre_trained_vocab, pre_trained_emb))

logger.info("loading...")
model = PBoS.load(args.model)
logger.debug(type(model.semb))
logging.info("generating...")
if args.queries:
    fin = open(args.queries, "r", encoding="utf-8")
else:
    fin = sys.stdin
if args.save:
    save_dir = os.path.dirname(args.save)
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        logger.warning("Things will get overwritten for directory {}".format(save_dir))
    fout = open(args.save, "w", encoding="utf-8")
else:
    fout = sys.stdout

for line in fin:
    ori_query = line.strip()
    query = bound_word(ori_query) if args.word_boundary else ori_query
    if args.pre_trained:
        vector = (
            pre_trained_emb_lookup[query]
            if query in pre_trained_emb_lookup
            else model.embed(query)
        )
    else:
        vector = model.embed(query)
    print(ori_query, *vector, file=fout)
