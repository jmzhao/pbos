import argparse
import logging
import os
import sys
import time

from pbos import PBoS
from subwords import add_word_args, bound_word
from utils.args import add_logging_args, set_logging_config

logger = logging.getLogger(__name__)


def predict(model, word_boundary, queries=None, save=None, pre_trained=None):
    """
    :return: The total time used in prediction
    """
    if pre_trained:
        from load import load_embedding

        pre_trained_vocab, pre_trained_emb = load_embedding(pre_trained)
        pre_trained_emb_lookup = dict(zip(pre_trained_vocab, pre_trained_emb))

    logger.info("loading...")
    model = PBoS.load(model)
    logging.info("generating...")
    if queries:
        fin = open(queries, "r", encoding="utf-8")
    else:
        fin = sys.stdin
    if save:
        save_dir = os.path.dirname(save)
        try:
            os.makedirs(save_dir)
        except FileExistsError:
            logger.warning("Things will get overwritten for directory {}".format(save_dir))
        fout = open(save, "w", encoding="utf-8")
    else:
        fout = sys.stdout

    start = time.time()
    for line in fin:
        ori_query = line.strip()
        query = bound_word(ori_query) if word_boundary else ori_query
        if pre_trained:
            vector = (
                pre_trained_emb_lookup[query]
                if query in pre_trained_emb_lookup
                else model.embed(query)
            )
        else:
            vector = model.embed(query)
        print(ori_query, *vector, file=fout)

    return time.time() - start


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bag of substrings: prediction")
    parser.add_argument("--pre_trained",
                        help="If this variable is specified, only use the model for OOV, "
                             "and use the pre_trainved vectors for query")
    parser.add_argument("--model", required=True)
    parser.add_argument("--save", help="If not specified, use stdin.")
    parser.add_argument("--queries", help="If not specified, use stdout.")
    add_logging_args(parser)
    add_word_args(parser)
    args = parser.parse_args()

    set_logging_config(args)

    predict(args.model, args.word_boundary, args.queries, args.save, args.pre_trained)
