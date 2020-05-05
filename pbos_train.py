import argparse
import datetime
import json
import logging
import os
import pickle
from time import time

import numpy as np
from tqdm import tqdm

from pbos import PBoS
from load import load_embedding
from subwords import (
    add_subword_prob_args,
    add_word_args,
    bound_word,
    subword_prob_post_process,
)
from utils import file_tqdm
from utils.args import add_logging_args, logging_config


logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='PBoS trainer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    return parser.parse_args()

def add_args(parser):
    parser.add_argument('--target_vectors', required=True,
        help='pretrained target word vectors')
    parser.add_argument('--model_path', required=True,
        default="./results/run_{timestamp}/model.pbos",
        help='save path')
    add_training_args(parser)
    add_model_args(parser)
    add_word_args(parser)
    add_subword_prob_args(parser)
    add_logging_args(parser)
    return parser

def add_training_args(parser):
    group = parser.add_argument_group('training hyperparameters')
    group.add_argument('--epochs', type=int, default=20,
        help='number of training epochs')
    group.add_argument('--lr', type=float, default=1.0,
        help='learning rate')
    group.add_argument('--random_seed', type=int, default=42,
        help='random seed used in training')
    group.add_argument('--lr_decay', action='store_true', default=True,
        help='reduce learning learning rate between epochs')
    group.add_argument('--no_lr_decay', dest='lr_decay', action='store_false')
    return group

def add_model_args(parser):
    group = parser.add_argument_group('PBoS model arguments')
    group.add_argument('--subword_vocab', required=True,
        help="list of subwords to maintain subword embeddings")
    group.add_argument('--subword_prob',
        help="dict of subwords and their likelihood of presence. "
        "If not specified, assume uniform likelihood, aka fall back to BoS.")
    group.add_argument('--subword_weight_threshold', type=float,
        help="minimum weight of a subword within a word for it to contribute "
        "to the word embedding")
    group.add_argument('--subword_prob_eps', type=float, default=1e-2,
        help="default likelihood of a subword if it is not present in "
        "the given `subword_prob`")
    return group


def main(args):
    logging_config(args)
    logger.info(json.dumps(args if isinstance(args, dict) else vars(args), indent=2))

    save_path = args.model_path.format(
        timestamp=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    save_dir, _ = os.path.split(save_path)
    try :
        os.makedirs(save_dir)
    except FileExistsError :
        logger.warning(
            "Things will get overwritten for directory {}".format(save_dir))

    with open(os.path.join(save_dir, 'args.json'), 'w') as fout :
        json.dump(vars(args), fout)

    logger.info(f'loading target vectors from `{args.target_vectors}`...')
    target_words, target_embs = \
        load_embedding(args.target_vectors, show_progress=True)
    logger.info(f'embeddings loaded with {len(target_words)} words')

    logger.info(f"loading subword vocab from `{args.subword_vocab}`...")
    with open(args.subword_vocab) as fin:
        subword_vocab = dict(json.loads(line) for line in file_tqdm(fin))
    logger.info(f"subword vocab size: {len(subword_vocab)}")

    if args.subword_prob:
        logger.info(f"loading subword prob from `{args.subword_prob}`...")
        with open(args.subword_prob) as fin:
            subword_prob = dict(json.loads(line) for line in file_tqdm(fin))
        subword_prob = subword_prob_post_process(
            subword_prob,
            min_prob=args.subword_prob_min_prob,
            # take_root=args.subword_prob_take_root,
        )
    else:
        subword_prob = None

    np.random.seed(args.random_seed)

    def MSE(pred, target) :
        return sum((pred - target) ** 2) / 2
    def MSE_backward(pred, target) :
        return (pred - target)

    model = PBoS(
        embedding_dim=len(target_embs[0]),
        subword_vocab=subword_vocab,
        subword_prob=subword_prob,
        weight_threshold=args.subword_weight_threshold,
        eps=args.subword_prob_eps,
        take_root=args.subword_prob_take_root,
    )
    start_time = time()
    for i_epoch in range(args.epochs) :
        h = []
        h_epoch = []
        lr = args.lr / (1 + i_epoch) ** 0.5 if args.lr_decay else args.lr
        logger.info('epoch {:>2} / {} | lr {:.5f}'.format(1 + i_epoch, args.epochs, lr))
        epoch_start_time = time()
        for i_inst, wi in enumerate(
            np.random.choice(len(target_words), len(target_words), replace=False),
            start=1,
        ) :
            target_emb = target_embs[wi]
            word = target_words[wi]
            model_word = bound_word(word) if args.word_boundary else word
            model_emb = model.embed(model_word)
            grad = MSE_backward(model_emb, target_emb)

            if i_inst % 20 == 0 :
                loss = MSE(model_emb, target_emb) / len(target_emb) # average over dimension for easy reading
                h.append(loss)
            if i_inst % 10000 == 0 :
                width = len(f"{len(target_words)}")
                fmt = 'processed {:%d}/{:%d} | loss {:.5f}' % (width, width)
                logger.info(fmt.format(i_inst, len(target_words), np.average(h)))
                h_epoch.extend(h)
                h = []

            d = - lr * grad
            model.step(model_word, d)
        now_time = time()
        logger.info('epoch {i_epoch:>2} / {n_epoch} | loss {loss:.5f} | time {epoch_time:.2f}s / {training_time:.2f}s'.format(
            i_epoch = 1 + i_epoch, n_epoch = args.epochs,
            loss = np.average(h_epoch),
            epoch_time = now_time - epoch_start_time,
            training_time = now_time - start_time,
        ))

    logger.info('saving model...')
    model.dump(save_path)


if __name__ == '__main__':
    main(parse_args())
