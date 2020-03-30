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
from utils import file_tqdm
from utils.args import add_logging_args, logging_config


def parse_args():
    parser = argparse.ArgumentParser(description='Bag of substrings trainer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    return parser.parse_args()

def add_args(parser):
    parser.add_argument('--target_vectors', required=True,
        help='pretrained target word vectors')
    parser.add_argument('--model_path', required=True,
        default="./results/run_{timestamp}/model.pbos",
        help='save path')
    add_logging_args(parser)
    add_training_args(parser)
    add_model_args(parser)
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
    return parser

def add_model_args(parser):
    group = parser.add_argument_group('PBoS model arguments')
    group.add_argument('--subword_vocab', required=True,
        help="list of subwords to maintain subword embeddings")
    group.add_argument('--subword_prob',
        help="dict of subwords and their likelihood of presence. "
        "If not specified, assume uniform likelihood, aka fall back to BoS.")
    group.add_argument('--subword_weight_threshold', type=float, default=1e-3,
        help="minimum weight of a subword within a word for it to contribute "
        "to the word embedding")
    group.add_argument('--subword_prob_eps', type=float, default=1e-6,
        help="default likelihood of a subword if it is not present in "
        "the given `subword_prob`")
    return parser


def main(args):
    logging_config(args)

    save_path = args.model_path.format(timestamp=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    save_dir, _ = os.path.split(save_path)
    try :
        os.makedirs(save_dir)
    except FileExistsError :
        logging.warning("Things will get overwritten for directory {}".format(save_dir))

    with open(os.path.join(save_dir, 'args.json'), 'w') as fout :
        json.dump(vars(args), fout)

    logging.info(f'loading target vectors from `{args.target_vectors}`...')
    target_words, target_emb = load_embedding(args.target_vectors, show_progress=True)
    logging.info(f'embeddings loaded with {len(target_words)} words')

    logging.info(f"loading subword vocab from `{args.subword_vocab}`...")
    with open(args.subword_vocab) as fin:
        subword_vocab = dict(json.loads(line) for line in file_tqdm(fin))
    logging.info(f"subword vocab size: {len(subword_vocab)}")

    if args.subword_prob:
        logging.info(f"loading subword prob from `{args.subword_prob}`...")
        with open(args.subword_prob) as fin:
            subword_prob = dict(json.loads(line) for line in file_tqdm(fin))
    else:
        subword_prob = None

    np.random.seed(args.random_seed)

    def MSE(pred, target) :
        return sum((pred - target) ** 2) / 2
    def MSE_backward(pred, target) :
        return (pred - target)

    model = PBoS(
        embedding_dim=len(target_emb[0]),
        subword_vocab=subword_vocab,
        subword_prob=subword_prob,
        weight_threshold=args.subword_weight_threshold,
        eps=args.subword_prob_eps,
    )
    start_time = time()
    for i_epoch in range(args.epochs) :
        h = []
        h_epoch = []
        lr = args.lr / (1 + i_epoch) ** 0.5 if args.lr_decay else args.lr
        logging.info('epoch {:>2} / {} | lr {:.5f}'.format(1 + i_epoch, args.epochs, lr))
        epoch_start_time = time()
        for i_inst, wi in enumerate(
            np.random.choice(len(target_words), len(target_words), replace=False),
            start=1,
        ) :
            w = target_words[wi]
            e = model.embed(w)
            tar = target_emb[wi]
            g = MSE_backward(e, tar)

            if i_inst % 20 == 0 :
                loss = MSE(e, tar) / len(tar) # average over dimension for easy reading
                h.append(loss)
            if i_inst % 10000 == 0 :
                width = len(f"{len(target_words)}")
                fmt = 'processed {:%d}/{:%d} | loss {:.5f}' % (width, width)
                logging.info(fmt.format(i_inst, len(target_words), np.average(h)))
                h_epoch.extend(h)
                h = []

            d = - lr * g
            model.step(w, d)
        now_time = time()
        logging.info('epoch {i_epoch:>2} / {n_epoch} | loss {loss:.5f} | time {epoch_time:.2f}s / {training_time:.2f}s'.format(
            i_epoch = 1 + i_epoch, n_epoch = args.epochs,
            loss = np.average(h_epoch),
            epoch_time = now_time - epoch_start_time,
            training_time = now_time - start_time,
        ))

    logging.info('saving model...')
    model.dump(save_path)


if __name__ == '__main__':
    main(parse_args())
