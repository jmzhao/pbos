from itertools import count
import argparse
import datetime
import json
import logging
import os
import pickle
from itertools import count
from time import time

import numpy as np
from utils.load import load_vocab, build_substring_counts
from utils.preprocess import normalize_prob

from pbos import PBoS


# from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Bag of substrings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    return parser.parse_args()

def add_args(parser):
    parser.add_argument('--target_vectors', required=True,
        help='pretrained target word vectors')
    parser.add_argument('--model_path', required=True,
        default="./results/run_{timestamp}/model.pbos",
        help='save path')
    parser.add_argument('--loglevel', default='INFO',
        help='log level used by logging module')
    add_training_args(parser)
    add_model_args(parser)
    add_subword_args(parser)

def add_training_args(parser):
    training_group = parser.add_argument_group('training arguments')
    training_group.add_argument('--epochs', type=int, default=20,
        help='number of training epochs')
    training_group.add_argument('--lr', type=float, default=1.0,
        help='learning rate')
    training_group.add_argument('--random_seed', type=int, default=42,
                                help='random seed used in training')
    training_group.add_argument('--lr_decay', action='store_true', default=True,
        help='reduce learning learning rate between epochs')

def add_model_args(parser):
    model_group = parser.add_argument_group('PBoS model arguments')
    parser.add_argument('--word_list', help="list of words to create subword vocab")
    parser.add_argument('--word_list_has_freq', action='store_true', default=True,
                        help="if the word list contains frequency")
    parser.add_argument('--word_list_size', type=int, default=10000000,
                        help="the maximum size of wordlist, ignore if there is more")
    parser.add_argument('--mock_bos', action='store_true',
        help="mock BoS model")

def add_subword_args(parser):
    parser.add_argument('--boundary', '-b', action='store_true',
        help="annotate word boundary")
    parser.add_argument('--sub_min_count', type=int, default=5,
        help="subword min count for it to be included in vocab")
    parser.add_argument('--sub_min_len', type=int, default=3,
        help="subword min length for it to be included in vocab")
    parser.add_argument('--sub_max_len', type=int, default=None,
        help="subword max length for it to be included in vocab")


def main(args):
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(level=numeric_level)

    save_path = args.model_path.format(timestamp=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    save_dir, _ = os.path.split(save_path)
    try :
        os.makedirs(save_dir)
    except FileExistsError :
        logging.warning("Things will get overwritten for directory {}".format(save_dir))

    with open(os.path.join(save_dir, 'args.json'), 'w') as fout :
        json.dump(vars(args), fout)

    logging.info('loading target vectors...')
    _, ext = os.path.splitext(args.target_vectors)
    if ext in (".txt", ) :
        vocab, emb = [], []
        for i, line in zip(count(1), open(args.target_vectors)) :
            ss = line.split()
            vocab.append(ss[0])
            emb.append([float(x) for x in ss[1:]])
            if i % 10000 == 0 :
                logging.info('{} lines loaded'.format(i))
    elif ext in (".pickle", ".pkl") :
        vocab, emb = pickle.load(open(args.target_vectors, 'rb'), encoding='bytes')
    else :
        raise ValueError('Unsupported target vector file extent: {}'.format(args.target_vectors))
    emb = np.array(emb)

    logging.info(f"building subword vocab from `{args.word_list or 'vocab'}`...")
    if args.word_list:
        subword_count = load_vocab(
            args.word_list,
            boundary=args.boundary,
            cutoff=args.sub_min_count,
            min_len=args.sub_min_len,
            max_len=args.sub_max_len,
            has_freq=args.word_list_has_freq,
            word_list_size=args.word_list_size,
        )
    else:
        subword_count = build_substring_counts(
            vocab,
            boundary=args.boundary,
            cutoff=args.sub_min_count,
            min_len=args.sub_min_len,
            max_len=args.sub_max_len,
        )

    subword_prob = normalize_prob(subword_count, take_root=True)
    logging.info(f"subword vocab size: {len(subword_prob)}")

    def MSE(pred, target) :
        return sum((pred - target) ** 2) / 2 #/ len(target)
    def MSE_backward(pred, target) :
        return (pred - target) #/ len(target)

    model = PBoS(embedding_dim=len(emb[0]),
        subword_prob=subword_prob,
        boundary=args.boundary,
        mock_bos=args.mock_bos,
    )
    start_time = time()
    # np.random.seed(args.random_seed)
    for i_epoch in range(args.epochs) :
        h = []
        h_epoch = []
        lr = args.lr / (1 + i_epoch) ** 0.5 if args.lr_decay else args.lr
        logging.info('epoch {:>2} / {} | lr {:.5f}'.format(1 + i_epoch, args.epochs, lr))
        epoch_start_time = time()
        for i_inst, wi in zip(count(1), np.random.choice(len(vocab), len(vocab), replace=False)) :
            w = vocab[wi]
            e = model.embed(w)
            tar = emb[wi]
            g = MSE_backward(e, tar)

            if i_inst % 20 == 0 :
                loss = MSE(e, tar) / len(tar) # only average over dimension for easy reading
                h.append(loss)
            if i_inst % 10000 == 0 :
                width = len(f"{len(vocab)}")
                fmt = 'processed {:%d}/{:%d} | loss {:.5f}' % (width, width)
                logging.info(fmt.format(i_inst, len(vocab), np.average(h)))
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
