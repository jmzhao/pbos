from collections import Counter
from itertools import count
from time import time
from random import choice
import os, pickle

import numpy as np
# from tqdm import tqdm

from bos import BoS, Hash

import argparse, datetime, json, logging, os
parser = argparse.ArgumentParser(description='Bag of substrings',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--target', required=True,
    help='pretrained target word vectors')
parser.add_argument('--save', required=True,
    help='save dir')
parser.add_argument('--no-timestamp', action='store_true',
    help='add no timestamp to save dir')
parser.add_argument('--loglevel', default='INFO',
    help='log level used by logging module')
training_group = parser.add_argument_group('training arguments')
training_group.add_argument('--epochs', type=int, default=2,
    help='number of training epochs')
training_group.add_argument('--lr', type=float, default=1.0,
    help='learning rate')
model_group = parser.add_argument_group('BoS model arguments')
model_group.add_argument('--cached', action='store_true',
    help='cached substring computations')
model_group.add_argument('--lmin', type=int, default=3,
    help='substring minimum length (inclusive)')
model_group.add_argument('--lmax', type=int, default=6,
    help='substring maximum length (inclusive)')
hashing_group = model_group.add_argument_group('hashing arguments')
hashing_group.add_argument('--hashed', action='store_true',
    help='hashing substrings with FNV-1a')
hashing_group.add_argument('--hash-range', type=int, default=2000000,
    help='hashing output range')
args = parser.parse_args()

numeric_level = getattr(logging, args.loglevel.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % args.loglevel)
logging.basicConfig(level=numeric_level)

save_dir = args.save if args.no_timestamp else \
    os.path.join(args.save, datetime.datetime.now().strftime('run_%Y-%m-%d_%H-%M-%S'))
try :
    os.makedirs(save_dir)
except FileExistsError :
    logging.warning("Things will get overwritten for directory {}".format(save_dir))

with open(os.path.join(save_dir, 'args.json'), 'w') as fout :
    json.dump(vars(args), fout)

logging.info('loading target vectors...')
_, ext = os.path.splitext(args.target)
if ext in (".txt", ) :
    vocab, emb = [], []
    for i, line in zip(count(1), open(args.target)) :
        ss = line.split()
        vocab.append(ss[0])
        emb.append([float(x) for x in ss[1:]])
        if i % 10000 == 0 :
            logging.info('{} lines loaded'.format(i))
elif ext in (".pickle", ".pkl") :
    vocab, emb = pickle.load(open(args.target, 'rb'))
else :
    raise ValueError('Unsupported target vector file extent: {}'.format(args.target))
emb = np.array(emb)

if False :
    scnt = Counter()
    for w in vocab :
        for s in set(_substrings(w)) :
            scnt[s] += 1
    freqsubs = set(s for s, c in scnt.items() if c  > 5)

def MSE(pred, target) :
    return sum((pred - target) ** 2 / 2) / len(target)
def MSE_backward(pred, target) :
    return (pred - target) / len(target)

model = BoS(embedding_dim=len(emb[0]),
    lmin=args.lmin, lmax=args.lmax,
    hashed=args.hashed, hash_range=args.hash_range,
    cached=args.cached,
)
h = []
start_time = time()
for i_epoch in range(args.epochs) :
    lr = args.lr / (1 + i_epoch) ** 0.5
    logging.info('epoch {:>2} / {} | lr {:.5f}'.format(1 + i_epoch, args.epochs, lr))
    epoch_start_time = time()
    for i_inst, wi in zip(count(1), np.random.choice(len(vocab), len(vocab), replace=False)) :
        w = vocab[wi]
        e = model.embed(w)
        tar = emb[wi]
        g = MSE_backward(e, tar)

        if i_inst % 20 == 0 :
            loss = MSE(e, tar)
            h.append(loss)
        if i_inst % 10000 == 0 :
            logging.info('processed {}/{} | loss {:.5f}'.format(i_inst, len(vocab), np.average(h)))
            h = []

        d = - lr * g
        model.step(w, d)
    now_time = time()
    logging.info('epoch {i_epoch:>2} / {n_epoch} | time {epoch_time:.2f} / {training_time:.2f}'.format(
        i_epoch = 1 + i_epoch, n_epoch = args.epochs,
        epoch_time = now_time - epoch_start_time,
        training_time = now_time - start_time,
    ))

logging.info('saving model...')
model.dump(os.path.join(save_dir, 'model.bos'))
