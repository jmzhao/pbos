import unicodedata
import logging
logging.basicConfig(level=logging.INFO)

import gensim
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description='Prepare target word vectors')
# parser.add_argument('--input', required=True)
parser.add_argument('--pretrained', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()

# Load Google's pre-trained Word2Vec model.
logging.info('loading...')
# model = gensim.models.KeyedVectors.load_word2vec_format(args.input, binary=True, encoding="ISO-8859-1")
import gensim.downloader as api
model = api.load(args.pretrained)

logging.info('normalizing...')
_words = []
for w in tqdm(model.vocab) :
    aw = unicodedata.normalize('NFKD', w).encode('ASCII', 'ignore')
    if 20 > len(aw) > 1 and not any(c in w for c in ' _./') and aw.islower() :
        _words.append((aw, w))

embeddings = [model[w] for aw, w in _words]
words = [aw.decode() for aw, w in _words]

logging.info('saving...')
with open(args.output, 'w') as fout :
    for w, e in tqdm(zip(words, embeddings), total=len(_words)) :
        print(w, *e, file=fout)
