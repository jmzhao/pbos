import argparse
import logging
import os
import unicodedata

import gensim
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Prepare target word vectors")
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

# Load Google's pre-trained Word2Vec model.
logging.info("loading...")
model = gensim.models.KeyedVectors.load_word2vec_format(args.input, binary=True)

logging.info("normalizing...")
_words = []
for w in tqdm(model.vocab):
    aw = unicodedata.normalize("NFKD", w).encode("ASCII", "ignore")
    if 20 > len(aw) > 1 and not any(c in w for c in " _./\\#:,") and aw.islower():
        _words.append((aw, w))

embeddings = [model[w] for aw, w in _words]
words = [aw.decode() for aw, w in _words]

logging.info("saving...")
_, ext = os.path.splitext(args.output)
if ext in (".pickle", ".pkl"):
    with open(args.output, "wb") as fout:
        import pickle
        pickle.dump((words, embeddings), fout)
else:
    with open(args.output, "w") as fout:
        for w, e in tqdm(zip(words, embeddings)):
            print(w, *e, file=fout)

