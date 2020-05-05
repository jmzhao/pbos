import os
import pickle

from load import load_embedding

from datasets.polyglot_embeddings import get_polyglot_embeddings_path, languages
from datasets.universal_dependencies import get_universal_dependencies_path

for language in languages:

    polyglot_path = get_polyglot_embeddings_path(language).pkl_path
    polyglot_vocab, _ = load_embedding(polyglot_path)
    polyglot_vocab = set(polyglot_vocab)
    
    _, ud_vocab_path =  get_universal_dependencies_path(language)
    ud_vocab = [l.strip() for l in open(ud_vocab_path)]
        
    inv = sum(w in polyglot_vocab for w in ud_vocab) / len(ud_vocab)
    print(language, inv)