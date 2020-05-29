from load import load_embedding

from datasets.polyglot_emb import prepare_polyglot_emb_paths, languages
from datasets.ud import prepare_ud_paths

for language in languages:
    polyglot_path = prepare_polyglot_emb_paths(language).pkl_path
    polyglot_vocab, _ = load_embedding(polyglot_path)
    polyglot_vocab = set(polyglot_vocab)

    _, ud_vocab_path = prepare_ud_paths(language)
    with open(ud_vocab_path) as f:
        ud_vocab = [w.strip() for w in f]

    inv = sum(w in polyglot_vocab for w in ud_vocab) / len(ud_vocab)
    print(language, inv)
