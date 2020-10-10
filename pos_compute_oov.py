from datasets import prepare_target_vector_paths, polyglot_languages, prepare_ud_paths
from load import load_embedding

for language in polyglot_languages:
    polyglot_path = prepare_target_vector_paths(language).pkl_emb_path
    polyglot_vocab, _ = load_embedding(polyglot_path)
    polyglot_vocab = set(polyglot_vocab)

    _, ud_vocab_path = prepare_ud_paths(language)
    with open(ud_vocab_path) as f:
        ud_vocab = [w.strip() for w in f]

    oov = sum(w not in polyglot_vocab for w in ud_vocab) / len(ud_vocab)
    print(language, oov)
