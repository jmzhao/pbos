from .glove import prepare_glove_paths
from .google import prepare_google_paths
from .polyglot import prepare_polyglot_emb_paths, polyglot_languages
from .wiki2vec import prepare_wiki2vec_emb_paths


def prepare_target_vector_paths(target_vector_name):
    target_vector_name = target_vector_name.lower()

    if target_vector_name.startswith("polyglot-"):
        return prepare_polyglot_emb_paths(target_vector_name[-2:])
    if target_vector_name.startswith("wiki2vec-"):
        return prepare_wiki2vec_emb_paths(target_vector_name[-2:])
    if target_vector_name == "google":
        return prepare_google_paths()
    if target_vector_name == "polyglot":
        return prepare_polyglot_emb_paths("en")
    if target_vector_name == "glove":
        return prepare_glove_paths()
    if target_vector_name in polyglot_languages:
        return prepare_polyglot_emb_paths(target_vector_name)
    raise NotImplementedError
