import os

from datasets.affix import prepare_affix_paths
from datasets.glove import prepare_glove_paths
from datasets.google import prepare_google_paths
from datasets.polyglot_emb import prepare_polyglot_emb_paths, prepare_polyglot_clean_en_paths
from datasets.ws_bench import BENCHS, prepare_bench_paths

dir_path = os.path.dirname(os.path.realpath(__file__))


def prepare_combined_query_path(
    *,
    dir_path=dir_path,
):
    """
    Prepare the combined query path for word similarity datasets & affix dataset
    """

    combined_query_path = f"{dir_path}/combined_query.txt"

    if not os.path.exists(combined_query_path):
        all_words = set()
        for bname in BENCHS:
            bench_paths = prepare_bench_paths(bname)
            with open(bench_paths.query_path) as fin:
                for line in fin:
                    all_words.add(line.strip())
                    all_words.add(line.strip().lower())
        affix_paths = prepare_affix_paths()
        with open(affix_paths.queries_path) as fin:
            for line in fin:
                all_words.add(line.strip())
                all_words.add(line.strip().lower())
        with open(combined_query_path, 'w') as fout:
            for w in all_words:
                print(w, file=fout)

    return combined_query_path


target_vector_names = ("google", "polyglot", "glove")


def prepare_en_target_vector_paths(target_vector_name):
    if target_vector_name.lower() == "google":
        return prepare_google_paths()
    if target_vector_name.lower() == "polyglot":
        return prepare_polyglot_clean_en_paths()
    if target_vector_name.lower() == "glove":
        return prepare_glove_paths()
    raise NotImplementedError
