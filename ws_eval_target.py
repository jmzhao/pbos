"""
Used to generate target vector statistics
"""
from datasets import prepare_target_vector_paths
from datasets.ws_bench import BENCHS, prepare_bench_paths, get_all_bnames_for_lang
from ws_eval import eval_ws

for lang in ("en", "it", "ru", "de"):
    target_vector_path = prepare_target_vector_paths(f"wiki2vec-{lang}").txt_emb_path
    for dataset in get_all_bnames_for_lang(lang):
        data_path = prepare_bench_paths(dataset).txt_path
        for oov_handling in ("drop", "zero"):
            print(eval_ws(target_vector_path, data_path, lower=True, oov_handling=oov_handling))
