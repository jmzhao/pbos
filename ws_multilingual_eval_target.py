"""
Used to generate target vector statistics
"""
from datasets import prepare_target_vector_paths, prepare_ws_dataset_paths, get_ws_dataset_names
from ws_eval import eval_ws

for lang in ("de", "en", "it", "ru",):
    target_vector_path = prepare_target_vector_paths(f"wiki2vec-{lang}").txt_emb_path
    for dataset in get_ws_dataset_names(lang):
        data_path = prepare_ws_dataset_paths(dataset).txt_path
        for oov_handling in ("drop", "zero"):
            print(eval_ws(target_vector_path, data_path, lower=True, oov_handling=oov_handling))