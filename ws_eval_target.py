"""
Used to generate target vector statistics
"""
from datasets import prepare_en_target_vector_paths
from datasets.ws_bench import BENCHS, prepare_bench_paths
from ws_eval import eval_ws

target_vector_names = ("EditSim", "polyglot", "google", "glove")
for target_vector_name in target_vector_names:
    if target_vector_name.lower() == "editsim":
        target_vector_path = "EditSim"
    else:
        target_vector_path = prepare_en_target_vector_paths(target_vector_name).txt_emb_path
    for dataset in BENCHS:
        data_path = prepare_bench_paths(dataset).txt_path
        for oov_handling in ("drop", "zero"):
            result = eval_ws(target_vector_path, data_path, lower=True, oov_handling=oov_handling)
            print(target_vector_name, result)
