from datasets.glove import prepare_glove_paths
from datasets.ws_bench import BENCHS, prepare_bench_paths
from ws_eval import eval_ws

from datasets.google import prepare_google_paths
from datasets.polyglot_emb import prepare_polyglot_emb_paths

models = {
    "EditSim": "EditSim",
    "ployglot": prepare_polyglot_emb_paths("en").txt_emb_path,
    "google": prepare_google_paths().txt_emb_path,
    "glove": prepare_glove_paths().txt_emb_path,
}

for model_name, model_path in models.items():
    for dataset in BENCHS:
        data_path = prepare_bench_paths(dataset).txt_path
        result = eval_ws(model_path, data_path, lower=False, oov_handling="zero")
        print(model_name, result)
