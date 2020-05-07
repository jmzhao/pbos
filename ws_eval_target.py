import subprocess as sp
from pathlib import Path
from ws_eval import eval_ws

from datasets.google_news import prepare_google_news_paths
from datasets.polyglot_embeddings import get_polyglot_embeddings_path

models = {
    "google news": prepare_google_news_paths().txt_emb_path,
    "ployglot": get_polyglot_embeddings_path("en").txt_emb_path,
}

for model_name, model_path in models.items():
    for dataset in ("wordsim353", "rw", "card660"):
        data_path = Path(".") / "datasets" / dataset / f"{dataset}.txt"
        # result = eval_ws(model_path, data_path, lower=True)
        result = eval_ws(model_path, data_path, lower=True, oov_handling="zero")
        print(model_name, result)
