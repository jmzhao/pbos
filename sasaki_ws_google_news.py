"""
Train Google News using the same parameter as table 6

To be specific, 
    Method = KVQ-FH 
    F = 0.50M
    H = 0.04M

"""

from pathlib import Path

from datasets.google_news import (
    prepare_google_news_paths,
    prepare_google_news_codecs_path,
)

from sasaki_utils import train, inference, evaluate_ws, get_latest


emb_path = prepare_google_news_paths().w2v_path
freq_path = prepare_google_news_paths().raw_count_path
codecs_path = prepare_google_news_codecs_path()
result_path = Path(".") / "results" / "compact_reconstruction" / "google_news"
# train(
#     emb_path,
#     result_path,
#     codecs_path=codecs_path,
#     H=40_000,
#     F=500_000,
#     embed_dim=300,
#     use_hash=True,
# )

result_path = get_latest(result_path / "sep_kvq") 
model_path = result_path / "model_epoch_300"

for dataset in ("card660", "wordsim353", "rw"):
# for dataset in ("rw", ):
    query_path = Path(".") / "datasets" / dataset / "queries.lower.txt"
    rw_data_path =  Path(".") / "datasets" / dataset / f"{dataset}.txt"
    result_emb_path = result_path / "inference_embedding_epoch300" / "embedding.txt"

    inference(model_path, codecs_path, query_path)
    evaluate_ws(rw_data_path, result_emb_path)