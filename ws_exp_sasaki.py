"""
Train Google News using the same parameter as table 6

To be specific,
    Method = KVQ-FH
    F = 0.50M
    H = 0.04M

"""

from pathlib import Path

from datasets.google import prepare_google_paths, prepare_google_codecs_path
from datasets.ws_bench import BENCHS, prepare_bench_paths
from sasaki_utils import train, inference, get_latest_in_dir
from ws_eval import eval_ws

emb_path = prepare_google_paths().w2v_path
freq_path = prepare_google_paths().raw_count_path
codecs_path = prepare_google_codecs_path()
result_path = Path(".") / "results" / "sasaki" / "google_news"
train(
    emb_path,
    result_path,
    codecs_path=codecs_path,
    H=40_000,
    F=500_000,
    embed_dim=300,
    use_hash=True,
)

result_path = get_latest_in_dir(result_path / "sep_kvq")
model_path = result_path / "model_epoch_300"

for name in BENCHS:
    bench_paths = prepare_bench_paths(name)
    inference(model_path, codecs_path, bench_paths.bquery_lower_path)

    result_emb_path = result_path / "inference_embedding_epoch300" / "embedding.txt"
    eval_ws(bench_paths.btxt_path, result_emb_path)
