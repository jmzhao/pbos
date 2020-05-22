import logging
from pathlib import Path
import multiprocessing as mp

from datasets.google import prepare_google_paths
from datasets.polyglot_emb import prepare_polyglot_emb_paths
from datasets import prepare_combined_query_path
from sasaki_utils import inference, get_latest_in_dir, prepare_codecs_path, train
from utils import dotdict
from ws_affix_exp_pbos import evaluate_ws_affix

epoch = 300

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def exp(ref_vec_path, ref_vec_name):
    logger.info("Starting...")
    result_path = Path(".") / "results" / "ws_affix" / f"{ref_vec_name}_sasaki"
    codecs_path = prepare_codecs_path(ref_vec_path, result_path)

    logger.info("Training...")
    train(
        ref_vec_path,
        result_path,
        codecs_path=codecs_path,
        H=40_000,
        F=500_000,
        epoch=epoch,
    )

    logger.info("Evaluating...")
    result_path = get_latest_in_dir(result_path / "sep_kvq")
    model_path = result_path / f"model_epoch_{epoch}"
    combined_query_path = prepare_combined_query_path()
    inference(model_path, codecs_path, combined_query_path)
    result_emb_path = result_path / f"inference_embedding_epoch{epoch}" / "embedding.txt"

    evaluate_ws_affix(dotdict(
        eval_result_path=result_path / "result.txt",
        pred_path=result_emb_path
    ))


with mp.Pool() as pool:
    ref_vec = {
        "polyglot": prepare_polyglot_emb_paths("en").w2v_path,
        "google": prepare_google_paths().w2v_path,
    }

    results = [pool.apply_async(exp, (ref_vec_path, ref_vec_name,))
               for ref_vec_name, ref_vec_path in ref_vec.items()]

    for r in results:
        r.get()
