import logging
import multiprocessing as mp
from pathlib import Path

from datasets import prepare_combined_query_path, prepare_en_target_vector_paths
from sasaki_utils import inference, prepare_codecs_path, train
from utils import dotdict
from ws_affix_exp_pbos import evaluate_ws_affix

logger = logging.getLogger(__name__)


def exp(ref_vec_name):
    result_path = Path("results") / "ws_affix" / f"{ref_vec_name}_sasaki"
    ref_vec_path = prepare_en_target_vector_paths(ref_vec_name).w2v_emb_path
    codecs_path = prepare_codecs_path(ref_vec_path, result_path)

    log_file = open(result_path / "log.txt", "w+")
    logging.basicConfig(level=logging.DEBUG, stream=log_file)

    logger.info("Training...")
    model_info = train(
        ref_vec_path,
        result_path,
        codecs_path=codecs_path,
        H=40_000,
        F=500_000,
        epoch=300,
    )

    logger.info("Inferencing...")
    combined_query_path = prepare_combined_query_path()
    result_emb_path = inference(model_info, combined_query_path)

    logger.info("Evaluating...")
    evaluate_ws_affix(dotdict(
        eval_result_path=result_path / "result.txt",
        pred_path=result_emb_path
    ))


if __name__ == '__main__':
    with mp.Pool() as pool:
        target_vector_names = ("polyglot", "google")

        results = [
            pool.apply_async(exp, (ref_vec_name,))
            for ref_vec_name in target_vector_names
        ]

        for r in results:
            r.get()
