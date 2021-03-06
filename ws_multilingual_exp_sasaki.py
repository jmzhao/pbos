import logging
import multiprocessing as mp
from pathlib import Path

from datasets import prepare_target_vector_paths, prepare_ws_combined_query_path
from sasaki_utils import inference, prepare_codecs_path, train, get_info_from_result_path
from utils import dotdict
from ws_multilingual_exp_pbos import evaluate

logger = logging.getLogger(__name__)


def exp(ref_vec_name):
    result_path = Path("results") / "ws_multi" / f"{ref_vec_name}_sasaki"
    ref_vec_path = prepare_target_vector_paths(f"wiki2vec-{ref_vec_name}").w2v_emb_path
    codecs_path = prepare_codecs_path(ref_vec_path, result_path)

    log_file = open(result_path / "log.txt", "w+")
    logging.basicConfig(level=logging.DEBUG, stream=log_file)

    logger.info("Training...")
    train(
        ref_vec_path,
        result_path,
        codecs_path=codecs_path,
        H=40_000,
        F=500_000,
        epoch=300,
    )

    model_info = get_info_from_result_path(result_path / "sep_kvq")

    logger.info("Inferencing...")
    combined_query_path = prepare_ws_combined_query_path(ref_vec_name)
    result_emb_path = inference(model_info, combined_query_path)

    logger.info("Evaluating...")
    evaluate(dotdict(
        model_type="sasaki",
        eval_result_path=result_path / "result.txt",
        pred_path=result_emb_path,
        target_vector_name=ref_vec_name,
        results_dir=result_path,
    ))


if __name__ == '__main__':
    with mp.Pool() as pool:
        target_vector_names = ("en", "de", "it", "ru")

        results = [
            pool.apply_async(exp, (ref_vec_name,))
            for ref_vec_name in target_vector_names
        ]

        for r in results:
            r.get()
