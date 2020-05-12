import subprocess as sp
import multiprocessing as mp
import os

from pathlib import Path

from datasets.affix import prepare_affix_paths

queries_path = prepare_affix_paths().queries_path


def pred(model_type, result_path, ref_vec_name):
    if model_type in ('bos', 'pbos'):
        model_path = Path("results") / f"ws_{ref_vec_name}_{model_type}" / "model.pbos"
        result_emb_path = result_path / "emb.txt"
        sp.call(f"""
              python pbos_pred.py \
                --queries {queries_path} \
                --save {result_emb_path} \
                --model {model_path} \
          """.split())

        return result_emb_path

    if model_type == "sasaki":
        from sasaki_utils import inference as inference_sasaki, get_info_from_result_path
        model_info = get_info_from_result_path(Path("results") / f"ws_{ref_vec_name}_{model_type}" / "sep_kvq")

        epoch = model_info["epoch"]
        result_path = model_info["result_path"]
        model_path = model_info["model_path"]
        codecs_path = model_info["codecs_path"]

        inference_sasaki(model_path, codecs_path, queries_path)
        result_emb_path = result_path / f"inference_embedding_epoch{epoch}" / "embedding.txt"
        return result_emb_path


def exp(model_type, ref_vec_name):
    result_path = Path("results") / f"affix_{ref_vec_name}_{model_type}"
    os.makedirs(result_path, exist_ok=True)

    result_emb_path = pred(model_type, result_path, ref_vec_name)

    with open(result_path / "result.txt", "w") as fout:
        sp.call(f"""
                python affix_eval.py \
                  --embeddings {result_emb_path} \
            """.split(), stdout=fout)


with mp.Pool() as pool:
    results = [
        pool.apply_async(exp, (model_type, ref_vec_name,))
        for model_type in ['sasaki', 'bos', 'pbos']
        for ref_vec_name in ["google", "polyglot"]
    ]

    for r in results:
        r.get()
