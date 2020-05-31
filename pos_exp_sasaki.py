import multiprocessing as mp
import subprocess as sp
from pathlib import Path

from datasets.polyglot_emb import prepare_polyglot_emb_paths, languages
from datasets.polyglot_freq import prepare_polyglot_freq_paths
from datasets.ud import prepare_ud_paths
from sasaki_utils import inference, train, prepare_codecs_path


def exp(language):
    result_path = Path("results") / "pos" / language / "sasaki"

    emb_path = prepare_polyglot_emb_paths(language).w2v_emb_path
    freq_path = prepare_polyglot_freq_paths(language).raw_count_path
    codecs_path = prepare_codecs_path(emb_path, result_path)
    ud_data_path, ud_vocab_path = prepare_ud_paths(language)

    model_info = train(
        emb_path,
        result_path,
        freq_path=freq_path,
        codecs_path=codecs_path,
        epoch=300,
        H=40_000,
        F=500_000
    )

    result_emb_path = inference(model_info, ud_vocab_path)

    with open(result_path / "ud.out", "w") as fout, open(result_path / "ud.log", "w") as ferr:
        cmd = f"""
            python pos_eval.py \
            --dataset {ud_data_path} \
            --embeddings {result_emb_path} \
            --C {70} \
        """.split()
        sp.call(cmd, stdout=fout, stderr=ferr)


if __name__ == "__main__":
    with mp.Pool() as pool:
        for lang in languages:
            pool.apply_async(exp, (lang,))

        pool.close()
        pool.join()
