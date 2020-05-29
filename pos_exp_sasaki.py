import multiprocessing as mp
from pathlib import Path

from datasets.polyglot_emb import prepare_polyglot_emb_paths, languages
from datasets.polyglot_freq import prepare_polyglot_freq_paths
from datasets.ud import prepare_ud_paths
from sasaki_utils import inference, train, evaluate_pos, prepare_codecs_path


def exp(lang):
    result_path = Path("results") / "pos" / "sasaki" / lang

    emb_path = prepare_polyglot_emb_paths(lang).w2v_path
    freq_path = prepare_polyglot_freq_paths(lang).raw_count_path
    codecs_path = prepare_codecs_path(emb_path, result_path)
    ud_data_path, ud_vocab_path = prepare_ud_paths(lang)

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
    score = evaluate_pos(ud_data_path, result_emb_path, C=70)

    with open(result_path / "score.txt", "w") as f:
        print(f"score for {lang} = {score}", file=f)


if __name__ == "__main__":
    with mp.Pool() as pool:
        for lang in languages:
            pool.apply_async(exp, (lang,))

        pool.close()
        pool.join()
