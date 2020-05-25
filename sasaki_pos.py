import multiprocessing as mp
from pathlib import Path

from datasets.polyglot_emb import get_polyglot_codecs_path, prepare_polyglot_emb_paths, languages
from datasets.polyglot_freq import prepare_polyglot_freq_paths
from datasets.ud import prepare_ud_paths
from sasaki_utils import inference, train, get_latest_in_dir, evaluate_pos


def exp(lang):
    epoch = 300

    emb_path = prepare_polyglot_emb_paths(lang).w2v_path
    freq_path = prepare_polyglot_freq_paths(lang).raw_count_path
    codecs_path = get_polyglot_codecs_path(lang)
    ud_data_path, ud_vocab_path = prepare_ud_paths(lang)

    result_path = Path(".") / "results" / "sasaki" / "polyglot_KVQ_F" / lang
    train(emb_path, result_path, freq_path, codecs_path, epoch=epoch, H=40_000, F=500_000)

    result_path = get_latest_in_dir(result_path / "sep_kvq")
    model_path = result_path / f"model_epoch_{epoch}"
    inference(model_path, codecs_path, ud_vocab_path)

    ud_vocab_embedding_path = result_path / f"inference_embedding_epoch{epoch}" / "embedding.txt"
    score = evaluate_pos(ud_data_path, ud_vocab_embedding_path)

    with open(result_path / "score.txt", "w") as f:
        print(f"score for {lang} = {score}", file=f)


if __name__ == "__main__":
    with mp.Pool() as pool:
        for lang in languages:
            pool.apply_async(exp, (lang,))

        pool.close()
        pool.join()
