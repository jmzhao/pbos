import multiprocessing as mp
import os
import pickle
import subprocess as sp
from pathlib import Path

from datasets.google_news import (prepare_google_news_codecs_path,
                                  prepare_google_news_paths)
from datasets.polyglot_embeddings import (get_polyglot_codecs_path,
                                          get_polyglot_embeddings_path,
                                          languages)
from datasets.polyglot_freq import get_polyglot_frequency_path
from sasaki_utils import evaluate_ws, inference, train, get_latest, evaluate_pos
from datasets.universal_dependencies import get_universal_dependencies_path

def exp(lang):
    epoch = 20

    emb_path = get_polyglot_embeddings_path(lang).w2v_path
    freq_path = get_polyglot_frequency_path(lang).raw_count_path
    codecs_path = get_polyglot_codecs_path(lang)
    ud_data_path, ud_vocab_path = get_universal_dependencies_path(lang)

    result_path = Path(".") / "results" / "compact_reconstruction" / "polyglot_KVQ_F" / lang
    train(emb_path, result_path, freq_path, codecs_path, epoch=epoch)

     
    result_path  = get_latest(result_path / "sep_kvq")
    model_path = result_path / f"model_epoch_{epoch}"
    inference(model_path, codecs_path, ud_vocab_path)

    ud_vocab_embedding_path = result_path / f"inference_embedding_epoch{epoch}" / "embedding.txt"
    score = evaluate_pos(ud_data_path, ud_vocab_embedding_path)

    print(f"score for {lang} = {score}")

if __name__ == "__main__":
    with mp.Pool() as pool:
        for lang in languages:
            pool.apply_async(exp, (lang, ))

        pool.close()
        pool.join()