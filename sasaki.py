import os
import pickle
import subprocess as sp
import multiprocessing as mp

from datasets.polyglot_embeddings import (
    get_polyglot_embeddings_path,
    get_polyglot_codecs_path,
    languages,
)
from datasets.polyglot_freq import get_polyglot_frequency_path
from datasets.google_news import (
    prepare_google_news_paths,
    prepare_google_news_codecs_path,
)


def train(
    emb_path,
    result_path,
    freq_path=None,
    codecs_path=None,
    embed_dim=64,
    H=100000,
    F=100000,
    use_hash=False,
):
    cmd = f"""
        python compact_reconstruction/src/train.py 
            --gpu 0 
            --ref_vec_path {emb_path} 
            --embed_dim {embed_dim} 
            --maxlen 200 
            --network_type 3 
            --limit_size {F} 
            --result_dir {result_path} 
            --unique_false 
        """

    if freq_path:
        cmd += f" --freq_path {freq_path} "

    if codecs_path:
        cmd += f" --codecs_path {codecs_path} "

    if use_hash:
        cmd += f"""
            --subword_type 4
            --multi_hash two
            --hashed_idx
            --bucket_size {H} 
        """
    else:
        cmd += " --subword_type 0 "

    sp.call(
        cmd.split(), env={**os.environ, "CUDA_PATH": "/usr/local/cuda-10.2"},
    )


def inference(model_path, codecs_path, oov_word_path):
    cmd = f"""
        python compact_reconstruction/src/inference.py 
            --gpu 0 
            --model_path {model_path} 
            --codecs_path {codecs_path} 
            --oov_word_path {oov_word_path} 
        """
    sp.call(cmd.split())


def evaluate_ws(data, model):
    sp.call(
        f"""
        python ws_eval.py \
          --data {data} \
          --model {model}
          --lower \
    """.split()
    )


def train_all_polyglot_models():
    with mp.Pool() as pool:
        for lang in languages[:1]:
            # input
            emb_path = get_polyglot_embeddings_path(lang).w2v_path
            freq_path = get_polyglot_frequency_path(lang).raw_count_path
            codecs_path = get_polyglot_codecs_path(lang)

            # output
            result_path = f"results/compact_reconstruction/polyglot_KVQ_F/{lang}"

            pool.apply_async(train, (emb_path, result_path, freq_path, codecs_path))

        pool.close()
        pool.join()


def train_demo():
    emb_path = "./compact_reconstruction/resources/crawl-300d-2M-subword.vec"
    freq_path = "./compact_reconstruction/resources/freq_count.crawl-300d-2M-subword.vec"
    codecs_path = "./compact_reconstruction/resources/ngram_dic.max30.min3"
    result_path = f"results/compact_reconstruction/example"
    train(
        emb_path,
        result_path,
        freq_path=freq_path,
        codecs_path=codecs_path,
        embed_dim=300,
        use_hash=False,
    )
