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
    epoch=20,
    embed_dim=64,
    H=100000,
    F=1000000,
    use_hash=True,
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
            --epoch {epoch}
            --snapshot_interval 10
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

def evaluate_pos(ud_data_path, ud_vocab_embedding_path):
    cmd = f"""
        python pos_eval.py \
        --dataset {ud_data_path} \
        --embeddings {ud_vocab_embedding_path} \
    """.split()
    output = sp.check_output(cmd)
    return output.decode('utf-8')


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


def get_latest(dir_path):
    return max(dir_path.iterdir(), key=lambda x: x.stat().st_mtime)