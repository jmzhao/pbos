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
    use_hash=False,
):
    cmd = f"""
        python compact_reconstruction/src/train.py 
            --gpu 0 
            --ref_vec_path {emb_path} 
            --embed_dim {embed_dim} 
            --maxlen 200 
            --network_type 3 
            --limit_size 100000 
            --result_dir {result_path} 
            --unique_false 
        """

    if freq_path:
        cmd += f" --freq_path {freq_path} "

    if codecs_path:
        cmd += f" --codecs_path {codecs_path} "

    if use_hash:
        cmd += """
            --subword_type 4
            --multi_hash two
            --hashed_idx
            --bucket_size 100000 
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


def evaluate_word_similarity(data, model):
    sp.call(
        f"""
        python ws_eval.py \
          --data {data} \
          --model 
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


def train_google_news():
    emb_path = prepare_google_news_paths().w2v_path
    freq_path = prepare_google_news_paths().raw_count_path
    codecs_path = prepare_google_news_codecs_path()
    result_path = f"results/compact_reconstruction/google_news"
    train(
        emb_path,
        result_path,
        codecs_path=codecs_path,
        embed_dim=300,
        use_hash=False,
    )

train_google_news()
# train_demo()
# model_path = "./results/compact_reconstruction/google_news/sep_kvq/20200414_21_54_50/model_epoch_300"
# oov_word_path="/nobackup/prob-subword-embedding/datasets/rw/queries.lower.txt"
# inference(model_path=model_path, codecs_path=codecs_path, oov_word_path=oov_word_path)

# model  = "./results/compact_reconstruction/google_news/sep_kvq/20200414_21_54_50/inference_embedding_epoch300/embedding.txt"
# evaluate()


# train_all_polyglot_models()
