import os
import pickle
import subprocess as sp
import multiprocessing as mp

emb_dir = "datasets/polyglot_embeddings"
freq_dir = "datasets/polyglot_freq"
result_dir = "results/compact_reconstruction"

langs = [filename[:-4] for filename in os.listdir(emb_dir) if filename.endswith(".pkl")]


def convert_pkl_to_word2vec_format(lang):
    with open(f"{emb_dir}/{lang}.pkl", "rb") as fin:
        with open(f"{emb_dir}/{lang}.txt", "w") as fout:
            vocab, emb = pickle.load(fin, encoding="bytes")
            print(len(vocab), len(emb[0]), file=fout)
            for v, e in zip(vocab, emb):
                print(v, *e, file=fout)


def train(lang):
    convert_pkl_to_word2vec_format(lang)
    sp.call(
        f"""
        python compact_reconstruction/src/train.py \
            --gpu 0 \
            --ref_vec_path {emb_dir}/{lang}.txt \
            --freq_path {freq_dir}/{lang}.txt \
            --multi_hash two \
            --embed_dim 64 \
            --maxlen 200 \
            --network_type 3 \
            --subword_type 4 \
            --limit_size 1000000 \
            --bucket_size 100000 \
            --result_dir {result_dir}/{lang} \
            --hashed_idx \
            --unique_false
        """.split(),
        env={**os.environ, "CUDA_PATH": "/usr/local/cuda-10.2"},
    )


with mp.Pool() as pool:
    for lang in langs:
        pool.apply_async(train, (lang,))

    pool.close()
    pool.join()
