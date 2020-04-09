import os
import pickle
import subprocess as sp
import multiprocessing as mp

from datasets.polyglot_embeddings import get_polyglot_embeddings_path, get_polyglot_codecs_path, languages
from datasets.polyglot_freq import get_polyglot_frequency_path


def train(emb_path, freq_path, codecs_path, result_path, use_hash=False):
    cmd = f"""
        python compact_reconstruction/src/train.py 
            --gpu 0 
            --ref_vec_path {emb_path} 
            --freq_path {freq_path or ""} 
            --codecs_path {codecs_path or ""}
            --embed_dim 64 
            --maxlen 200 
            --network_type 3 
            --subword_type {4 if use_hash else 0}
            --limit_size 100000 
            --result_dir {result_path} 
            --unique_false 
        """
    
    if use_hash:
        cmd += """
            --multi_hash two
            --hashed_idx
            --bucket_size 100000 
        """
    
    sp.call(
        cmd.split(),
        env={**os.environ, "CUDA_PATH": "/usr/local/cuda-10.2"},
    )


with mp.Pool() as pool:
    for lang in languages[:1]:
        # input
        emb_path = get_polyglot_embeddings_path(lang).w2v_path
        freq_path = get_polyglot_frequency_path(lang).raw_count_path
        codecs_path = get_polyglot_codecs_path(lang)

        # output
        result_path = f"results/compact_reconstruction/polyglot_KVQ_F/{lang}"

        pool.apply_async(train, (emb_path, freq_path, codecs_path, result_path))

    pool.close()
    pool.join()
