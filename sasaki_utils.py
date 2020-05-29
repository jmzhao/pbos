import json
import os
import subprocess as sp
from pathlib import Path

from utils import dotdict


def train(
    emb_path,
    result_path,
    epoch,
    H,
    F,
    freq_path=None,
    codecs_path=None,
):
    """
    :return: a model info object
    """
    with open(emb_path) as f:
        _, embed_dim = f.readline().strip().split()

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
            --subword_type 4
            --multi_hash two
            --hashed_idx
            --bucket_size {H} 
        """

    if freq_path:
        cmd += f" --freq_path {freq_path} "

    if codecs_path:
        cmd += f" --codecs_path {codecs_path} "

    sp.call(
        cmd.split(), env={**os.environ, "CUDA_PATH": "/usr/local/cuda-10.2"},
    )

    return get_info_from_result_path(result_path / "sep_kvq")


def inference(model_info, query_path):
    """
    :return: resulting embedding path
    """
    result_path = Path(model_info["result_path"])
    model_path = model_info["model_path"]
    codecs_path = model_info["codecs_path"]
    epoch = model_info["epoch"]

    cmd = f"""
        python compact_reconstruction/src/inference.py 
            --gpu 0 
            --model_path {model_path} 
            --codecs_path {codecs_path} 
            --oov_word_path {query_path} 
        """
    sp.call(cmd.split())

    return result_path / f"inference_embedding_epoch{epoch}" / "embedding.txt"


def prepare_codecs_path(ref_vec_path, result_path, n_min=3, n_max=30):
    """
    See https://github.com/losyer/compact_reconstruction/tree/master/src/preprocess
    """
    os.makedirs(result_path, exist_ok=True)
    unsorted_codecs_path = os.path.join(result_path, f"codecs-min{n_min}max{n_max}.unsorted")
    sorted_codecs_path = os.path.join(result_path, f"codecs-min{n_min}max{n_max}.sorted")

    if not os.path.exists(unsorted_codecs_path):
        from sasaki_codecs import main as make_codecs

        make_codecs(dotdict(
            ref_vec_path=ref_vec_path,
            output=unsorted_codecs_path,
            n_max=n_max,
            n_min=n_min,
            test=False,
        ))

    if not os.path.exists(sorted_codecs_path):
        with open(sorted_codecs_path, 'w') as fout:
            sp.run(f"sort -k 2,2 -n -r {unsorted_codecs_path}".split(), stdout=fout)

    return sorted_codecs_path


def _get_latest_in_dir(dir_path):
    return max(dir_path.iterdir(), key=lambda x: x.stat().st_mtime)


def get_info_from_result_path(result_path):
    result_path = _get_latest_in_dir(result_path)
    settings_path = result_path / "settings.json"
    data = json.load(open(settings_path, "r"))
    epoch = data['epoch']
    data["model_path"] = result_path / f"model_epoch_{epoch}"
    data["result_path"] = result_path
    return data
