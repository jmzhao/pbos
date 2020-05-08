import os
import subprocess as sp
from utils import dotdict
import json



def train(
        emb_path,
        result_path,
        freq_path=None,
        codecs_path=None,
        epoch=20,
        H=100000,
        F=1000000,
        use_hash=True,
):
    with open(emb_path) as f:
        _,  embed_dim = f.readline().strip().split()

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


def get_latest_in_dir(dir_path):
    return max(dir_path.iterdir(), key=lambda x: x.stat().st_mtime)


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


def get_info_from_result_path(result_path):
    result_path = get_latest_in_dir(result_path)
    settings_path = result_path / "settings.json"
    data = json.load(open(settings_path, "r"))
    epoch = data['epoch']
    data["model_path"] = result_path / f"model_epoch_{epoch}"
    data["result_path"] = result_path
    return data
