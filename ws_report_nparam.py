"""
A simple script to print the number of parameters for all models in a directory.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


def count_nparam_sasaki(result_dir):
    from sasaki_utils import get_info_from_result_path

    model_info = get_info_from_result_path(result_dir / "sep_kvq")
    model_path = model_info["model_path"]
    model = np.load(model_path)

    nparam = 0
    for filename in model.files:
        # print(filename, model[filename].size)
        nparam += model[filename].size

    return nparam


def count_nparam_pbos(result_dir):
    model_path = result_dir / "model.pkl"
    if not model_path.exists():
        return -1
    with open(model_path, "rb") as f:
        model = pickle.load(f, encoding='bytes')
        nsubwords = len(model)
        embed_dim = len(next(iter(model.values())))
        return nsubwords * embed_dim


def main(results_dir):
    for result_dir in sorted(Path(results_dir).iterdir()):
        if "bos" in result_dir.name:
            nparam = count_nparam_pbos(result_dir)
        elif "sasaki" in result_dir.name:
            nparam = count_nparam_sasaki(result_dir)
        else:
            nparam = -1
        print(f"{result_dir.name:<20}{nparam:,}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results_dir', '-d',
        help="path to the results directory",
        default="results/ws"
    )
    args = parser.parse_args()
    main(args.results_dir)
