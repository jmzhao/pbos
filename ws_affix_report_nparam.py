"""
A simple script to print the number of parameters for all models in a directory.
"""

import argparse
import pickle
from pathlib import Path


def count_nparam(result_dir):
    model_path = result_dir / "model.pkl"
    if not model_path.exists():
        return -1
    with open(model_path, "rb") as f:
        model = pickle.load(f, encoding='bytes')
        nsubwords =len(model)
        embed_dim = len(next(iter(model.values())))
        return nsubwords * embed_dim


def main(results_dir):
    for result_dir in sorted(Path(results_dir).iterdir()):
        nparam = count_nparam(result_dir)
        print(result_dir.name, nparam)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results_dir',
        help="path to the results directory",
        default="results/ws_affix"
    )
    args = parser.parse_args()
    main(args.results_dir)
