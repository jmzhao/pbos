"""
Script used to tune regression C for POS
"""
import argparse
import os
import subprocess as sp
import multiprocessing as mp
from itertools import product

C_interval = sorted(x * 10 ** b for x, b in product(range(1, 10), range(-1, 5)))


def evaluate(results_dir, embeddings, dataset, C):
    with open(f"{results_dir}/{C:.1f}", "w+") as f:
        sp.call(
            f"python pos_eval.py \
            --embeddings {embeddings} \
            --C {C} \
            --dataset {dataset} \
            ".split(),
            stdout=f
        )


def main(results_dir, embeddings, dataset):
    os.makedirs(results_dir, exist_ok=True)
    with mp.Pool() as pool:
        results = [
            pool.apply_async(evaluate, (results_dir, embeddings, dataset, C))
            for C in C_interval
        ]

        for r in results:
            r.get()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help="path to dataset")
    parser.add_argument(
        '--embeddings',
        required=True,
        help="path to word embeddings"
    )
    parser.add_argument(
        '--results_dir',
        help="path to the results directory",
        default="results/pos_reg_search"
    )
    args = parser.parse_args()
    main(args.results_dir, args.embeddings, args.dataset)
