"""
Script used to tune regression C for affix
"""
import argparse
import os
import subprocess as sp
import multiprocessing as mp
from itertools import product


def evaluate(results_dir, embeddings, C):
    with open(f"{results_dir}/affix_C={C}", "w+") as f:
        sp.call(f"python affix_eval.py --embeddings {embeddings} --C {C}".split(), stdout=f)


def main(results_dir, embeddings):
    os.makedirs(results_dir, exist_ok=True)
    with mp.Pool() as pool:
        results = [
            pool.apply_async(evaluate, (results_dir, embeddings, C,))
            for C in sorted(x * 10 ** b for x, b in product(range(1, 10), range(-1, 4)))
        ]

        for r in results:
            r.get()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', help="path to word embeddings")
    parser.add_argument('--results_dir', help="path to the results directory", default="results/affix_search")
    args = parser.parse_args()
    main(args.results_dir, args.embeddings)
