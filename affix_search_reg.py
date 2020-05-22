"""
Script used to tune regression C for affix
"""
import argparse
import os
import subprocess as sp


def main(args):
    results_dir = "results/affix"
    os.makedirs(results_dir, exist_ok=True)
    for C in [500, 600, 700, 800, 1000, 2000, 3000, 4000, 5000]:
        with open(f"{results_dir}/affix_C={C}", "w+") as f:
            sp.call(f"python affix_eval.py --embeddings {args.embeddings} --C {C}".split(), stdout=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', help="path to word embeddings")
    args = parser.parse_args()
    main(args)
