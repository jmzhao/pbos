"""
Script used to tune regression C for affix
"""
import argparse
import subprocess as sp


def main(args):
    for C in [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20]:
        with open(f"results/affix/affix_C={C}", "w+") as f:
            sp.call(f"python affix_eval.py --embeddings {args.embeddings} --C {C}".split(), stdout=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', help="path to word embeddings")
    args = parser.parse_args()
    main(args)
