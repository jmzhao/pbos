"""
Script used to plot C vs score
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', help="path to the results directory", default="results/pos_reg_search")
args = parser.parse_args()

xs = []
ys = []

for path in Path(args.results_dir).iterdir():
    with open(path, 'r') as f:
        line = f.readline()
        if len(line) == 0:
            continue
        _, score_str = line.split(":")
        xs.append(float(path.name))
        ys.append(float(score_str.strip()))

plt.plot(xs, ys, 'o')
plt.xscale('log', basex=10)
plt.show()

for x, y in sorted(zip(xs, ys)):
    print(f"{x}: {y:.6f}")
