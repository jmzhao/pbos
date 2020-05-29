"""
A simple script to gather the result for POS
"""

from pathlib import Path
from pos_exp import model_types

pos_result_dir = Path("results") / "pos"


def get_acc(lang, model_type):
    out_path = lang / model_type / "ud.out"
    if not out_path.exists():
        return -1
    with open(out_path, 'r') as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
        _, acc = last_line.split(":")
        return acc.strip()


if __name__ == "__main__":
    print("lang", *model_types, sep="\t")
    for lang in sorted(pos_result_dir.iterdir()):
        print(lang.name, *(get_acc(lang, m) for m in model_types), sep="\t")
