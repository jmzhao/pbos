"""
A simple script to gather the result for POS
"""

from pathlib import Path

pos_result_dir = Path("results") / "polyglot"


def get_acc(lang, model_type):
    out_path = lang / model_type / "ud.out"
    with open(out_path, 'r') as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
        _, acc = last_line.split(":")
        return acc.strip()


model_types = ("bos", "pbos", "pbosn")
print("", *model_types, sep="\t")
for lang in sorted(pos_result_dir.iterdir()):
    print(lang.name, *(get_acc(lang, m) for m in model_types), sep="\t")
