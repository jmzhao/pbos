import os
import subprocess as sp
import sys

for model_type, subword_prob_take_root in [
    ('bos', None),
    ('pbos', False),
    ('pbos', True)
]:
    results_dir = f"results/trials/{model_type}/subword_min_count=1"
    if model_type == 'pbos':
        results_dir = os.path.join(
            results_dir,
            "unigram_freq",
            "take_root" if subword_prob_take_root else "no_take_root",
        )
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "demo.log")
    model_path = os.path.join(results_dir, "model.pbos")
    cmd = f"python pbos_demo.py \
        --model_path {model_path} \
        --model_type {model_type} \
        --{'' if subword_prob_take_root else 'no_'}subword_prob_take_root \
    ".split()
    with sp.Popen(['/usr/bin/tee', '-a', log_path], stdin=sp.PIPE) as tee:
        sp.call(cmd, stderr = tee.stdin)
    # sp.call(cmd)
