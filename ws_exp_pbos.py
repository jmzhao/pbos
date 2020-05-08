import os
import subprocess as sp
import multiprocessing as mp


def exp(model_type, target_vectors, subword_prob_eps=None):
    results_dir = f"results/ws_{target_vectors}_{model_type}"
    if subword_prob_eps:
        results_dir += f"_eps={subword_prob_eps}"
    os.makedirs(results_dir, exist_ok=True)

    model_path = os.path.join(results_dir, "model.pbos")
    cmd = f"python pbos_demo.py \
            --model_path {model_path} \
            --model_type {model_type} \
            --target_vectors {target_vectors} \
            --epochs 50 \
        "
    if model_type == 'bos':
        cmd += "--subword_min_len 3 " \
               "--subword_max_len 6"
    else:
        cmd += f"--subword_prob_eps {subword_prob_eps}"

    log_path = os.path.join(results_dir, "demo.log")
    with sp.Popen(['/usr/bin/tee', '-a', log_path], stdin=sp.PIPE) as tee:
        sp.call(cmd.split(), stderr=tee.stdin)


with mp.Pool() as pool:
    results = []
    for target_vectors in ("polyglot", "google_news"):
        for subword_prob_eps in (0.01, 0.1, 0.2, 0.3, 0.5):
            results.append(pool.apply_async(exp, ('pbos', target_vectors, subword_prob_eps)))
        results.append(pool.apply_async(exp, ('bos', target_vectors)))

    for r in results:
        r.get()
