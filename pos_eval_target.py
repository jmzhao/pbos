"""
A simple script used to evaluate the raw PolyGlot vector for POS

One can redirect the starnard output of this file to get rid of the training log

python pos_exp_polyglot.py 2>train.log 1>eval.log
"""

import subprocess as sp

from datasets.polyglot_emb import prepare_polyglot_emb_paths, languages
from datasets.ud import prepare_ud_paths

for language_code in languages:
    ud_vocab_embedding_path = prepare_polyglot_emb_paths(language_code).pkl_emb_path
    ud_data_path, ud_vocab_path = prepare_ud_paths(language_code)

    cmd = f"""
        python pos_eval.py \
        --dataset {ud_data_path} \
        --embeddings {ud_vocab_embedding_path} \
    """.split()
    output = sp.check_output(cmd)
    print(f"{language_code}: {output.decode('utf-8')}")
