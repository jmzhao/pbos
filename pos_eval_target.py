"""
A simple script used to evaluate the raw PolyGlot vector for POS

One can redirect the starnard output of this file to get rid of the training log

python pos_exp_polyglot.py 2>train.log 1>eval.log
"""

import subprocess as sp

from datasets.polyglot_embeddings import get_polyglot_embeddings_path, languages
from datasets.universal_dependencies import get_universal_dependencies_path

for language_code in languages:
    ud_vocab_embedding_path = get_polyglot_embeddings_path(language_code).pkl_path
    ud_data_path, ud_vocab_path = get_universal_dependencies_path(language_code)

    cmd = f"""
        python pos_eval.py \
        --dataset {ud_data_path} \
        --embeddings {ud_vocab_embedding_path} \
    """.split()
    output = sp.check_output(cmd)
    print(f"{language_code}: {output.decode('utf-8')}")
