#!/usr/bin/python3
import os
import subprocess as sp
import sys

datasets_dir="./datasets"
results_dir="./results/pbos/demo"

os.makedirs(results_dir, exist_ok=True)
os.makedirs(datasets_dir, exist_ok=True)

pretrained="word2vec-google-news-300"
pretrained_processed_path = f"{datasets_dir}/{pretrained}/processed.txt"
if not os.path.exists(pretrained_processed_path):
    sp.call(f'''
    GENSIM_DATA_DIR="{datasets_dir}/gensim/" python prepare_target_embedding.py \
        --pretrained {pretrained} \
        --output {pretrained_processed_path}'''.split())
wordlist_path = f"{datasets_dir}/{pretrained}/word_list.txt"
if not os.path.exists(wordlist_path):
    with open(pretrained_processed_path) as f, open(wordlist_path, 'w') as fout:
        for line in f:
            print(line.split()[0], file=fout)

model_path = f"{results_dir}/model.pbos"
sp.call(f'''
    python pbos_train.py \
      --target {pretrained_processed_path} \
      --word_list {wordlist_path} \
      --save {model_path} \
      --epochs 10 --lr_decay
'''.split())

BENCHS = {
    'rw' : {
        'url' : "https://nlp.stanford.edu/~lmthang/morphoNLM/rw.zip",
        'btxt_rel_path' : 'rw/rw.txt',
    },
    'wordsim353' : {
        'url' : "https://leviants.com/wp-content/uploads/2020/01/wordsim353.zip",
        'btxt_rel_path' : 'combined.tab',
        'skip_lines' : 1,
    }
}

for bname, binfo in BENCHS.items():
    btxt_rel_path = binfo.get("btxt_rel_path", f"{bname}.txt")
    btxt_path = f"{datasets_dir}/{bname}/{btxt_rel_path}"
    if not os.path.exists(btxt_path):
        sp.call(f'''
            wget -c {binfo['url']} -P {datasets_dir}
        '''.split())
        sp.call(f'''
            unzip {datasets_dir}/{bname}.zip -d {datasets_dir}/{bname}
        '''.split())
    bquery_path = f"{datasets_dir}/{bname}/queries.txt"
    if not os.path.exists(bquery_path):
        words = set()
        with open(btxt_path) as f:
            for i, line in enumerate(f):
                if i < binfo.get('skip_lines', 0):
                    continue
                w1, w2 = line.split()[:2]
                words.add(w1)
                words.add(w2)
        with open(bquery_path, 'w') as fout:
            for w in words:
                print(w, file=fout)

    bpred_path = f"{results_dir}/{bname}_vectors.txt"
    sp.call(f'''
        python pbos_pred.py \
          --queries {bquery_path} \
          --save {bpred_path} \
          --model {model_path}
    '''.split())
    sp.call(f'''
        python ./fastText/eval.py \
          --data {btxt_path} \
          --model {bpred_path}
    '''.split())
