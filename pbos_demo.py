#!/usr/bin/python3
import argparse
import os
import subprocess as sp
import sys

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_path', '-m',
    default= "./results/pbos/demo/model.pbos",
    help="The path to the model to be evaluated. "
    "If the model is not there, a new model will be trained and saved.")
args = parser.parse_args()

datasets_dir="./datasets"
results_dir, _ = os.path.split(args.model_path)

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

model_path = args.model_path
if not os.path.exists(model_path):
    sp.call(f'''
        python pbos_train.py \
          --target {pretrained_processed_path} \
          --word_list {wordlist_path} \
          --save {model_path} \
          --epochs 10 --lr 1 --lr_decay
    '''.split())

BENCHS = {
    'rw' : {
        'url' : "https://nlp.stanford.edu/~lmthang/morphoNLM/rw.zip",
        'raw_txt_rel_path' : 'rw/rw.txt',
    },
    'wordsim353' : {
        'url' : "https://leviants.com/wp-content/uploads/2020/01/wordsim353.zip",
        'raw_txt_rel_path' : 'combined.tab',
        'skip_lines' : 1,
    }
}

for bname, binfo in BENCHS.items():
    raw_txt_rel_path = binfo["raw_txt_rel_path"]
    raw_txt_path = f"{datasets_dir}/{bname}/{raw_txt_rel_path}"
    btxt_path = f"{datasets_dir}/{bname}/{bname}.txt"
    if not os.path.exists(raw_txt_path):
        sp.call(f'''
            wget -c {binfo['url']} -P {datasets_dir}
        '''.split())
        sp.call(f'''
            unzip {datasets_dir}/{bname}.zip -d {datasets_dir}/{bname}
        '''.split())
    if not os.path.exists(btxt_path):
        with open(raw_txt_path) as f, open(btxt_path, 'w') as fout:
            for i, line in enumerate(f):
                if i < binfo.get('skip_lines', 0):
                    continue
                print(line, end='', file=fout)
    bquery_path = f"{datasets_dir}/{bname}/queries.txt"
    if not os.path.exists(bquery_path):
        words = set()
        with open(btxt_path) as f:
            for line in f:
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
