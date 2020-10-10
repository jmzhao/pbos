import os
import subprocess as sp

from utils import dotdict

datasets_dir = os.path.dirname(os.path.realpath(__file__))

_ws_datasets = {
    # default word similarity datasets for English
    None: {
        "wordsim353": {
            "url": "https://leviants.com/wp-content/uploads/2020/01/wordsim353.zip",
            "raw_txt_rel_path": "combined.tab",
            "skip_lines": 1,
        },
        "rw": {
            "url": "https://nlp.stanford.edu/~lmthang/morphoNLM/rw.zip",
            "raw_txt_rel_path": "rw/rw.txt",
        },
        "card660": {
            "url": "https://pilehvar.github.io/card-660/dataset.tsv",
            "no_zip": True,
            "raw_txt_rel_path": "dataset.tsv",
        },
    },
    # multilingual word similarity datasets
    "en": {},
    "it": {},
    "ru": {},
    "de": {},
}

for lang, full_name in [('en', 'english'), ('it', 'italian'), ('ru', 'russian'), ('de', 'german')]:
    for ws_suffix in ("-rel", "-sim", ""):
        _ws_datasets[lang][f"ws353-{lang}{ws_suffix}"] = {
            "url": f"https://raw.githubusercontent.com/iraleviant/eval-multilingual-simlex/master/evaluation/ws-353/wordsim353-{full_name}{ws_suffix}.txt",
            "raw_txt_rel_path": f"wordsim353-{full_name}{ws_suffix}.txt",
            "no_zip": True,
            "skip_lines": 1,
        }

    _ws_datasets[lang][f"simlex999-{lang}"] = {
        "url": f"https://raw.githubusercontent.com/nmrksic/eval-multilingual-simlex/master/evaluation/simlex-{full_name}.txt",
        "raw_txt_rel_path": f"simlex-{full_name}.txt",
        "no_zip": True,
        "skip_lines": 1,
    }


def get_ws_dataset_names(lang=None):
    return list(_ws_datasets[lang])


def _get_ws_dataset_info(name):
    for datasets in _ws_datasets.values():
        if name in datasets:
            return datasets[name]

    raise NotImplementedError


def prepare_ws_dataset_paths(name):
    binfo = _get_ws_dataset_info(name)

    raw_txt_path = f"{datasets_dir}/{name}/{binfo['raw_txt_rel_path']}"
    txt_path = f"{datasets_dir}/{name}/{name}.txt"
    query_path = f"{datasets_dir}/{name}/queries.txt"

    if not os.path.exists(raw_txt_path):
        sp.call(
            f"""
            wget -c {binfo['url']} -P {datasets_dir}/{name}
        """.split()
        )
        if not binfo.get("no_zip"):
            sp.call(
                f"""
                unzip {datasets_dir}/{name}/{name}.zip -d {datasets_dir}/{name}
            """.split()
            )

    if not os.path.exists(txt_path):
        with open(raw_txt_path) as f, open(txt_path, "w") as fout:
            for i, line in enumerate(f):
                # discard head lines
                if i < binfo.get("skip_lines", 0):
                    continue
                # NOTE: in `fastText/eval.py`, golden words get lowercased anyways,
                # but predicted words remain as they are.
                print(line, end="", file=fout)

    if not os.path.exists(query_path):
        words = set()
        with open(txt_path) as f:
            for line in f:
                w1, w2 = line.split()[:2]
                words.add(w1)
                words.add(w2)
        with open(query_path, "w") as fout:
            for w in words:
                print(w, file=fout)

    return dotdict(
        txt_path=txt_path,
        query_path=query_path,
    )


def prepare_ws_combined_query_path(lang=None):
    """
    Prepare the combined query path for word similarity datasets dataset
    """

    combined_query_path = f"{datasets_dir}/combined_query_{lang}.txt"

    if not os.path.exists(combined_query_path):
        all_words = set()
        for bname in get_ws_dataset_names(lang):
            query_path = prepare_ws_dataset_paths(bname).query_path
            with open(query_path) as fin:
                for line in fin:
                    all_words.add(line.strip())
                    all_words.add(line.strip().lower())
        with open(combined_query_path, 'w') as fout:
            for w in all_words:
                print(w, file=fout)

    return combined_query_path


if __name__ == '__main__':
    for lang in [None, "en", "it", "ru", "de"]:
        for bname in get_ws_dataset_names(lang):
            prepare_ws_dataset_paths(bname)
        prepare_ws_combined_query_path(lang)
