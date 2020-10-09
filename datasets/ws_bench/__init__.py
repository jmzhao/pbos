import os
import subprocess as sp

from utils import dotdict

BENCHS = {
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
}

multi_bench_languages = {
    "en": "english",
    "it": "italian",
    "ru": "russian",
    "de": "german"
}

MULTI_BENCHS = {}

for lang, full_name in multi_bench_languages.items():
    for ws_suffix in ("-rel", "-sim", ""):
        MULTI_BENCHS[f"ws353-{lang}{ws_suffix}"] = {
            "url": f"https://raw.githubusercontent.com/iraleviant/eval-multilingual-simlex/master/evaluation/ws-353/wordsim353-{full_name}{ws_suffix}.txt",
            "raw_txt_rel_path": f"wordsim353-{full_name}{ws_suffix}.txt",
            "no_zip": True,
            "skip_lines": 1,
        }

    MULTI_BENCHS[f"simlex999-{lang}"] = {
        "url": f"https://raw.githubusercontent.com/nmrksic/eval-multilingual-simlex/master/evaluation/simlex-{full_name}.txt",
        "raw_txt_rel_path": f"simlex-{full_name}.txt",
        "no_zip": True,
        "skip_lines": 1,
    }

datasets_dir = os.path.dirname(os.path.realpath(__file__))


def get_all_bnames_for_lang(lang):
    return [f"ws353-{lang}", f"ws353-{lang}-rel", f"ws353-{lang}-sim", f"simlex999-{lang}"]


def prepare_combined_query_path_for_lang(lang):
    combined_query_path = f"{datasets_dir}/combined_query_{lang}.txt"

    if not os.path.exists(combined_query_path):
        all_words = set()
        for bname in get_all_bnames_for_lang(lang):
            bench_paths = prepare_bench_paths(bname)
            with open(bench_paths.query_path) as fin:
                for line in fin:
                    all_words.add(line.strip())
                    all_words.add(line.strip().lower())
        with open(combined_query_path, 'w') as fout:
            for w in all_words:
                print(w, file=fout)

    return combined_query_path


def prepare_combined_query_path():
    """
    Prepare the combined query path for word similarity datasets dataset
    """

    combined_query_path = f"{datasets_dir}/combined_query.txt"

    if not os.path.exists(combined_query_path):
        all_words = set()
        for bname in BENCHS:
            bench_paths = prepare_bench_paths(bname)
            with open(bench_paths.query_path) as fin:
                for line in fin:
                    all_words.add(line.strip())
                    all_words.add(line.strip().lower())
        with open(combined_query_path, 'w') as fout:
            for w in all_words:
                print(w, file=fout)

    return combined_query_path


def prepare_bench_paths(name):
    binfo = BENCHS[name] if name in BENCHS else MULTI_BENCHS[name]

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


if __name__ == '__main__':
    for bname in BENCHS:
        prepare_bench_paths(bname)
    for lang in multi_bench_languages:
        prepare_combined_query_path_for_lang(lang)
