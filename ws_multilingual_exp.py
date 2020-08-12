"""
Evaluate multilingual word sim on POS model
"""

from pathlib import Path

from datasets.ws_bench import prepare_bench_paths, get_all_bnames_for_lang
from pbos_pred import predict
from ws_eval import eval_ws


def exp(model_type, lang, bname):
    query_path = prepare_bench_paths(bname).query_path

    if model_type == "sasaki":
        from sasaki_utils import inference, get_info_from_result_path
        model_info = get_info_from_result_path(Path(f"results/pos/{lang}/sasaki/sep_kvq"))
        result_emb_path = inference(model_info, query_path)
    else:
        result_emb_path = f"results/ws_multi/{bname}/{model_type}/emb.txt"
        predict(
            model=f"results/pos/{lang}/{model_type}/model.pbos",
            queries=query_path,
            save=result_emb_path,
            word_boundary=(model_type == "bos"),
        )

    bench_path = prepare_bench_paths(bname).txt_path
    with open("results/ws_multi/result.txt", "a") as fout:
        for lower in (True, False):
            print(model_type.ljust(10), eval_ws(result_emb_path, bench_path, lower=lower, oov_handling='zero'), file=fout)


def main():
    model_types = ("pbos", "bos", "sasaki")

    for lang in ("en", "it", "ru"):
        for bname in get_all_bnames_for_lang(lang):
            for model_type in model_types:
                exp(model_type, lang, bname)


if __name__ == '__main__':
    main()
