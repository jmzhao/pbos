from pathlib import Path

from datasets import prepare_target_vector_paths
from datasets.ws_bench import prepare_bench_paths
from pbos_pred import predict
from ws_eval import eval_ws


def evaluate(target_vector_name, pred_path, eval_result_path):
    with open(eval_result_path, "w") as fout:
        bname = f"simlex999-{target_vector_name}"
        bench_path = prepare_bench_paths(bname).txt_path
        for lower in (True, False):
            print(eval_ws(pred_path, bench_path, lower=lower, oov_handling='zero'), file=fout)


def exp_pbos(target_vector_name, model_type, query_path, eval_result_path):
    pred_path = f"results/ws_multi/{target_vector_name}/{model_type}/emb.txt"
    predict(
        model=f"results/pos/{target_vector_name}/{model_type}/model.pbos",
        queries=query_path,
        save=pred_path,
        word_boundary=(model_type == "bos"),
    )
    evaluate(target_vector_name, pred_path, eval_result_path)


def exp_sasaki(target_vector_name, query_path, eval_result_path):
    from sasaki_utils import inference, get_info_from_result_path
    model_info = get_info_from_result_path(Path(f"results/pos/{target_vector_name}/sasaki/sep_kvq"))
    result_emb_path = inference(model_info, query_path)

    evaluate(target_vector_name, result_emb_path, eval_result_path)


def main():
    # model_types = ("pbos", "bos", "sasaki")
    model_types = ("sasaki",)
    target_vector_names = ("it", "ru")

    for target_vector_name in target_vector_names:
        prepare_target_vector_paths(target_vector_name)
        for model_type in model_types:
            eval_result_path = f"results/ws_multi/{target_vector_name}/{model_type}/result.txt"
            query_path = f"datasets/ws_bench/simlex999-{target_vector_name}/queries.txt"

            if model_type == "sasaki":
                exp_sasaki(target_vector_name, query_path, eval_result_path)
            else:
                exp_pbos(target_vector_name, model_type, query_path, eval_result_path)


if __name__ == '__main__':
    main()
