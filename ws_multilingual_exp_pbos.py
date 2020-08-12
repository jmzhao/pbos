from datasets import prepare_target_vector_paths
from datasets.ws_bench import prepare_bench_paths
from pbos_pred import predict
from ws_eval import eval_ws

if __name__ == '__main__':

    model_types = ("pbos", "bos")
    target_vector_names = ("it", "ru")

    for target_vector_name in target_vector_names:
        prepare_target_vector_paths(target_vector_name)
        for model_type in model_types:
            pred_path = f"results/ws_multi/{target_vector_name}/{model_type}/emb.txt"
            eval_result_path = f"results/ws_multi/{target_vector_name}/{model_type}/result.txt"

            predict(
                model=f"results/pos/{target_vector_name}/{model_type}/model.pbos",
                queries=f"datasets/ws_bench/simlex999-{target_vector_name}/queries.txt",
                save=pred_path,
                word_boundary=(model_types == "bos"),
            )

            with open(eval_result_path, "w") as fout:
                bname = f"simlex999-{target_vector_name}"
                bench_path = prepare_bench_paths(bname).txt_path
                for lower in (True, False):
                    print(eval_ws(pred_path, bench_path, lower=lower, oov_handling='zero'), file=fout)
