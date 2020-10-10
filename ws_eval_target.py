import argparse

from datasets import prepare_target_vector_paths, get_ws_dataset_names, prepare_ws_dataset_paths
from ws_eval import eval_ws


def main(targets):
    for target in targets:
        target_vector_path = target if target == "EditSim" else prepare_target_vector_paths(target).txt_emb_path

        for dataset in get_ws_dataset_names():
            data_path = prepare_ws_dataset_paths(dataset).txt_path
            for oov_handling in ("drop", "zero"):
                result = eval_ws(target_vector_path, data_path, lower=True, oov_handling=oov_handling)
                print(target, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Show target vector statistics for the word similarity task")
    parser.add_argument(
        '--targets',
        '-t',
        nargs='+',
        choices=["EditSim", "polyglot", "google", "glove"],
        default=["polyglot", "google"]
    )

    args = parser.parse_args()

    main(args.targets)
