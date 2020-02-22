import logging
import multiprocessing as mp
import os
import subprocess as sp

from datasets.pre_trained.polyglot import get_polyglot_embeddings_path
from datasets.universal_dependencies import get_universal_dependencies_path
from datasets.word_freq.polyglot import get_polyglot_frequency_path

logging.basicConfig(level=logging.INFO)


def evaluate_pbos(language_code, mock_bos=False):
    print(f"evaluate_pbos({language_code}, mock_bos={mock_bos})")

    # Input files
    target_embeddings_path = get_polyglot_embeddings_path(language_code)
    word_frequency_path = get_polyglot_frequency_path(language_code)

    # Output/result files
    result_path = f'./results/polyglot/{language_code}/{"bos" if mock_bos else "pbos"}'
    os.makedirs(result_path, exist_ok=True)
    subword_embedding_model_path = result_path + "/model.pbos"
    training_log_path = subword_embedding_model_path + ".log"

    # train subword embedding model using target embeddings and word freq
    if not os.path.exists(subword_embedding_model_path):
        with open(training_log_path, "w+") as log:
            command = f"""
            python pbos_train.py \
              --target_vectors {target_embeddings_path} \
              --model_path {subword_embedding_model_path} \
              --word_list {word_frequency_path} \
              --word_list_has_freq \
              --boundary \
              --word_list_size 1000000 \
              --sub_min_len 3 
            """

            if mock_bos:
                command += " --mock_bos"

            sp.call(command.split(), stdout=log, stderr=log)

    ud_data_path, ud_vocab_path = get_universal_dependencies_path(language_code)
    ud_vocab_embedding_path = result_path + "/ud_vocab_embedding.txt"

    # predict embeddings for ud vocabs
    if not os.path.exists(ud_vocab_embedding_path):
        sp.call(
            f"""
            python pbos_pred.py \
            --queries {ud_vocab_path} \
            --save {ud_vocab_embedding_path} \
            --model {subword_embedding_model_path}
            """.split()
        )

    # train pos tagging
    ud_log_path = result_path + "/ud-log"
    sp.call(
        f"""
        python ./Mimick/model.py \
        --dataset {ud_data_path} \
        --word-embeddings {ud_vocab_embedding_path}  \
        --log-dir {ud_log_path} \
        --dropout 0.5 \
        --no-we-update 
        """.split()
    )


if __name__ == "__main__":
    languages = [
        "kk",
        "ta",
        "lv",
        "vi",
        "hu",
        "tr",
        "el",
        "bg",
        "sv",
        "eu",
        "ru",
        "da",
        "id",
        "zh",
        "fa",
        "he",
        "ro",
        "en",
        "ar",
        "hi",
        "it",
        "es",
        "cs",
    ]

    # for language_code in ['lv']:
    #     evaluate_pbos(language_code, False)
    #     evaluate_pbos(language_code, True)

    with mp.Pool() as pool:
        for language_code in languages:
            pool.apply_async(evaluate_pbos, (language_code, False,))
            pool.apply_async(evaluate_pbos, (language_code, True,))

        pool.close()
        pool.join()
