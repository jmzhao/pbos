import logging
import multiprocessing as mp
import os
import subprocess as sp

from datasets.polyglot_emb import prepare_polyglot_emb_paths
from datasets.polyglot_emb import languages as all_language_codes
from datasets.polyglot_freq import prepare_polyglot_freq_paths
from datasets.ud import prepare_ud_paths
from utils.args import add_logging_args, set_logging_config
from load import load_embedding

logger = logging.getLogger(__name__)


def tee_open(path):
    return sp.Popen(['/usr/bin/tee', '-a', path], stdin=sp.PIPE)


def pos_eval(ud_data_path, ud_vocab_embedding_path, result_path):
    """train pos tagging and report scores"""
    ud_log_path = os.path.join(result_path , "ud.log")
    ud_out_path = os.path.join(result_path , "ud.out")
    cmd = f"""
        python pos_eval.py \
        --dataset {ud_data_path} \
        --embeddings {ud_vocab_embedding_path} \
    """.split()
    with \
        tee_open(ud_log_path) as log_tee, \
        tee_open(ud_out_path) as out_tee:
        sp.call(cmd, stdout=out_tee.stdin, stderr = log_tee.stdin)


def evaluate_pbos(language_code, model_type):
    logger.info(f"[evaluate_pbos({language_code}, model_type={model_type})] start...")

    # Input files
    polyglot_embeddings_path = prepare_polyglot_emb_paths(language_code)
    polyglot_frequency_path = prepare_polyglot_freq_paths(language_code)

    # Output/result files
    result_path = os.path.join("results", "polyglot", language_code, model_type)
    os.makedirs(result_path, exist_ok=True)
    subword_vocab_path = os.path.join(result_path, "subword_vocab.jsonl")
    subword_prob_path = os.path.join(result_path, "subword_prob.jsonl")
    subword_embedding_model_path = os.path.join(result_path , "model.pbos")
    training_log_path = subword_embedding_model_path + ".log"
    logger.info(f"[evaluate_pbos({language_code}, model_type={model_type})]"
        f" result_path=`{result_path}`")

    # train subword embedding model using target embeddings and word freq
    if not os.path.exists(subword_embedding_model_path):
        # build subword vocab from target words
        logger.info(f"[evaluate_pbos({language_code}, model_type={model_type})]"
            f" building subword vocab...")
        cmd = f"""
            python subwords.py build_vocab \
                --word_freq {polyglot_embeddings_path.word_freq_path} \
                --output {subword_vocab_path} \
                --word_boundary \
        """.split()
        sp.call(cmd)

        if model_type == 'pbos':
            # build subword prob from word freqs
            logger.info(f"[evaluate_pbos({language_code}, model_type={model_type})]"
                f" building subword prob...")
            cmd = f"""
                python subwords.py build_prob \
                    --word_freq {polyglot_frequency_path.word_freq_path} \
                    --output {subword_prob_path} \
                    --word_boundary \
                    --subword_prob_min_prob 1e-6 \
            """.split()
            sp.call(cmd)
        else:
            logger.info(f"[evaluate_pbos({language_code}, model_type={model_type})]"
                f" skipped building subword prob.")

        # invoke training of subword model
        logger.info(f"[evaluate_pbos({language_code}, model_type={model_type})]"
            f" training subword model...")
        cmd = f"""
            python pbos_train.py \
              --target_vectors {polyglot_embeddings_path.pkl_path} \
              --model_path {subword_embedding_model_path} \
              --subword_vocab {subword_vocab_path} \
              --word_boundary \
              --subword_prob_normalize_emb \
        """
        if model_type == 'pbos':
            cmd += f" --subword_prob {subword_prob_path}"
        cmd = cmd.split()
        # with open(training_log_path, "w+") as log:
        #     sp.call(cmd, stdout=log, stderr=log)
        with tee_open(training_log_path) as log_tee:
            sp.call(cmd, stdout=log_tee.stdin, stderr=log_tee.stdin)
    else:
        logger.info(f"[evaluate_pbos({language_code}, model_type={model_type})]"
            f" skipped training subword model.")

    ud_data_path, ud_vocab_path = prepare_ud_paths(language_code)
    ud_vocab_embedding_path = os.path.join(result_path, "ud_vocab_embedding.txt")

    # predict embeddings for ud vocabs
    if not os.path.exists(ud_vocab_embedding_path):
        logger.info(f"[evaluate_pbos({language_code}, model_type={model_type})]"
            f" predicting word embeddings...")
        cmd = f"""
            python pbos_pred.py \
            --queries {ud_vocab_path} \
            --save {ud_vocab_embedding_path} \
            --model {subword_embedding_model_path} \
            --word_boundary \
        """.split()
            # --pre_trained {polyglot_embeddings_path.pkl_path} \
        sp.call(cmd)
    else:
        logger.info(f"[evaluate_pbos({language_code}, model_type={model_type})]"
            f" skipped predicting word embeddings.")

    logger.info(f"[evaluate_pbos({language_code}, model_type={model_type})]"
        f" evaluating on POS tagging...")
    pos_eval(ud_data_path, ud_vocab_embedding_path, result_path)

    logger.info(f"[evaluate_pbos({language_code}, model_type={model_type})]"
        f" done.")



def main():
    import argparse

    parser = argparse.ArgumentParser("Run POS tagging experiments on PolyGlot and UD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--languages', '-langs', nargs='+', metavar="LANG_CODE",
        choices=all_language_codes + ["ALL"],
        help="languages to evaluate over")
    parser.add_argument('--num_processes', '-nproc', type=int,
        help="number of processers to use")
    add_logging_args(parser)
    args = parser.parse_args()

    set_logging_config(args)

    language_codes = all_language_codes if "ALL" in args.languages else args.languages
    logger.debug(f"language_codes: {language_codes}")
    def job(apply):
        for language_code in language_codes:
            # prepare raw data without multiprocessing,
            # otherwise trouble comes with race conditions of file write
            prepare_polyglot_emb_paths(language_code)
            prepare_polyglot_freq_paths(language_code)
            prepare_ud_paths(language_code)
            apply(evaluate_pbos, (language_code, 'pbos',))
            # apply(evaluate_pbos, (language_code, 'bos',))
    if args.num_processes == 1:
        def apply(func, args):
            return func(*args)
        job(apply)
    else:
        with mp.Pool(args.num_processes) as pool:
            results = []
            def apply(func, args):
                return results.append(pool.apply_async(func, args))
            job(apply)
            for r in results:
                r.get()
    logger.debug("done.")

if __name__ == "__main__":
    main()
