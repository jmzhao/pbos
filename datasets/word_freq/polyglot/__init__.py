import json
import logging
import os
import shutil
import subprocess as sp
import tarfile

from utils import dotdict

logger = logging.getLogger(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))


def prepare_polyglot_freq_paths(
    language_code,
    *,
    dir_path=dir_path,
    word_freq_max_size=1000000,
):
    language_dir_path = os.path.join(dir_path, language_code)
    tar_path = os.path.join(language_dir_path, f"{language_code}.voc.tar.bz2")
    raw_count_path = os.path.join(language_dir_path, "count.txt")
    word_freq_path = os.path.join(language_dir_path, "word_freq.jsonl")

    os.makedirs(language_dir_path, exist_ok=True)

    if not os.path.exists(tar_path):
        logger.info(f"Downloading {tar_path}")
        url = f"http://polyglot.cs.stonybrook.edu/~polyglot/counts2/{language_code}/{language_code}.voc.tar.bz2"
        sp.run(f"wget -O {tar_path} {url}".split())

    if not os.path.exists(raw_count_path):
        logger.info(f"Unzipping {raw_count_path}")
        with tarfile.open(tar_path) as tar, open(raw_count_path, 'wb+') as dst_file:
            src_file = tar.extractfile(f"counts/{language_code}.docs.txt.voc")
            shutil.copyfileobj(src_file, dst_file)

    if not os.path.exists(word_freq_path):
        with open(raw_count_path) as fin, open(word_freq_path, "w") as fout:
            for i_line, line in enumerate(fin):
                if i_line >= word_freq_max_size:
                    break
                word, count = line.split()
                count = int(count)
                print(json.dumps((word, count)), file=fout)

    return dotdict(
        dir_path = dir_path,
        language_dir_path = language_dir_path,
        tar_path = tar_path,
        raw_count_path = raw_count_path,
        word_freq_path = word_freq_path,
    )


if __name__ == '__main__':
    from datasets import polyglot_languages

    for language_code in polyglot_languages:
        prepare_polyglot_freq_paths(language_code)
