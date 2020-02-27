import logging
import os
import shutil
import tarfile

import subprocess as sp

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_polyglot_frequency_path(language_code):
    pkl_path = f"{dir_path}/polyglot-{language_code}.txt"
    tar_path = f"{dir_path}/polyglot-{language_code}.voc.tar.bz2"

    if not os.path.exists(tar_path):
        logging.info(f"Downloading {tar_path}")
        url = f"http://polyglot.cs.stonybrook.edu/~polyglot/counts2/{language_code}/{language_code}.voc.tar.bz2"
        sp.run(f"wget -O {tar_path} {url}".split())

    if not os.path.exists(pkl_path):
        logging.info(f"Unzipping {pkl_path}")
        with tarfile.open(tar_path) as tar, open(pkl_path, 'wb+') as dst_file:
            src_file = tar.extractfile(f"counts/{language_code}.docs.txt.voc")
            shutil.copyfileobj(src_file, dst_file)

    return pkl_path


if __name__ == '__main__':
    languages = ['kk', 'ta', 'lv', 'vi', 'hu', 'tr', 'el', 'bg', 'sv', 'eu', 'ru', 'da', 'id', 'zh', 'fa', 'he', 'ro',
                 'en', 'ar', 'hi', 'it', 'es', 'cs']
    for language_code in languages:
        get_polyglot_frequency_path(language_code)
