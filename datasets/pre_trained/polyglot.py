import os
import shutil
import tarfile

import subprocess as sp

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_polyglot_embeddings_path(language_code):
    pkl_path = f"{dir_path}/polyglot-{language_code}.pkl"
    tar_path = f"{dir_path}/polyglot-{language_code}.tar.bz2"

    if not os.path.exists(tar_path):
        url = f"http://polyglot.cs.stonybrook.edu/~polyglot/embeddings2/{language_code}/embeddings_pkl.tar.bz2"
        sp.run(f"wget -O {tar_path} {url}".split())

    if not os.path.exists(pkl_path):
        with tarfile.open(tar_path) as tar, open(pkl_path, 'wb+') as dst_file:
            src_file = tar.extractfile("./words_embeddings_32.pkl")
            shutil.copyfileobj(src_file, dst_file)

    return pkl_path


if __name__ == '__main__':
    languages = ['kk', 'ta', 'lv', 'vi', 'hu', 'tr', 'el', 'bg', 'sv', 'eu', 'ru', 'da', 'id', 'zh', 'fa', 'he', 'ro',
                 'en', 'ar', 'hi', 'it', 'es', 'cs']
    for language_code in languages:
        get_polyglot_embeddings_path(language_code)
