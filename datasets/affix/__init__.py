import gzip
from itertools import islice
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess as sp

from utils import dotdict


logger = logging.getLogger(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))


def prepare_affix_paths(
    *,
    dir_path=dir_path,
):
    dir_path = Path(dir_path)
    gz_path = dir_path / "affix_complete_set.txt.gz"
    raw_path = dir_path / "affix_complete_set.txt"
    queries_path = dir_path / "queries.txt"

    if not os.path.exists(gz_path):
        logger.info(f"Downloading {gz_path}")
        url = "http://marcobaroni.org/PublicData/affix_complete_set.txt.gz"
        sp.run(f"wget -O {gz_path} {url}".split())

    if not os.path.exists(raw_path):
        logger.info(f"Unzipping {raw_path}")
        with gzip.open(gz_path, 'rb') as fin, open(raw_path, 'wb') as fout:
            shutil.copyfileobj(fin, fout)

    if not os.path.exists(queries_path):
        logger.info(f"Making {queries_path}")
        with open(raw_path) as fin, open(queries_path, 'w') as fout:
            for line in islice(fin, 1, None): ## skip the title row
                ## row fmt: affix	stem	stemPOS	derived	derivedPOS	type	...
                affix, stem, _, derived, _, split = line.split()[:6]
                print(derived, file=fout)
                if derived.lower() != derived:
                    print(derived.lower(), file=fout)

    return dotdict(
        dir_path = dir_path,
        gz_path = gz_path,
        raw_path = raw_path,
        queries_path = queries_path,
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    prepare_affix_paths()
