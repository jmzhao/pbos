#!/usr/bin/env bash
DATADIR=./datasets
RESULTSDIR=./results/pbos/demo
PRETRAINED=word2vec-google-news-300

mkdir -p "${RESULTSDIR}"
mkdir -p "${DATADIR}"

if [ ! -f "${DATADIR}/word2vec/processed.txt" ]
then
  GENSIM_DATA_DIR="${DATADIR}/gensim/" python prepare_target_embedding.py \
    --pretrained "${PRETRAINED}" \
    --output "${DATADIR}/${PRETRAINED}/processed.txt" 
fi

if [ ! -f "${DATADIR}/rw/rw.txt" ]
then
  wget -c 'https://nlp.stanford.edu/~lmthang/morphoNLM/rw.zip' -P "${DATADIR}"
  unzip "${DATADIR}/rw.zip" -d "${DATADIR}"
fi
if [ ! -f "${DATADIR}/rw/queries.txt" ]
then
  cut -f 1,2 "${DATADIR}/rw/rw.txt" | awk '{print tolower($0)}' | tr '\t' '\n' > "${DATADIR}/rw/queries.txt"
fi

python pbos_train.py --target "${DATADIR}/word2vec/processed.txt" --save "${RESULTSDIR}" --no-timestamp --epochs 20
python pbos_pred.py --queries "${DATADIR}/rw/queries.txt" --save "${RESULTSDIR}/rw_vectors.txt" --model "${RESULTSDIR}/model.bos"
python ./fastText/eval.py --data "${DATADIR}/rw/rw.txt" --model "${RESULTSDIR}/rw_vectors.txt"
