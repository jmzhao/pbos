#!/usr/bin/env bash
DATADIR=./datasets
RESULTSDIR=./results/pbos/demo
PRETRAINED=word2vec-google-news-300

mkdir -p "${RESULTSDIR}"
mkdir -p "${DATADIR}"

if [ ! -f "${DATADIR}/${PRETRAINED}/processed.txt" ]
then
  GENSIM_DATA_DIR="${DATADIR}/gensim/" python prepare_target_embedding.py \
    --pretrained "${PRETRAINED}" \
    --output "${DATADIR}/${PRETRAINED}/processed.txt"
fi
if [ ! -f "${DATADIR}/${PRETRAINED}/word_list.txt" ]
then
  cut -d " " -f 1 "${DATADIR}/${PRETRAINED}/processed.txt" \
    > "${DATADIR}/${PRETRAINED}/word_list.txt"
fi


if [ ! -f "${DATADIR}/rw/rw.txt" ]
then
  wget -c 'https://nlp.stanford.edu/~lmthang/morphoNLM/rw.zip' -P "${DATADIR}"
  unzip "${DATADIR}/rw.zip" -d "${DATADIR}"
fi
if [ ! -f "${DATADIR}/rw/queries.txt" ]
then
  cut -f 1,2 "${DATADIR}/rw/rw.txt" | awk '{print tolower($0)}' | tr '\t' '\n' \
    > "${DATADIR}/rw/queries.txt"
fi

python pbos_train.py \
  --target "${DATADIR}/${PRETRAINED}/processed.txt" \
  --word_list "${DATADIR}/${PRETRAINED}/word_list.txt" \
  --save "${RESULTSDIR}/model.pbos" \
  --epochs 10 --lr_decay #--boundary
python pbos_pred.py \
  --queries "${DATADIR}/rw/queries.txt" \
  --save "${RESULTSDIR}/rw_vectors.txt" \
  --model "${RESULTSDIR}/model.pbos"
python ./fastText/eval.py \
  --data "${DATADIR}/rw/rw.txt" \
  --model "${RESULTSDIR}/rw_vectors.txt"
