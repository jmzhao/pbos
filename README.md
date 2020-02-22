# Usage

- Download datasets

```shell script
cd datasets && make
```

- Run n-shortest-path segmentation

```shell script
python n_shortest_path.py
```

- To replicate BoS results
```
python pbos_demo.py --boundary --sub_min_len 3 --model_path ./results/pbos/demo/model.bos --mock_bos
```

- To replicate PBoS results
```
python pbos_demo.py --boundary --sub_min_len 3 --model_path ./results/pbos/demo/model.pbos
```


# Mimick

0. Initialize git submodule and apply patch

```shell script
git submodule init
git submodule update
git apply --stat mimick.patch
cd mimick
```

1. Download and unzip Universal Dependencies 1.4
 
```shell script
wget -O ud-treebanks-v1.4.tgz https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1827/ud-treebanks-v1.4.tgz?sequence=4&isAllowed=y
tar zxvf ud-treebanks-v1.4.tgz
```

2. Unzip models 

```shell script
for f in *.tar.gz
do
  tar -zxvf $f 
done
mv mimick/models/models/* mimick/models
```

3. Make Universal Dependencies dataset

```shell script
LANG_DIR=./ud-treebanks-v1.4/UD_English
LANG_PREFIX=en-ud
LANG_CODE=en
python make_dataset.py \
  --training-data $LANG_DIR/$LANG_PREFIX-train.conllu \
  --dev-data $LANG_DIR/$LANG_PREFIX-dev.conllu \
  --test-data $LANG_DIR/$LANG_PREFIX-test.conllu \
  --output $LANG_DIR/$LANG_PREFIX.pkl \
  --vocab $LANG_DIR/vocab-ud.txt
```

4. Download pre-trained word embeddings from polyglot

```shell script
wget -O $LANG_DIR/enembeddings_pkl.tar.bz2 http://polyglot.cs.stonybrook.edu/~polyglot/embeddings2/$LANG_CODE/embeddings_pkl.tar.bz2
tar -xjf $LANG_DIR/enembeddings_pkl.tar.bz2 -C $LANG_DIR

wget -O $LANG_DIR/freq.tar.bz2 http://polyglot.cs.stonybrook.edu/~polyglot/counts2/$LANG_CODE/$LANG_CODE.voc.tar.bz2
tar -xjf $LANG_DIR/freq.tar.bz2 -C $LANG_DIR
mv $LANG_DIR/counts/en.docs.txt.voc $LANG_DIR/freq.txt
```


5. mimick: Predict vocab embeddings using the target embeddings

```shell script
python mimick/inter_nearest_vecs.py \
  --mimick mimick/models/$LANG_CODE-lstm-est.bin \
  --c2i mimick/models/$LANG_CODE-lstm-est.c2i \
  --vectors $LANG_DIR/words_embeddings_32.pkl \
  --vocab $LANG_DIR/vocab-ud.txt \
  --output $LANG_DIR/embeddings-mimick.txt 
```

6. mimick: Test POS and morphosyntactic attributes

```shell script
python model.py \
  --dataset $LANG_DIR/$LANG_PREFIX.pkl \
  --word-embeddings $LANG_DIR/embeddings-mimick.txt  \
  --log-dir $LANG_DIR/log-mimick \
  --no-we-update 
```
