#!/bin/bash

set -x

source ../paths.sh

## paths to training and development datasets
src_ext=src
trg_ext=trg
train_data_prefix=$DATA_DIR/train_no_bpe.tok
dev_data_prefix=$DATA_DIR/dev_no_bpe.tok

# path to subword nmt
SUBWORD_NMT=$SOFTWARE_DIR/subword-nmt
# path to Fairseq-Py
FAIRSEQPY=$SOFTWARE_DIR/fairseq-py

mkdir -p processed/
cp $train_data_prefix.$src_ext processed/train.all.src
cp $train_data_prefix.$trg_ext processed/train.all.trg
cp  $dev_data_prefix.$src_ext processed/dev.src
cp $dev_data_prefix.$trg_ext processed/dev.trg
cp $dev_data_prefix.$src_ext processed/dev.input.txt

less processed/train.all.src > processed/train.src
less processed/train.all.trg > processed/train.trg

#########################
# preprocessing
python $FAIRSEQPY/preprocess.py --source-lang src --target-lang trg --trainpref processed/train --validpref processed/dev --testpref processed/dev --destdir processed/bin

