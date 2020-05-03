#!/bin/bash

set -x

source ../paths.sh

## paths to training and development datasets
src_ext=src
trg_ext=trg
train_data_prefix=$DATA_DIR/train_no_bpe_no_ds.tok
dev_data_prefix=$DATA_DIR/dev_no_bpe_no_ds.tok

# path to subword nmt
SUBWORD_NMT=$SOFTWARE_DIR/subword-nmt
# path to Fairseq-Py
FAIRSEQPY=$SOFTWARE_DIR/fairseq-py

mkdir -p processed/
# cp $train_data_prefix.$src_ext processed/train.all.src_no_ds
# cp $train_data_prefix.$trg_ext processed/train.all.trg_no_ds
cp $dev_data_prefix.$src_ext processed/dev.src_no_ds
cp $dev_data_prefix.$trg_ext processed/dev.trg_no_ds

less processed/train.all.src > processed/train.src_no_ds
less processed/train.all.trg > processed/train.trg_no_ds

#########################
# preprocessing
python $FAIRSEQPY/preprocess.py --source-lang src_no_ds --target-lang trg_no_ds --trainpref processed/train --validpref processed/dev --testpref processed/dev --destdir processed/bin

