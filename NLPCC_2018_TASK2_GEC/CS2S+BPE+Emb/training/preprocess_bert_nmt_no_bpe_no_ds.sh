#!/bin/bash

set -x

source ../paths.sh

## paths to training and development datasets
src_ext=src
trg_ext=trg
train_data_prefix=$DATA_DIR/train_no_bpe_no_ds.tok
dev_data_prefix=$DATA_DIR/dev_no_bpe_no_ds.tok

# path to subword nmt
# SUBWORD_NMT=$SOFTWARE_DIR/subword-nmt
# path to Fairseq-Py
FAIRSEQPY=$SOFTWARE_DIR/bert-nmt

mkdir -p processed/
# cp $train_data_prefix.$src_ext processed/train.all.src
# cp $train_data_prefix.$trg_ext processed/train.all.trg
cp -vf $dev_data_prefix.$src_ext processed/dev.src_bert_nmt
cp -vf $dev_data_prefix.$trg_ext processed/dev.trg_bert_nmt
# cp $dev_data_prefix.$src_ext processed/dev.input.txt

less processed/train.all.src > processed/train.src_bert_nmt
less processed/train.all.trg > processed/train.trg_bert_nmt

# bash ${FAIRSEQPY}/makedataforbert.sh src_bert_nmt
# bash ${FAIRSEQPY}/makedataforbert.sh trg_bert_nmt

#########################
# preprocessing
python $FAIRSEQPY/preprocess.py --source-lang src_bert_nmt --target-lang trg_bert_nmt --trainpref processed/train --validpref processed/dev --testpref processed/dev --destdir processed/bin_bert_nmt_no_ds --joined-dictionary --bert-model-name voidful/albert_chinese_tiny --workers 32
