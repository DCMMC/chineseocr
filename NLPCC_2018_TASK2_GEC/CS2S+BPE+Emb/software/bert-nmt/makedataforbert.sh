#!/usr/bin/env bash
prefix='/data/xiaowentao/chineseocr/NLPCC_2018_TASK2_GEC/CS2S+BPE+Emb/software/bert-nmt'
prefix2='/data/xiaowentao/chineseocr/NLPCC_2018_TASK2_GEC/CS2S+BPE+Emb/training/processed'
lng=$1
echo "src lng $lng"
for sub  in train dev
do
    sed -r 's/(@@ )|(@@ ?$)//g' ${prefix2}/${sub}.${lng} > ${prefix2}/${sub}.bert.${lng}.tok
    ${prefix}/mosesdecoder/scripts/tokenizer/detokenizer.perl -l $lng < ${prefix2}/${sub}.bert.${lng}.tok > ${prefix2}/${sub}.bert.${lng}
    rm ${prefix2}/${sub}.bert.${lng}.tok
done
