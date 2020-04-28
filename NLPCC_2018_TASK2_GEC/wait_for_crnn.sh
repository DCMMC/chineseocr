while [[ -n `ps -u --pid 104849 | grep 'hdf5'` ]]; do
	echo 'wait for 10min'
	sleep 600s
done
echo 'start training CS2S!'
cp -v ../cnocr/train_no_bpe* ../cnocr/dev_no_bpe* ./CS2S+BPE+Emb/data/ \
	&& cd CS2S+BPE+Emb/training && chmod +x preprocess_no_bpe.sh \
	&& ./preprocess_no_bpe.sh \
	&& chmod +x train_no_bpe.sh && ./train_no_bpe.sh

echo 'training CS2S done!'
