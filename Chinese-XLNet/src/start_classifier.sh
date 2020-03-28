#!/bin/bash

base='/data/xiaowentao/chineseocr/Chinese-XLNet'

python -u run_classifier.py \
	--spiece_model_file=${base}/models/XLNet_spiece.model \
	--model_config_path=${base}/models/xlnet_config.json \
	--init_checkpoint=${base}/models/xlnet_model.ckpt \
	--task_name=csv \
	--data_dir=${base}/dataset/ChnSentiCorp \
	--output_dir=${base}/dataset \
	--predict_dir=${base}/dataset \
	--model_dir=${base}/models \
	--num_hosts=1 \
	--num_core_per_host=1 \
	--max_seq_length=256 \
	--do_predict=True \
	--predict_batch_size=2 \
	# --do_eval=True \
	# --eval_batch_size=2 \
	# --eval_all_ckpt=False
	# --do_train=True \
	# --train_batch_size=2 \
	# --learning_rate=2e-5 \
	# --save_steps=5000 \
	# --num_train_epochs=3
	#
