#!/bin/bash

base='/data/xiaowentao/chineseocr/Chinese-XLNet'

python -u run_classifier.py \
	--spiece_model_file=${base}/models/spiece.model \
	--model_config_path=${base}/models/xlnet_config.json \
	--init_checkpoint=${base}/models/xlnet_model.ckpt \
	--task_name=csv \
	--do_eval=True \
	--eval_all_ckpt=False \
	--data_dir=${base}/dataset \
	--output_dir=${base}/dataset \
	--model_dir=${base}/models \
	--eval_batch_size=48 \
	--num_hosts=1 \
	--num_core_per_host=16 \
	--max_seq_length=256
