# coding: utf-8
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
""" An example of predicting CAPTCHA image data with a LSTM network pre-trained with a CTC loss"""

from __future__ import print_function

import sys
import os
import logging
import argparse
import h5py
import numpy as np
from Levenshtein import distance
from time import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnocr import CnOcr
from cnocr.utils import set_logger


logger = set_logger(log_level=logging.INFO)


def main():
    '''
    DCMMC: A very critical flaw: results predicted on multi-gpu are misplaced!
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", help="model name", type=str, default='conv-lite-fc'
    )
    parser.add_argument("--model_epoch", type=int, default=None, help="model epoch")
    parser.add_argument(
        '--dataset',
        type=str,
        help='location of hdf5 dataset'
    )
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='train ratio of the dataset')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--gpus', type=int, default=0,
                        help='number of gpus, 0 indicates cpu only')
    args = parser.parse_args()

    ocr = CnOcr(model_name=args.model_name, model_epoch=args.model_epoch,
                gpus=args.gpus)
    log_cp = 50
    dataset = h5py.File(args.dataset, 'r')
    batch_size = args.batch_size
    gold_train, gold_val = [], []
    res_train, res_val = [], []
    num_train = int(len(dataset) * args.train_ratio)
    num_train_batch = int(np.ceil(num_train / batch_size))
    num_val_batch = int(np.ceil((len(dataset) - num_train) / batch_size))
    logger.info(f'total num samples={len(dataset)}')
    logger.info(f'num_train_batch={num_train_batch}, num_val_batch={num_val_batch}')
    logger.info('start train dataset')
    s_t = time()
    for idx_batch in range(num_train_batch):
        data = []
        for idx in range(idx_batch * args.batch_size,
                         min((idx_batch+1) * args.batch_size, num_train)):
            data.append(dataset[str(idx)]['img'][...])
            gold_train.append(str(dataset[str(idx)]['y'][...]))
        res = ocr.ocr_for_single_lines(data)
        res = [''.join(r) for r in res]
        res_train += res
        if idx_batch % log_cp == 0:
            log_str = f'batch [{idx_batch + 1}/{num_train_batch}]: '
            log_str += '{:.2f}s/batch'.format((time() - s_t)/(idx_batch + 1))
            logger.info(log_str)
        # if idx_batch >= 4:
        #     break
    logger.info('start val dataset')
    s_t = time()
    for idx_batch in range(num_val_batch):
        data = []
        for idx in range(idx_batch * args.batch_size + num_train,
                         min((idx_batch+1) * args.batch_size + num_train,
                             len(dataset))):
            data.append(dataset[str(idx)]['img'][...])
            gold_val.append(str(dataset[str(idx)]['y'][...]))
        res = ocr.ocr_for_single_lines(data)
        res = [''.join(r) for r in res]
        res_val += res
        if idx_batch % log_cp == 0:
            log_str = f'batch [{idx_batch + 1}/{num_val_batch}]: '
            log_str += '{:.2f}s/batch'.format((time() - s_t)/(idx_batch + 1))
            logger.info(log_str)
        # if idx_batch >= 4:
        #     break
    assert len(res_val) == len(gold_val)
    assert len(res_train) == len(gold_train)
    acc_train = sum([r == p for r, p in zip(res_train, gold_train)]) / len(res_train)
    acc_val = sum([r == p for r, p in zip(res_val, gold_val)]) / len(res_val)
    logger.info(f'acc_train={acc_train}, acc_val={acc_val}')
    dist_fn = lambda r, p: distance(r, p) / max(len(r), len(p))
    dist_train = sum([dist_fn(r, p) for r, p in zip(res_train, gold_train)]) / len(res_train)
    dist_val = sum([dist_fn(r, p) for r, p in zip(res_val, gold_val)]) / len(res_val)
    logger.info(f'dist_train={dist_train}, dist_val={dist_val}')
    logger.info(f'write to file. #train={len(res_train)}, #val={len(res_val)}')
    with open('train_no_bpe.tok.src', 'w') as f:
        f.writelines([r + '\n' for r in res_train])
    with open('train_no_bpe.tok.trg', 'w') as f:
        f.writelines([r + '\n' for r in gold_train])
    with open('dev_no_bpe.tok.src', 'w') as f:
        f.writelines([r + '\n' for r in res_val])
    with open('dev_no_bpe.tok.trg', 'w') as f:
        f.writelines([r + '\n' for r in gold_val])
    logger.info('write done.')


if __name__ == '__main__':
    main()
