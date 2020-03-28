# 全部导入内存，批处理，然后计算编辑距离并归一化
# 1. best path greedy 2. CTC beam search with n-gram LM (grid search for hyperparameters)
# 3. replace n-gram LM in CTC beam search with XLNet
# 4. 复现 Transormfer-based spell corrector

# debug
import faulthandler

faulthandler.enable()
import gc
import h5py
from PIL import Image
import os
import sys
from time import time
# 选用归一化后的编辑距离作为 evaluation metric
from Levenshtein import distance as levenshtein_distance
# model
from crnn.network_torch import CRNN
from crnn.keys import alphabetChinese as alphabet
from config import ocrModelTorchLstm as ocrModel
# decoders 取自 baidu 的 [DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech)
import swig_decoders
import random
import torch
from torch.nn.functional import softmax
import numpy as np
# matplotlib
# %matplotlib notebook
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class Scorer(swig_decoders.Scorer):
    def __init__(self, alpha, beta, model_path, vocabulary):
        swig_decoders.Scorer.__init__(self, alpha, beta, model_path, vocabulary)


def ctc_beam_search_decoder_batch_pred(probs_split,
                                      vocabulary,
                                      beam_size,
                                      num_processes,
                                      cutoff_prob=1.0,
                                      cutoff_top_n=40,
                                      ext_scoring_func=None):
    probs_split = [probs.tolist() for probs in probs_split]

    batch_beam_results = swig_decoders.ctc_beam_search_decoder_batch(
        probs_split, vocabulary, beam_size, num_processes, cutoff_prob,
        cutoff_top_n, ext_scoring_func)
    return [result[0][1] for result in batch_beam_results]


def levenshtein_distance_norm(str1, str2):
    '''
    归一化的编辑距离, 越小越好
    @return [0, 1]
    '''
    max_len = max(len(str1), len(str2), 1)
    return levenshtein_distance(str1, str2) / max_len


# set seed
seed = int(sys.argv[3])
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

GPUID = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
s_t = time()
nclass = len(alphabet) + 1
GPU = True
LSTMFLAG = True
crnn = CRNN(32, 1, nclass, 256, leakyRelu=False, lstmFlag=LSTMFLAG, GPU=GPU, alphabet=alphabet,
            vertical_text=False)
crnn.load_weights(ocrModel)
print('Load model done in {:.4f}s.'.format(time() - s_t))

alphabet_list = [c for c in alphabet]
ctc_buffer_size = 20
s_t = time()


# 因为数据规模不大，所以直接全部加载到内存
dataset_boxes = []
with h5py.File('/data/xiaowentao/chineseocr/dataset/ICPR_2018_MTWI_STR.hdf5', 'r') as f:
    print('Loading dataset into memory.')
    for img in f:
        # 暂时只考虑水平文本行
        if not bool(f[img]['vertical'][...]):
            if len(dataset_boxes) >= len(f) // 24:
                break
            dataset_boxes.append({
                # 灰度图
                'img': Image.fromarray(f[img]['img'][...]).convert('RGB').convert('L'),
                # 标签
                'true_text': str(f[img]['txt'][...])
            })
print('Dataset loaded, #sample={}, took {:.4f}s.'.format(len(dataset_boxes), time() - s_t))
# random sample 1/4 from dataset
# dataset_boxes = random.sample(dataset_boxes_all, len(dataset_boxes) // 24)
print('Randomly sample 1/24 from dataset: {} samples.'.format(len(dataset_boxes)))

# inference
alpha, beta = float(sys.argv[1]), float(sys.argv[2])
# CTC beam search with language model Scorer
scorer = Scorer(alpha=alpha, beta=beta, model_path=os.path.join('/data/xiaowentao/chineseocr',
                                                             'models/zh_giga.no_cna_cmn.prune01244.klm'),
                vocabulary=alphabet_list)
print('scorer done.')
# 默认的 beam_size=500 特别慢
ctc_beam_lm_decoder = lambda raw_preds: ctc_beam_search_decoder_batch_pred(
        probs_split=[softmax(raw_pred, dim=1) for raw_pred in raw_preds],
        vocabulary=alphabet_list,
        beam_size=10,
        num_processes=ctc_buffer_size,
        ext_scoring_func=scorer,
        cutoff_prob=1.0,
        cutoff_top_n=50)
boxes, losses, loss_total = crnn.predict_batch(dataset_boxes, batch_size=1, evaluation_per_batch=50,
                                               evaluation_metric=levenshtein_distance_norm,
                                               ctc_beam_lm_decoder=ctc_beam_lm_decoder,
                                               ctc_buffer_size=ctc_buffer_size)
print('-' * 60, '\nloss for alpha={} and beta={} is: {:.6f}'.format(alpha,
                                                                                   beta,
                                                                                   loss_total))
