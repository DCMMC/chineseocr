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

# Tuning hyperparameters \alpha and \beta of Scorer by grid search
alpha_start, alpha_end, num_alpha = 0.0, 4.0, 20
beta_start, beta_end, num_beta = 0.0, 1.0, 10
alphabet_list = [c for c in alphabet]
# create grid for search
cand_alphas = np.linspace(alpha_start, alpha_end, num_alpha)
cand_betas = np.linspace(beta_start, beta_end, num_beta)
params_grid = [(alpha, beta) for alpha in cand_alphas
               for beta in cand_betas]
# 应为 CTC beam search 实在是有点慢，所以最好采用多线程，这里分配 30 核
ctc_buffer_size = 20
print('{} trials for grid search created.'.format(len(params_grid)))
s_t = time()


# 因为数据规模不大，所以直接全部加载到内存
dataset_boxes_all = []
with h5py.File('/data/xiaowentao/chineseocr/dataset/ICPR_2018_MTWI_STR.hdf5', 'r') as f:
    print('Loading dataset into memory.')
    for img in f:
        # 暂时只考虑水平文本行
        if not bool(f[img]['vertical'][...]):
            if len(dataset_boxes_all) >= 1 * 256 * 10:
                break
            dataset_boxes_all.append({
                # 灰度图
                'img': Image.fromarray(f[img]['img'][...]).convert('RGB').convert('L'),
                # 标签
                'true_text': str(f[img]['txt'][...])
            })
print('Dataset loaded, #sample={}, took {:.4f}s.'.format(len(dataset_boxes_all), time() - s_t))
# random sample 1/4 from dataset
dataset_boxes = random.sample(dataset_boxes_all, len(dataset_boxes_all) // 4)
del dataset_boxes_all
print('Randomly sample 1/4 from dataset: {} samples.'.format(len(dataset_boxes)))

# inference
grid_res = []
scorer = None
for idx, (alpha, beta) in enumerate(params_grid):
    if scorer:
        del scorer
    gc.collect()
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
    grid_res.append(loss_total)
    print('-' * 60, '\nTrial {} done, loss for alpha={} and beta={} is: {:.6f}'.format(idx,
                                                                                       alpha,
                                                                                       beta,
                                                                                       loss_total))
    
# plot the results
best_res_idx = np.argmax(grid_res)
print('All done, the best result for grid research is: {} when alpha={} and beta={}.'.format(
    params_grid[best_res_idx], *grid_res[best_res_idx]))
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface([i[0] for i in params_grid], [i[1] for i in params_grid],
                       grid_res, cmap=cm.coolwarm, alpha=0.7,
                       linewidth=0, antialiased=False)
ax.set_zlim(-0.1, 1.0)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.04f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()