#!/usr/bin/python
# encoding: utf-8
import numpy as np
from PIL import Image


def resizeNormalize(img, imgH=32):
#     print('*'*70, '\nDEBUG: img size:', img.size, ', type:', type(img))
    scale = img.size[1] * 1.0 / imgH
    w = img.size[0] / scale
    w = int(w)
    img = img.resize((w, imgH), Image.BILINEAR)
    w, h = img.size
#     print(np.array(img).size)
    # PIL 是 W * H 而 np.array 和 Pytorch 都是 H * W 形状，这里 np 会自动转换
    img = (np.array(img) / 255.0 - 0.5) / 0.5
    return img


def strLabelConverter(res, alphabet):
    # Naive version of CTC
    N = len(res)
    raw = []
    for i in range(N):
        if res[i] != 0 and (not (i > 0 and res[i - 1] == res[i])):
            raw.append(alphabet[res[i] - 1])
    return ''.join(raw)


def index_to_str(res_one_sample_in_batch, alphabet, blank='-'):
    assert res_one_sample_in_batch.shape[0] == 1
    res_new = [[alphabet[index - 1] if index else blank for index in pos]
               for pos in res_one_sample_in_batch[0]]
    res_str = '\n'
    for index in range(len(res_new[0])):
        res_str += ''.join([pos[index] for pos in res_new])
        res_str += '\n'
    return res_str
