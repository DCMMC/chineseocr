#!/usr/bin/python
# encoding: utf-8
import numpy as np
from PIL import Image

def resizeNormalize(img,imgH=32):
    # print('*'*70, '\nDEBUG: img size:', img.size)
    scale = img.size[1]*1.0 / imgH
    w     = img.size[0] / scale
    w     = int(w)
    img   = img.resize((w,imgH),Image.BILINEAR)
    w,h   = img.size
    img = (np.array(img)/255.0-0.5)/0.5
    return img

def strLabelConverter(res,alphabet):
    # Naive version of CTC
    N = len(res)
    raw = []
    for i in range(N):
        if res[i] != 0 and (not (i > 0 and res[i - 1] == res[i])):
            raw.append(alphabet[res[i] - 1])
    return ''.join(raw)
