# -*- coding: utf-8 -*-
'''
@author DCMMC
'''
import os
GPUID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

from config import IMGSIZE, chineseModel
import sys
from PIL import Image
from main import TextOcrModel
from text.darknet_detect import text_detect
if chineseModel:
    from crnn.keys import alphabetChinese as alphabet
else:
    from crnn.keys import alphabetEnglish as alphabet
from config import ocrModelTorchLstm as ocrModel
from crnn.network_torch import CRNN
import numpy as np

GPU = True
LSTMFLAG = True
nclass = len(alphabet) + 1
scale, maxScale = IMGSIZE

if __name__ == '__main__':
    img_file = None
    img_file = img_file or sys.argv[1]
    img = np.array(Image.open(img_file).convert('RGB'))

    crnn = CRNN(
        32,
        1,
        nclass,
        256,
        leakyRelu=False,
        lstmFlag=LSTMFLAG,
        GPU=GPU,
        alphabet=alphabet)
    crnn.load_weights(ocrModel)
    ocr = crnn.predict_job

    model = TextOcrModel(ocr, text_detect, None)
    billList = ['通用OCR', '火车票', '身份证']

    detectAngle = False
    result, angle = model.model(
        img,
        scale=scale,
        maxScale=maxScale,
        detectAngle=detectAngle,  ##是否进行文字方向检测，通过web传参控制
        MAX_HORIZONTAL_GAP=100,  ##字符之间的最大间隔，用于文本行的合并
        MIN_V_OVERLAPS=0.6,
        MIN_SIZE_SIM=0.6,
        TEXT_PROPOSALS_MIN_SCORE=0.1,
        TEXT_PROPOSALS_NMS_THRESH=0.3,
        TEXT_LINE_NMS_THRESH=0.99,  ##文本行之间测iou值
        LINE_MIN_SCORE=0.1,
        leftAdjustAlph=0.01,  ##对检测的文本行进行向左延伸
        rightAdjustAlph=0.01,  ##对检测的文本行进行向右延伸
    )

    for res in result:
        for k in res:
            print(k, ':\t', res[k])
        print('-'*70)
