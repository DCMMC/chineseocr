#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
torch ocr model
@author: chineseocr
'''
import numpy as np
import torch.nn as nn
import torch
from collections import OrderedDict
from torch.autograd import Variable
from crnn.util import resizeNormalize, strLabelConverter, index_to_str
from torchsummary import summary


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):
    def __init__(self,
                 imgH,
                 nc,
                 nclass,
                 nh,
                 leakyRelu=False,
                 lstmFlag=True,
                 GPU=False,
                 alphabet=None):
        """
        是否加入lstm特征层
        """
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        # number of channels in each layer
        nm = [64, 128, 256, 256, 512, 512, 512]
        self.lstmFlag = lstmFlag
        self.GPU = GPU
        self.alphabet = alphabet
        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16
        self.cnn = cnn
        if self.lstmFlag:
            # two-layer LSTM: 512-d to nClass-d
            self.rnn = nn.Sequential(
                BidirectionalLSTM(512, nh, nh),
                BidirectionalLSTM(nh, nh, nclass))
        else:
            # nh * 2 masu equal to 512
            self.linear = nn.Linear(nh * 2, nclass)

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()

        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        if self.lstmFlag:
            # rnn features
            output = self.rnn(conv)
            T, b, h = output.size()
            output = output.view(T, b, -1)
        else:
            T, b, h = conv.size()
            t_rec = conv.contiguous().view(T * b, h)
            output = self.linear(t_rec)  # [T * b, nOut]
            output = output.view(T, b, -1)
        return output

    def load_weights(self, path):
        trainWeights = torch.load(
            path, map_location=lambda storage, loc: storage)
        modelWeights = OrderedDict()
        for k, v in trainWeights.items():
            name = k.replace('module.', '')  # remove `module.`
            modelWeights[name] = v
        self.load_state_dict(modelWeights)
        if torch.cuda.is_available() and self.GPU:
            self.cuda()
        self.eval()

    def predict(self, image):
        image = resizeNormalize(image, 32)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        if torch.cuda.is_available() and self.GPU:
            image = image.cuda()
        else:
            image = image.cpu()
        image = image.view(1, 1, *image.size())
        image = Variable(image)
        preds = self(image)
        # B * T * 4, debug by DCMMC
        _, topk_res = preds.transpose(1, 0).topk(k=4, dim=2)
        topk_res = index_to_str(topk_res, self.alphabet)
        _, preds_logit = preds.max(2)
        # B * T where B = 1
        preds_logit = preds_logit.transpose(1, 0).contiguous().view(-1)
        raw = strLabelConverter(preds_logit, self.alphabet)
        return raw, topk_res, preds.transpose(1, 0)

    def predict_job(self, boxes, print_summary=False):
        n = len(boxes)
        if print_summary:
            summary(self, input_size=(1, 32, 128))
        for i in range(n):
            res_new = boxes[i]['img']
            # PIL image object
            if res_new.size[1] < 16:
                boxes[i]['text'] = ''
                boxes[i]['raw res'] = ''
                boxes[i]['raw preds'] = []
                continue
            res = self.predict(res_new)
            boxes[i]['text'] = res[0]
            # DCMCM: for debugging
            boxes[i]['raw res'] = res[1]
            boxes[i]['raw preds'] = res[2]
        return boxes

    def predict_batch(self, boxes, batch_size=1):
        """
        predict on batch
        """
        N = len(boxes)
        res = []
        imgW = 0
        batch = N // batch_size
        if batch * batch_size != N:
            batch += 1
        for i in range(batch):
            tmpBoxes = boxes[i * batch_size:(i + 1) * batch_size]
            imageBatch = []
            imgW = 0
            for box in tmpBoxes:
                img = box['img']
                image = resizeNormalize(img, 32)
                h, w = image.shape[:2]
                imgW = max(imgW, w)
                imageBatch.append(np.array([image]))
            # (N, C, H, W)
            imageArray = np.zeros((len(imageBatch), 1, 32, imgW),
                                  dtype=np.float32)
            n = len(imageArray)
            # pad 0 to ensure length is imgW
            for j in range(n):
                _, h, w = imageBatch[j].shape
                imageArray[j][:, :, :w] = imageBatch[j]
            image = torch.from_numpy(imageArray)
            image = Variable(image)
            if torch.cuda.is_available() and self.GPU:
                image = image.cuda()
            else:
                image = image.cpu()
            # (T, B, nClass)
            preds = self(image)
            preds = preds.argmax(2)
            n = preds.shape[1]
            for j in range(n):
                res.append(strLabelConverter(preds[:, j], self.alphabet))

        for i in range(N):
            boxes[i]['text'] = res[i]
        return boxes
