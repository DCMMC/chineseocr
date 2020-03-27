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
from crnn.util import resiezeNormalizeVerticalText
from torchsummary import summary
from time import time


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
                 alphabet=None,
                 vertical_text=False
                ):
        """
        是否加入lstm特征层
        """
        super(CRNN, self).__init__()
        # imgH 在这里完全没用到啊
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        # number of channels in each layer
        nm = [64, 128, 256, 256, 512, 512, 512]
        self.lstmFlag = lstmFlag
        self.GPU = GPU
        self.alphabet = alphabet
        self.vertical_text = vertical_text
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

        # assume the shape of input is: 1x32x128 (C, H, W) for horizontal text and 1x128x32 for vertical text
        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64 or 64x64x16
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32 or 128x32x8
        convRelu(2, True)
        convRelu(3)
        if not self.vertical_text:
            cnn.add_module('pooling{0}'.format(2),
                           nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x32
        else:
            cnn.add_module('pooling{0}'.format(2),
                           nn.MaxPool2d((2, 2), (1, 2), (1, 0))) # 256x32x4
        convRelu(4, True)
        convRelu(5)
        if not self.vertical_text:
            cnn.add_module('pooling{0}'.format(3),
                           nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x32
        else:
            cnn.add_module('pooling{0}'.format(3),
                           nn.MaxPool2d((2, 2), (1, 2), (1, 0)))  # 512x32x2
        convRelu(6, True)  # 512x1x30 or 512x30x1
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
#         print('input size:', input.size())
        # conv features
        conv = self.cnn(input)
#         print('debug conv feature size in forward (b, c, h, w): ', conv.size())
        b, c, h, w = conv.size()

        if not self.vertical_text:
            assert h == 1, "the height of conv must be 1"
        else:
            assert w == 1, 'The width of conv must be 1'
        # (B, 512, 1, W/4) => (W/4, B, 512) or (B, 512, H/4, 1) => (H/4, B, 512)
        if not self.vertical_text:
            conv = conv.squeeze(2)
            conv = conv.permute(2, 0, 1)  # [w, b, c]
        else:
            conv = conv.squeeze(3).permute(2, 0, 1)
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
        if not self.vertical_text:
            image = resizeNormalize(image, 32)
        else:
            image = resiezeNormalizeVerticalText(image, 32)
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
            # PIL image object, if hight < 16 for horizontal text and width < 16 for vertical text
            if res_new.size[0 if self.vertical_text else 1] < 16:
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

    def predict_batch(self, boxes, batch_size=1, evaluation_per_batch=0, evaluation_metric=None,
                      ctc_beam_lm_decoder=None, ctc_buffer_size=1):
        """
        predict on batch
        DCMMC: 暂时只考虑横排文字
        """
        assert ctc_buffer_size >= 1
        ctc_buffer = []
        N = len(boxes)
        imgW = 0
        batch = N // batch_size
        if batch * batch_size != N:
            batch += 1
        # loss for each sample and in total
        losses = []
        loss_total = 0.
        print('#Batch: {} with batch_size={}'.format(batch, batch_size))
        s_t = time()
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
            n_sample = len(imageArray)
            # pad 0 to ensure length is imgW
            for j in range(n_sample):
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
            if ctc_beam_lm_decoder:
                # (B, T, nClass)
                preds = preds.transpose(1, 0)
                for pred in preds:
                    ctc_buffer.append(pred)
                for j in range(1, n_sample + 1):
                    if ((i * batch_size) + j) % ctc_buffer_size == 0:
                        pred_texts = ctc_beam_lm_decoder(ctc_buffer)
                        curr = i * batch_size + j
                        for idx, k in enumerate(range(curr - len(ctc_buffer), curr)):
                            boxes[k]['pred_text'] = pred_texts[idx]
                            if evaluation_metric:
                                loss = evaluation_metric(boxes[k]['true_text'], pred_texts[idx])
                                losses.append(loss)
                                loss_total += loss
                        ctc_buffer = []
            else:
                preds = preds.argmax(2)
            if not ctc_beam_lm_decoder:
                for j in range(n_sample):
                    # Best path greedy algorithm for CTC decode
                    pred_text = strLabelConverter(preds[:, j], self.alphabet)
                    boxes[i * batch_size + j]['pred_text'] = pred_text
                    if evaluation_metric:
                        loss = evaluation_metric(boxes[i * batch_size + j]['true_text'], pred_text)
                        losses.append(loss)
                        loss_total += loss
            if evaluation_per_batch > 0 and (i + 1) % evaluation_per_batch == 0:
                print('overall loss in {} batches with batch_size={} is {:.6f}, took {:.4f}s.'.format(
                      i + 1, batch_size,
                      loss_total / (i * batch_size + n_sample),
                      time() - s_t
                     ))
            # release cuda memory used by image
            del image
            del preds
            torch.cuda.empty_cache()
        print('All done after {:.4f}s, the overall loss is: {:.6f}'.format(time() - s_t, loss_total / N))
        return boxes, losses, loss_total / N
