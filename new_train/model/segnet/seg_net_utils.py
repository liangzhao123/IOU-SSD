import torch
import torch.nn as nn
import numpy as np
from pvdet.tools.utils import loss_utils
from functools import partial

class SegNet(nn.Module):
    def __init__(self,inchannel):
        super().__init__()
        norm_fun = partial(nn.BatchNorm1d,eps=1e-3,momentum=0.01)
        block = self.conv_block

        self.conv1 = nn.Sequential(nn.Conv1d(inchannel,32,3,bias=False,stride=1,padding=1),
                                   norm_fun(32),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(32,32,3,bias=False,stride=1,padding=1),
                                   norm_fun(32),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(32,32,3,bias=False,stride=1,padding=1),
                                   norm_fun(32),
                                   nn.ReLU())
        self.deconv1  = nn.Sequential(nn.ConvTranspose1d(32,64,3,stride=1,bias=False,padding=1),
                                     norm_fun(64),
                                     nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(32,32,3,bias=False,stride=1,padding=1),
                                   norm_fun(32),
                                   nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(32,32,3,bias=False,stride=1,padding=1),
                                   norm_fun(32),
                                   nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv1d(32,32,3,bias=False,stride=1,padding=1),
                                   norm_fun(32),
                                   nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose1d(32,64,3,stride=1,bias=False,padding=1),
                                     norm_fun(64),
                                     nn.ReLU())

        self.cls_layer = nn.Conv1d(128,2,1,bias=True)
        self.seg_loss_func = loss_utils.SigmoidFocalClassificationLoss_v1(
            alpha=0.25, gamma=2.0)
        self.ret = {}
        self.init_weights()
    def init_weights(self):

        pi = 0.01
        nn.init.constant_(self.cls_layer.bias, -np.log((1 - pi) / pi))

    def conv_block(self,key,inchannel,outchannel,kernel_size,stride,padding,bias=False):
        norm_fun = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        conv = None
        if key =="conv1d":
            norm_fun = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
            conv = nn.Sequential(nn.Conv1d(inchannel,outchannel,kernel_size,stride=stride,padding=padding,bias=bias),
                                 norm_fun(outchannel),
                                 nn.ReLU())
        elif key == "ConvTranspose1d":
            conv = nn.Sequential(
                nn.ConvTranspose1d(inchannel, outchannel, kernel_size, stride=stride, padding=padding, bias=bias),
                norm_fun(outchannel),
                nn.ReLU())
        return conv


    def forward(self,x_conv1):
        x_input = x_conv1.features.view(-1,16).permute(1,0).unsqueeze(dim=0)#16000
        ups = []
        x = self.conv1(x_input)
        x = self.conv2(x)
        x = self.conv3(x)
        ups.append(self.deconv1(x))

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        ups.append(self.deconv2(x))

        x = torch.cat(ups,dim=1)
        seg_pred = self.cls_layer(x)


        return seg_pred.view(-1,2)