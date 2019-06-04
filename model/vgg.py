import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from collections import OrderedDict

norm_mean, norm_var = 0.0, 1.0

defaultcfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 512]

class ChannelSelection(nn.Module):
    def __init__(self, indexes, fc=False):
        super(ChannelSelection, self).__init__()
        self.indexes = indexes
        self.fc = fc

    def forward(self, input_tensor):
        if self.fc:
            return input_tensor[:, self.indexes]

        if len(self.indexes) == input_tensor.size()[1]:
            return input_tensor

        return input_tensor[:, self.indexes, :, :]

class Mask(nn.Module):
    def __init__(self, init_value=[1], fc=False):
        super().__init__()
        self.weight = Parameter(torch.Tensor(init_value))
        self.fc = fc

    def forward(self, input):
        if self.fc:
            weight = self.weight
        else:
            weight = self.weight[None, :, None, None]
        return input * weight

class VGG(nn.Module):
    def __init__(self, num_classes=10, init_weights=True, is_sparse=False, cfg=None, index=None):
        super(VGG, self).__init__()
        self.features = nn.Sequential()

        if cfg is None:
            cfg = defaultcfg

        if is_sparse:
            self.features = self.make_sparse_layers(cfg[:-1], True)
            m = Normal(torch.tensor([norm_mean]*cfg[-1]), torch.tensor([norm_var]*cfg[-1])).sample()
            self.classifier = nn.Sequential(OrderedDict([
                ('linear1', nn.Linear(cfg[-2], cfg[-1])),
                ('norm1', nn.BatchNorm1d(cfg[-1])),
                ('relu1', nn.ReLU(inplace=True)),
                ('mask', Mask(m, fc=True)),
                ('linear2', nn.Linear(cfg[-1], num_classes)),
            ]))
        else:
            self.features = self.make_layers(cfg[:-1], True)
            self.classifier = nn.Sequential(OrderedDict([
                ('linear1', nn.Linear(cfg[-2], cfg[-1])),
                ('norm1', nn.BatchNorm1d(cfg[-1])),
                ('relu1', nn.ReLU(inplace=True)),
                ('linear2', nn.Linear(cfg[-1], num_classes)),
            ]))
        
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=True):
        layers = nn.Sequential()
        in_channels = 3
        for i, v in enumerate(cfg):
            if v == 'M':
                layers.add_module('pool%d' % i, nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

                layers.add_module('conv%d' % i, conv2d)
                layers.add_module('norm%d' % i, nn.BatchNorm2d(v))
                layers.add_module('relu%d' % i, nn.ReLU(inplace=True))
                in_channels = v

        return layers

    def make_sparse_layers(self, cfg, batch_norm=True):
        in_channels = 3
        sparse_layers = nn.Sequential()
        for i, v in enumerate(cfg):
            if v == 'M':
                sparse_layers.add_module('pool%d' % i,nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    sparse_layers.add_module('conv%d' % i, conv2d)
                    sparse_layers.add_module('norm%d' % i, nn.BatchNorm2d(v))
                    sparse_layers.add_module('relu%d' % i, nn.ReLU(inplace=True))

                m = Normal(torch.tensor([norm_mean]*int(v)), torch.tensor([norm_var]*int(v))).sample()
                init_value = m
                sparse_layers.add_module('mask%d' % i, Mask(init_value))
                in_channels = v

        return sparse_layers

    def forward(self, x):
        x = self.features(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def vgg_16_bn(**kwargs):
    model = VGG(**kwargs)
    return model

def vgg_16_bn_sparse(**kwargs):
    model = VGG(is_sparse=True, **kwargs)
    return model


    