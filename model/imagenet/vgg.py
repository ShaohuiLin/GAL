import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.distributions.normal import Normal
from torch.nn import Parameter
import math
from collections import OrderedDict

norm_mean, norm_var = 1.0, 0.1

__all__ = ['VGG', 'vgg_16', 'vgg_16_sparse', 'vgg_16_bn', 'vgg_16_bn_sparse']

defaultcfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 512 * 7 * 7, 4096, 4096]

class Mask(nn.Module):
    def __init__(self, init_value=[1], fc=False):
        super().__init__()
        self.weight = Parameter(torch.Tensor(init_value))
        self.fc = fc

    def forward(self, input):
        """
        input_tensor: (N,C,H,W). 
        """
        if self.fc:
            weight = self.weight
        else:
            weight = self.weight[None, :, None, None]
        return input * weight


class VGG(nn.Module):

    def __init__(self, num_classes=1000, is_sparse=False, cfg=None, init_weights=True, batch_norm=False):
        super(VGG, self).__init__()
        # self.features = features
        self.batch_norm = batch_norm

        if cfg is None:
            cfg = defaultcfg

        if is_sparse:
            # m1 = Normal(torch.tensor([0.0]*int(cfg[-2])), torch.tensor([0.5]*int(cfg[-2]))).sample()
            # m2 = Normal(torch.tensor([0.0]*cfg[-1]), torch.tensor([0.5]*int(cfg[-1]))).sample()
            self.features = self.make_sparse_layers(cfg=cfg[0:-3], batch_norm=self.batch_norm)
            # self.classifier = nn.Sequential(OrderedDict([
            #     ('fc1', nn.Linear(cfg[-3], cfg[-2])),
            #     ('relu1', nn.ReLU(True)),
            #     ('mask1', Mask(m1,fc=True)),
            #     ('drop1', nn.Dropout()),
            #     ('fc2', nn.Linear(cfg[-2], cfg[-1])),
            #     ('relu2', nn.ReLU(True)),
            #     ('mask2', Mask(m2,fc=True)),
            #     ('drop2', nn.Dropout()),
            #     ('fc3', nn.Linear(cfg[-1], num_classes))
            # ]))
            
        else:
            self.features = self.make_layers(cfg=cfg[0:-3], batch_norm=self.batch_norm)
            # self.classifier = nn.Sequential(OrderedDict([
            #     ('fc1', nn.Linear(cfg[-3], cfg[-2])),
            #     ('relu1', nn.ReLU(True)),
            #     ('drop1', nn.Dropout()),
            #     ('fc2', nn.Linear(cfg[-2], cfg[-1])),
            #     ('relu2', nn.ReLU(True)),
            #     ('drop2', nn.Dropout()),
            #     ('fc3', nn.Linear(cfg[-1], num_classes))
            # ]))
            
        self.classifier = nn.Sequential(
            nn.Linear(cfg[-5] * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        ) 

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # print('after classifier', x.size())
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, cfg, batch_norm=False):
        layers = nn.Sequential()
        in_channels = 3
        for i, v in enumerate(cfg):
            if v == 'M':
                layers.add_module('pool%d' % i, nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers.add_module('conv%d' % i, conv2d)
                if batch_norm:
                    layers.add_module('norm%d' %i, nn.BatchNorm2d(v))
                layers.add_module('relu%d' % i, nn.ReLU(inplace=True))
                in_channels = v

        return layers

    def make_sparse_layers(self, cfg, batch_norm=False):
        layers = nn.Sequential()
        in_channels = 3
        for i, v in enumerate(cfg):
            if v == 'M':
                layers.add_module('pool%d' % i, nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers.add_module('conv%d' % i, conv2d)
                if batch_norm:
                    layers.add_module('norm%d' %i, nn.BatchNorm2d(v))
                layers.add_module('relu%d' % i, nn.ReLU(inplace=True))

                m = Normal(torch.tensor([norm_mean]*int(v)), torch.tensor([norm_var]*int(v))).sample()
                layers.add_module('mask%d' % i, Mask(m))

                in_channels = v

        return layers

def vgg_16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(**kwargs)
    return model

def vgg_16_sparse(pretrained=False, **kwargs):

    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(is_sparse=True, **kwargs)
    return model

def vgg_16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(batch_norm=True, **kwargs)
    return model

def vgg_16_bn_sparse(pretrained=False, **kwargs):

    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(is_sparse=True, batch_norm=True, **kwargs)
    return model



