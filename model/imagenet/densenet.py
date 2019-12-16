import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from torch.distributions.normal import Normal
from torch.autograd import Variable
from torch.nn import Parameter


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, filters, index, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class channel_selection(nn.Module):
    def __init__(self, indexes):
        super(channel_selection, self).__init__()
        self.indexes = indexes

    def forward(self, input_tensor):
        """
        input_tensor: (N,C,H,W)
        """
        if len(self.indexes) == input_tensor.size()[1]:
            return input_tensor

        output = input_tensor[:, self.indexes, :, :]
        return output

class Mask(nn.Module):
    def __init__(self, init_value=[1]):
        super().__init__()
        self.weight = Parameter(torch.Tensor(init_value))

    def forward(self, input):
        """
        input_tensor: (N,C,H,W). 
        """
        # print("input size", input.size())
        weight = self.weight[None, :, None, None]
        return input * weight

class _SparseDenseLayer(nn.Sequential):
    def __init__(self, num_input_features, filters, index, growth_rate, bn_size, drop_rate):
        super(_SparseDenseLayer, self).__init__()
        m = Normal(torch.tensor([1.0]*filters), torch.tensor([0.1]*filters)).sample()
        self.init_value = m
        
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('select', channel_selection(index))
        self.add_module('mask', Mask(m))
        self.add_module('conv1', nn.Conv2d(filters, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_SparseDenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, dense, num_layers, num_input_features, filters, index, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = dense(num_input_features + i * growth_rate, filters[i], index[i], growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, filters, index):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class _SparseTransition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, filters, index):
        super(_SparseTransition, self).__init__()
        m = Normal(torch.tensor([1.0]*filters), torch.tensor([0.1]*filters)).sample()

        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('select', channel_selection(index))
        self.add_module('mask', Mask(m))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, dense=_DenseLayer, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, filters=None, indexes=None):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        if filters == None:
            filters = []
            start = growth_rate*2
            for i in range(4):
                n = int(block_config[i])
                filters.append([start + growth_rate*i for i in range(n+1)])
                start = (start + growth_rate * n) // 2
                # print(start)
            filters = [item for sub_list in filters for item in sub_list]
            
            indexes = []

            for f in filters:
                indexes.append([1]*f)

        # Each denseblock
        transition = _SparseTransition if 'Sparse' in str(dense) else _Transition

        num_features = num_init_features
        filter_start = 0
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(dense=dense, num_layers=num_layers, num_input_features=num_features,filters=filters[filter_start: filter_start+num_layers], index=indexes[filter_start: filter_start+num_layers], bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

            # filter_start += 1
            filter_start = filter_start + num_layers
            # print(filter_start)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = transition(num_input_features=num_features, num_output_features=num_features // 2, filters=filters[filter_start], index=indexes[filter_start])
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
                filter_start += 1

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def densenet_121(pretrained=False, **kwargs):
    model = DenseNet(dense=_DenseLayer, num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
        return model, state_dict

    return model

def densenet_121_sparse(pretrained=False,  **kwargs):
    model = DenseNet(dense=_SparseDenseLayer, num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),**kwargs)
    return model


