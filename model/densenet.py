import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import numpy as np
from torch.autograd import Variable
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
import pdb

norm_mean, norm_var = 0.0, 1.0

class ChannelSelection(nn.Module):
    def __init__(self, indexes):
        super(ChannelSelection, self).__init__()
        self.indexes = indexes

    def forward(self, input_tensor):
        if len(self.indexes) == input_tensor.size()[1]:
            return input_tensor

        output = input_tensor[:, self.indexes, :, :]
        return output

class Mask(nn.Module):
    def __init__(self, init_value=[1]):
        super().__init__()
        self.weight = Parameter(torch.Tensor(init_value))
        # pdb.set_trace()

    def forward(self, input):
        weight = self.weight[None, :, None, None]
        return input * weight

class DenseBasicBlock(nn.Module):
    def __init__(self, inplanes, filters, index, expansion=1, growthRate=12, dropRate=0):
        super(DenseBasicBlock, self).__init__()
        planes = expansion * growthRate

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(filters, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out

class SparseDenseBasicBlock(nn.Module):
    def __init__(self, inplanes, filters, index, expansion=1, growthRate=12, dropRate=0):
        super(SparseDenseBasicBlock, self).__init__()
        planes = expansion * growthRate

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(filters, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

        m = Uniform(torch.tensor([norm_mean]*filters), torch.tensor([norm_var]*filters)).sample()

        self.mask = Mask(m)
        self.select = ChannelSelection(index)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.select(out)
        out = self.mask(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, inplanes, outplanes, filters, index):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(filters, outplanes, kernel_size=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out

class SparseTransition(nn.Module):
    def __init__(self, inplanes, outplanes, filters, index):
        super(SparseTransition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = ChannelSelection(index)
        self.conv1 = nn.Conv2d(filters, outplanes, kernel_size=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        m = Normal(torch.tensor([norm_mean]*filters), torch.tensor([norm_var]*filters)).sample()
        self.mask = Mask(m)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.select(out)
        out = self.mask(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet(nn.Module):

    def __init__(self, depth=40, block=DenseBasicBlock, 
        dropRate=0, num_classes=10, growthRate=12, compressionRate=2, filters=None, indexes=None):
        super(DenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 3 if 'DenseBasicBlock' in str(block) else (depth - 4) // 6
        transition = SparseTransition if 'Sparse' in str(block) else Transition
        if filters == None:
            filters = []
            start = growthRate*2
            for i in range(3):
                filters.append([start + growthRate*i for i in range(n+1)])
                start = (start + growthRate*n) // compressionRate
            filters = [item for sub_list in filters for item in sub_list]

            indexes = []
            for f in filters:
                indexes.append(np.arange(f))

        self.growthRate = growthRate
        self.dropRate = dropRate

        self.inplanes = growthRate * 2 
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_denseblock(block, n, filters[0:n], indexes[0:n])
        self.trans1 = self._make_transition(transition, compressionRate, filters[n], indexes[n])
        self.dense2 = self._make_denseblock(block, n, filters[n+1:2*n+1], indexes[n+1:2*n+1])
        self.trans2 = self._make_transition(transition, compressionRate, filters[2*n+1], indexes[2*n+1])
        self.dense3 = self._make_denseblock(block, n, filters[2*n+2:3*n+2], indexes[2*n+2:3*n+2])
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.inplanes, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks, filters, indexes):
        layers = []
        assert blocks == len(filters), 'Length of the filters parameter is not right.'
        assert blocks == len(indexes), 'Length of the indexes parameter is not right.'
        for i in range(blocks):
            # print("denseblock inplanes", filters[i])
            layers.append(block(self.inplanes, filters=filters[i], index=indexes[i], growthRate=self.growthRate, dropRate=self.dropRate))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, transition, compressionRate, filters, index):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return transition(inplanes, outplanes, filters, index)


    def forward(self, x):
        x = self.conv1(x)

        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def densenet_40(pretrained=False, **kwargs):
    return DenseNet(depth=40, block=DenseBasicBlock, compressionRate=1, **kwargs)

def densenet_40_sparse(pretrained=False,**kwargs):
    return DenseNet(depth=40, block=SparseDenseBasicBlock, compressionRate=1, **kwargs)
