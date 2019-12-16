import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import pdb
# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
        #    'resnet152']
norm_mean, norm_var = 0.0, 1.0

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Mask(nn.Module):
    def __init__(self, init_value=[1.0], planes=None):
        super(Mask, self).__init__()
        self.planes = planes
        self.weight = Parameter(torch.Tensor(init_value))

    def forward(self, input):
        weight = self.weight

        if self.planes is not None:
            weight = self.weight[None, :, None, None]

        return input * weight

class ResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, has_mask=None, stride=1, downsample=None):
        super(ResBasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SparseResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, has_mask=None, stride=1, downsample=None):
        super(SparseResBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.inplanes = inplanes
        self.planes = planes
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.has_mask = has_mask
        m = Normal(torch.tensor([0.0]), torch.tensor([1.0])).sample()
        self.init_value = m
        self.mask = Mask(self.init_value)
        # self.mask = Mask()

    def forward(self, x):
        residual = x

        out = torch.zeros_like(residual)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, has_mask=None, cfg=None, index=None, stride=1, downsample=None):
        super(ResBottleneck, self).__init__()
        # print("inplanes",inplanes, "planes",planes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SparseResBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, has_mask=None, cfg=None, index=None, stride=1, downsample=None):
        super(SparseResBottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.has_mask = has_mask
        m = Normal(torch.tensor([norm_mean]), torch.tensor([norm_var])).sample()
        self.init_value = m
        self.mask = Mask(self.init_value)
        # self.mask = Mask()

    def forward(self, x):
        residual = x

        # print('input size', x.size())
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # sparse mask
        out = self.mask(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class MultiSparseResBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, has_mask=None, cfg=None, index=None, stride=1, downsample=None):
        # print('in multisparse',cfg)
        super(MultiSparseResBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.has_mask = has_mask
        m = Normal(torch.tensor([norm_mean]), torch.tensor([norm_var])).sample()
        self.mask = Mask(m)
        m1 = Normal(torch.tensor([norm_mean]*cfg[0]), torch.tensor([norm_var]*cfg[0])).sample()
        m2 = Normal(torch.tensor([norm_mean]*cfg[1]), torch.tensor([norm_var]*cfg[1])).sample()
        m3 = Normal(torch.tensor([norm_mean]*cfg[2]), torch.tensor([norm_var]*cfg[2])).sample()
        self.select = channel_selection(index)
        self.mask_channel1 = Mask(m1, planes=cfg[0])
        self.mask_channel2 = Mask(m2, planes=cfg[1])
        self.mask_channel3 = Mask(m3, planes=cfg[2])


    def forward(self, x):
        residual = x
        # out = x
        out = self.select(x)
        out = self.mask_channel1(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.mask_channel2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.mask_channel3(out)
        out = self.conv3(out)
        out = self.bn3(out)

            # sparse mask
        out = self.mask(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Downsample(nn.Module):
    def __init__(self, downsample):
        super(Downsample, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.downsample(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, has_mask=None, indexes=None, cfg=None):
        self.inplanes = 64
        super(ResNet, self).__init__()

        if has_mask is None : has_mask = [1]*sum(layers)

        if cfg is None:
            # bottleneck 
            cfg =[[64, 64, 64], [256, 64, 64]*(layers[0]-1), [256, 128, 128], [512, 128, 128]*(layers[1]-1), [512, 256, 256], [1024, 256, 256]*(layers[2]-1), [1024, 512, 512], [2048, 512, 512]*(layers[3]-1), [2048]]
            cfg = [item for sub_list in cfg for item in sub_list]
        
        if indexes is None:
            indexes = []
            for i in range(len(cfg)):
                indexes.append(np.arange(cfg[i]))

        start = 0
        cfg_start = 0
        cfg_end = 3*layers[0]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        remain_block = [int(m) for m in np.argwhere(np.array(has_mask))]
        layers_start = [0, 3, 7, 13]

        layers_remain = []
        for i in range(3):
            # print(len(np.where((np.array(remain_block)>=layers_start[i]) & (np.array(remain_block)<layers_start[i+1]))[0]))
            layers_remain.append(len((np.where((np.array(remain_block)>=layers_start[i]) & (np.array(remain_block)<layers_start[i+1]))[0])))
        layers_remain.append(len(np.where(np.array(remain_block) >= layers_start[3])[0]))

        # print("remain layers",layers_remain)


        self.layer1 = self._make_layer(block, 64, layers[0], has_mask=has_mask[start:layers[0]], cfg=cfg[cfg_start:cfg_end], indexes=indexes[0:cfg_end])

        start = layers[0]
        cfg_start += 3*layers[0]
        cfg_end += 3*layers[1]

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, has_mask=has_mask[start:start+layers[1]], cfg=cfg[cfg_start:cfg_end], indexes=indexes[cfg_start:cfg_end])

        start += layers[1]
        cfg_start += 3*layers[1]
        cfg_end += 3*layers[2]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, has_mask=has_mask[start:start+layers[2]], cfg=cfg[cfg_start:cfg_end], indexes=indexes[cfg_start:cfg_end])

        start += layers[2]
        cfg_start += 3*layers[2]
        cfg_end += 3*layers[3]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,has_mask=has_mask[start:start+layers[3]], cfg=cfg[cfg_start:cfg_end], indexes=indexes[cfg_start:cfg_end])

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, has_mask=None, cfg=None, indexes=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if has_mask[0] == 0 and downsample is not None:
            layers.append(Downsample(downsample))
        elif not has_mask[0] == 0:
            layers.append(block(self.inplanes, planes, has_mask[0], cfg[0:3], indexes[0], stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if not has_mask[i] == 0:
                layers.append(block(self.inplanes, planes, has_mask[i], cfg[3*i:3*(i+1)], indexes[3*i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # 256 x 56 x 56
        x = self.layer2(x)
        # 512 x 28 x 28
        # 1024 x 14 x 14
        x = self.layer3(x)
        # 2048 x 7 x 7
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet_18(pretrained=False, **kwargs):
    model = ResNet(ResBasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        load_url = model_zoo.load_url(model_urls['resnet18'])
        model.load_state_dict(load_url)
        return model, load_url
    return model

def resnet_18_sparse(pretrained=False, **kwargs):
    model = ResNet(SparseResBasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet_50(pretrained=False, **kwargs):
    model = ResNet(ResBottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        load_url = model_zoo.load_url(model_urls['resnet50'])
        model.load_state_dict(load_url)
        return model, load_url
    return model

def resnet_50_sparse(pretrained=False, **kwargs):
    model = ResNet(SparseResBottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet_50_multisparse(pretrained=False, **kwargs):
    model = ResNet(MultiSparseResBottleneck, [3, 4, 6, 3], **kwargs)
    return model

