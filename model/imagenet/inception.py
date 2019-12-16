import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal
import pdb

__all__ = ['Inception3', 'inception_v3', 'inception_v3_sparse']


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}

norm_mean, norm_var = 0.0, 1.0

class Inception3(nn.Module):

    def __init__(self, is_sparse=False, num_classes=1000, aux_logits=False, transform_input=True, has_mask=None, filters=None):
        super(Inception3, self).__init__()

        # [inchanel, branch1x1, branch5x5, branch3x3, branch_pool]
        if filters is None:
            filters = [
                [64, 64, 96, 32], # A
                [64, 64, 96, 64], # A
                [64, 64, 96, 64], # A
                [384, 96, 288, 0], # B
                [192, 192, 192, 192], # C
                [192, 192, 192, 192], # C
                [192, 192, 192, 192], # C
                [192, 192, 192, 192], # C
                [320, 192, 768, 0], # D
                [320, 768, 768, 192], # E
                [320, 768, 768, 192], # E
            ]

        if has_mask is None: 
            has_mask = [1]*40


        if is_sparse:
            inceptionA = SparseInceptionA
            inceptionB = SparseInceptionB
            inceptionC = SparseInceptionC
            inceptionD = SparseInceptionD
            inceptionE = SparseInceptionE
        else:
            inceptionA = InceptionA
            inceptionB = InceptionB
            inceptionC = InceptionC
            inceptionD = InceptionD
            inceptionE = InceptionE
            

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)

        self.Mixed_5b = inceptionA(192, pool_features=32, has_mask=has_mask[0:4])
        self.Mixed_5c = inceptionA(sum(filters[0]), pool_features=64, has_mask=has_mask[4:8])
        self.Mixed_5d = inceptionA(sum(filters[1]), pool_features=64, has_mask=has_mask[8:12])
        self.Mixed_6a = inceptionB(sum(filters[2]), has_mask=has_mask[12:14])
        self.Mixed_6b = inceptionC(sum(filters[3]), channels_7x7=128, has_mask=has_mask[14:18])
        self.Mixed_6c = inceptionC(sum(filters[4]), channels_7x7=160, has_mask=has_mask[18:22])
        self.Mixed_6d = inceptionC(sum(filters[5]), channels_7x7=160, has_mask=has_mask[22:26])
        self.Mixed_6e = inceptionC(sum(filters[6]), channels_7x7=192, has_mask=has_mask[26:30])
        if aux_logits:
            self.AuxLogits = InceptionAux(sum(filters[6]), num_classes)
        self.Mixed_7a = inceptionD(sum(filters[7]), has_mask=has_mask[30:32])
        self.Mixed_7b = inceptionE(sum(filters[8]), has_mask=has_mask[32:36])
        self.Mixed_7c = inceptionE(sum(filters[9]), has_mask=has_mask[36:40])
        self.fc = nn.Linear(sum(filters[10]), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.numel()))
                values = values.view(m.weight.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # print('after Mixed_6a', x.size())
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
        if self.training and self.aux_logits:
            return x, aux
        return x


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, has_mask):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        # print("A: branch1x1", branch1x1.shape[1])
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        # print("A: branch5x5", branch5x5.shape[1])
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        # print("A: branch3x3dbl", branch3x3dbl.shape[1])
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        # print("A: branch_pool", branch_pool.shape[1])
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class SparseInceptionA(InceptionA):
    def __init__(self, in_channels, pool_features, has_mask):
        super(SparseInceptionA, self).__init__(in_channels, pool_features, has_mask)
        m1 = Normal(torch.tensor([norm_mean]), torch.tensor([norm_var])).sample()
        m2 = Normal(torch.tensor([norm_mean]), torch.tensor([norm_var])).sample()
        m3 = Normal(torch.tensor([norm_mean]), torch.tensor([norm_var])).sample()
        m4 = Normal(torch.tensor([norm_mean]), torch.tensor([norm_var])).sample()
        self.mask1 = Mask(m1)
        self.mask2 = Mask(m2)
        self.mask3 = Mask(m3)
        self.mask4 = Mask(m4)
        self.has_mask = has_mask

    def forward(self, x):
        out = []

        if self.has_mask[0] > 0:
            branch1x1 = self.branch1x1(x)
            branch1x1 = self.mask1(branch1x1)
            out.append(branch1x1)
            
        if self.has_mask[1] > 0:
            branch5x5 = self.branch5x5_1(x)
            branch5x5 = self.branch5x5_2(branch5x5)
            branch5x5 = self.mask2(branch5x5)
            out.append(branch5x5)
            
        if self.has_mask[2] > 0:
            branch3x3dbl = self.branch3x3dbl_1(x)
            branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
            branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
            branch3x3dbl = self.mask3(branch3x3dbl)
            out.append(branch3x3dbl)
            
        if self.has_mask[3] > 0:
            branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
            branch_pool = self.branch_pool(branch_pool)
            branch_pool = self.mask4(branch_pool)
            out.append(branch_pool)

        # pdb.set_trace()
        return torch.cat(out, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels, has_mask):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        # print("B: branch3x3", branch3x3.shape[1])
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        # print("B: branch3x3dbl", branch3x3dbl.shape[1])
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        # print("B: branch_pool", branch_pool.shape[1])
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class SparseInceptionB(InceptionB):

    def __init__(self, in_channels, has_mask):
        super(SparseInceptionB, self).__init__(in_channels, has_mask)
        m1 = Normal(torch.tensor([norm_mean]), torch.tensor([norm_var])).sample()
        m2 = Normal(torch.tensor([norm_mean]), torch.tensor([norm_var])).sample()
        self.mask1 = Mask(m1)
        self.mask2 = Mask(m2)
        self.has_mask = has_mask
        # print('SparseIncpetionB',self.has_mask)

    def forward(self, x):
        out = []
        if self.has_mask[0] > 0:
            branch3x3 = self.branch3x3(x)
            branch3x3 = self.mask1(branch3x3)
            # print("B: branch3x3", branch3x3.shape[1])
            out.append(branch3x3)

        if self.has_mask[1] > 0:
            branch3x3dbl = self.branch3x3dbl_1(x)
            branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
            branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
            branch3x3dbl = self.mask2(branch3x3dbl)
            # print("B: branch3x3dbl", branch3x3dbl.shape[1])
            out.append(branch3x3dbl)

        # if self.has_mask[2] > 0:
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        # branch_pool = self.mask3(branch_pool)
        # print("B: branch_pool", branch_pool.shape[1])
        out.append(branch_pool)

        # pdb.set_trace()
        return torch.cat(out, 1)

class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7, has_mask):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
        
        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        
        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        # print("C: branch1x1", branch1x1.shape[1])
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        # print("C: branch7x7", branch7x7.shape[1])
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        # print("C: branch7x7dbl", branch7x7dbl.shape[1])
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        # print("C: branch_pool", branch_pool.shape[1])
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)

class SparseInceptionC(InceptionC):
    def __init__(self, in_channels, channels_7x7, has_mask):
        super(SparseInceptionC, self).__init__(in_channels, channels_7x7, has_mask)
        
        m1 = Normal(torch.tensor([norm_mean]), torch.tensor([norm_var])).sample()
        m2 = Normal(torch.tensor([norm_mean]), torch.tensor([norm_var])).sample()
        m3 = Normal(torch.tensor([norm_mean]), torch.tensor([norm_var])).sample()
        m4 = Normal(torch.tensor([norm_mean]), torch.tensor([norm_var])).sample()
        self.mask1 = Mask(m1)
        self.mask2 = Mask(m2)
        self.mask3 = Mask(m3)
        self.mask4 = Mask(m4)

        self.has_mask = has_mask

    def forward(self, x):
        out = []
        if self.has_mask[0] > 0:
            branch1x1 = self.branch1x1(x)
            branch1x1 = self.mask1(branch1x1)
            out.append(branch1x1)

        if self.has_mask[1] > 0:
            branch7x7 = self.branch7x7_1(x)
            branch7x7 = self.branch7x7_2(branch7x7)
            branch7x7 = self.branch7x7_3(branch7x7)
            branch7x7 = self.mask2(branch7x7)
            out.append(branch7x7)

        if self.has_mask[2] > 0:
            branch7x7dbl = self.branch7x7dbl_1(x)
            branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
            branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
            branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
            branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
            branch7x7dbl = self.mask3(branch7x7dbl)
            out.append(branch7x7dbl)

        if self.has_mask[3] > 0:
            branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
            branch_pool = self.branch_pool(branch_pool)
            branch_pool = self.mask4(branch_pool)
            out.append(branch_pool)

        # pdb.set_trace()
        return torch.cat(out, 1)

class InceptionD(nn.Module):

    def __init__(self, in_channels, has_mask):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        # print("D: branch3x3", branch3x3.shape[1])
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        # print("D: branch7x7x3", branch7x7x3.shape[1])
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        # print("D: branch_pool", branch_pool.shape[1])
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class SparseInceptionD(InceptionD):

    def __init__(self, in_channels, has_mask):
        super(SparseInceptionD, self).__init__(in_channels, has_mask)
        m1 = Normal(torch.tensor([norm_mean]), torch.tensor([norm_var])).sample()
        m2 = Normal(torch.tensor([norm_mean]), torch.tensor([norm_var])).sample()
        self.mask1 = Mask(m1)
        self.mask2 = Mask(m2)
        # self.mask3 = Mask()
        self.has_mask = has_mask

    def forward(self, x):
        out = []
        if self.has_mask[0] > 0:
            branch3x3 = self.branch3x3_1(x)
            branch3x3 = self.branch3x3_2(branch3x3)
            branch3x3 = self.mask1(branch3x3)
            out.append(branch3x3)

        if self.has_mask[1] > 0:
            branch7x7x3 = self.branch7x7x3_1(x)
            branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
            branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
            branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
            branch7x7x3 = self.mask2(branch7x7x3)
            out.append(branch7x7x3)

        # if self.has_mask[2] > 0:
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        out.append(branch_pool)

        # pdb.set_trace()
        return torch.cat(out, 1)

class InceptionE(nn.Module):

    def __init__(self, in_channels, has_mask):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        # print("E: branch1x1", branch1x1.shape[1])
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)
        # print("E: branch3x3", branch3x3.shape[1])
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        # print("E: branch3x3dbl", branch3x3dbl.shape[1])
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        # print("E: branch_pool", branch_pool.shape[1])
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class SparseInceptionE(InceptionE):

    def __init__(self, in_channels, has_mask):
        super(SparseInceptionE, self).__init__(in_channels, has_mask)
        m1 = Normal(torch.tensor([norm_mean]), torch.tensor([norm_var])).sample()
        m2 = Normal(torch.tensor([norm_mean]), torch.tensor([norm_var])).sample()
        m3 = Normal(torch.tensor([norm_mean]), torch.tensor([norm_var])).sample()
        m4 = Normal(torch.tensor([norm_mean]), torch.tensor([norm_var])).sample()
        self.mask1 = Mask(m1)
        self.mask2 = Mask(m2)
        self.mask3 = Mask(m3)
        self.mask4 = Mask(m4)
        
        self.has_mask = has_mask

    def forward(self, x):
        out = []
        if self.has_mask[0] > 0:
            branch1x1 = self.branch1x1(x)
            branch1x1 = self.mask1(branch1x1)
            out.append(branch1x1)

        if self.has_mask[1] > 0:
            branch3x3 = self.branch3x3_1(x)
            branch3x3 = [
                self.branch3x3_2a(branch3x3),
                self.branch3x3_2b(branch3x3),
            ]
            branch3x3 = torch.cat(branch3x3, 1)
            branch3x3 = self.mask2(branch3x3)
            out.append(branch3x3)

        if self.has_mask[2] > 0:
            branch3x3dbl = self.branch3x3dbl_1(x)
            branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
            branch3x3dbl = [
                self.branch3x3dbl_3a(branch3x3dbl),
                self.branch3x3dbl_3b(branch3x3dbl),
            ]
            branch3x3dbl = torch.cat(branch3x3dbl, 1)
            branch3x3dbl = self.mask3(branch3x3dbl)
            out.append(branch3x3dbl)

        # pdb.set_trace()

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        if self.has_mask[3] > 0:
            branch_pool = self.branch_pool(branch_pool)
            branch_pool = self.mask4(branch_pool)
            out.append(branch_pool)

        return torch.cat(out, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Mask(nn.Module):
    def __init__(self, init_value=[1]):
        super().__init__()
        self.weight = Parameter(torch.Tensor(init_value))

    def forward(self, input):
        return input * self.weight


def inception_v3(pretrained=False, **kwargs):
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        model = Inception3(**kwargs)
        load_url = model_zoo.load_url(model_urls['inception_v3_google'])
        new_state_dict= {}
        for k, v in load_url.items():
            if 'AuxLogits' not in k:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        return model, new_state_dict

    return Inception3(**kwargs)

def inception_v3_sparse(pretrained=False, **kwargs):
    return Inception3(is_sparse=True, **kwargs)


# def test():
#     index = [1, 3, 5, 10, 25]
#     mask = np.ones(42)
#     mask[index] = 0
#     net = inception_v3_sparse(has_mask=mask)
#     input = Variable(torch.randn(3,3,299,299))
#     logits = net(input)
#     print(logits.size())

# test()

