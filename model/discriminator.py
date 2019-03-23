import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()
        self.filters = [num_classes, 128, 256, 128]
        block = [
            nn.Linear(self.filters[i], self.filters[i+1]) \
            for i in range(3)
        ]
        self.body = nn.Sequential(*block)

        self._initialize_weights()

    def forward(self, input):
        x = self.body(input)
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