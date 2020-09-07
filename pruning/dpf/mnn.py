import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Masker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        return x * mask

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class MaskConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(MaskConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias, padding_mode)
        self.mask = Parameter(torch.ones(self.weight.size()), requires_grad=False)

    def forward(self, input):
        masked_weight = Masker.apply(self.weight, self.mask)
        return super(MaskConv2d, self)._conv_forward(input, masked_weight)