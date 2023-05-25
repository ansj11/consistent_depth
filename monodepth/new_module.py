import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import collections
from itertools import repeat

class Swish(nn.Module):
    dump_patches = True
    def __init__(self):
        super(Swish, self).__init__()
        self.register_buffer('k', torch.zeros(12))
        self.register_buffer('b', torch.zeros(12))
        self.exp_points()
    def forward(self, x):
        x = F.hardtanh(x, -4, 4)
        x_ = torch.unsqueeze(x, 4)
        x_seg = (-x_)*Variable(self.k, requires_grad=False)+ Variable(self.b, requires_grad=False)
        x_exp, _ = torch.max(x_seg, dim=4)
        out = x/(1 + x_exp) + 0.2785
        out = F.hardtanh(out, -4, 4, inplace=False)
        out.data[:] = torch.round(out.data[:] * (255. / 4.)) * (4. / 255.)
        return out
    def exp_points(self):
        t = np.arange(-6., 7., 1.)
        y_p = 2.**t
        y_p_diff = y_p[1:] - y_p[:-1]
        b = y_p[1:] - y_p_diff*t[1:]
        k = y_p_diff/np.log(2)
        self.k, self.b = (torch.from_numpy(k)).float(), \
                         (torch.from_numpy(b)).float()

class AvgQuant(nn.Module):
    dump_patches = True
    def __init__(self):
        super(AvgQuant, self).__init__()
    def forward(self, x1, x2):
        out = (x1 + x2)/2
        #out.data[:] = torch.floor(out.data[:] * (255. / 4.)) * (4. / 255.)
        return out


class HardQuant(nn.Module):
    def __init__(self, left_value, right_value):
        super(HardQuant, self).__init__()
        self.l = left_value
        self.r = right_value
        self.delta = self.r - self.l
    def forward(self, x):
        out = F.hardtanh(x, self.l, self.r)
        out.data[:] = torch.round(out.data[:] * (255. / self.delta)) * (self.delta / 255.)
        return out

class Reorg(nn.Module):
    dump_patches = True 
    def __init__(self):
        super(Reorg, self).__init__()
    
    def forward(self, x):
        ss = x.size()
        out = x.view(ss[0], ss[1], ss[2] // 2, 2, ss[3]).view(ss[0], ss[1], ss[2] // 2, 2, ss[3] // 2, 2).\
                permute(0, 1, 3, 5, 2, 4).contiguous().view(ss[0], -1, ss[2] // 2, ss[3] // 2)
        return out


class UpsampleQuant(nn.Upsample):
    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        super(UpsampleQuant, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.maxv = 4

    def forward(self, x):
        if hasattr(self, 'align_corners'):
            out = F.upsample(x, self.size, self.scale_factor, self.mode, True)                                 
        else:
            out = F.upsample(x, self.size, self.scale_factor, self.mode)
            #out.data[:] = torch.round(out.data[:] * (255. / self.maxv)) * (self.maxv / 255.)
        return out

class AdaptiveAvgPool(nn.AdaptiveAvgPool2d):    
    dump_patches = True
    def __init__(self, output_size):
        super(AdaptiveAvgPool, self).__init__(output_size)
    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, self.output_size)
        #out.data[:] = torch.floor(out.data[:] * (255. / 4.)) * (4. / 255.)
        return out


class AvgPool(nn.AvgPool2d):    
    dump_patches = True
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                                         count_include_pad=True):
        super(AvgPool, self).__init__(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,                                                                                   count_include_pad=count_include_pad)
    def forward(self, x):
        out = F.avg_pool2d(x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)
        #out.data[:] = torch.floor(out.data[:] * (255. / 4.)) * (4. / 255.)
        return out


class SliceMul(nn.Module):
    """
    x1: NxCxHxW feature map
    x2: NxCx1x1 SE layer output
    """
    def __init__(self, weight_quant):
        super(SliceMul, self).__init__()
        self.weight_quant = weight_quant

    def forward(self, x1, x2):
        out = x1 * x2
        if self.weight_quant:
            out.data[:] = torch.floor(out.data[:] * (255. / 4.)) * (4. / 255.)
        return out


class PixelMul(nn.Module):
    """
    x1: NxCxHxW feature map
    x2: Nx1xHxW SE2 feature map
    """ 
    def __init__(self, weight_quant):
        super(PixelMul, self).__init__()
        self.weight_quant = weight_quant

    def forward(self, x1, x2):
        out = x1 * x2
        if self.weight_quant:
            out.data[:] = torch.floor(out.data[:] * (255. / 4.)) * (4. / 255.)
        return out


class AvgQuant(nn.Module):
    dump_patches = True

    def __init__(self):
        super(AvgQuant, self).__init__()

    def forward(self, x1, x2):
        out = (x1 + x2)/2
        out.data[:] = torch.floor(out.data[:] * (255. / 4.)) * (4. / 255.)
        return out


class HardQuant(nn.Module):
    def __init__(self, left_value, right_value):
        super(HardQuant, self).__init__()
        self.l = left_value
        self.r = right_value
        self.delta = self.r - self.l

    def forward(self, x):
        out = F.hardtanh(x, self.l, self.r)
        #out.data[:] = torch.round(out.data[:] * (255. / self.delta)) * (self.delta / 255.)
        return out


def conv_layer(channel_in, channel_out, ks=1, stride=1, padding=0, dilation=1, bias=False, bn=True, cut=True, relu=True,
               group=1, weight_quant=True):
    if weight_quant:
        _conv = Conv2dQuant
    else:
        _conv = nn.Conv2d
    sequence = [_conv(channel_in, channel_out, kernel_size=ks, stride=stride, padding=padding, dilation=dilation,
                bias=bias, groups=group)]
    if bn:
        sequence.append(nn.BatchNorm2d(channel_out))
    if relu:
        if cut:
            sequence.append(HardQuant(0, 4))
        else:
            sequence.append(nn.ReLU())
    print('computation:', channel_in, channel_out, ks, stride)
    return nn.Sequential(*sequence)


def linear_layer(channel_in, channel_out, bias=False, bn=True, relu=True, unit=False, weight_quant=True):
    if not unit:
        if weight_quant:
            _linear = LinearQuant
        else:
            _linear = nn.Linear
        sequence = [_linear(channel_in, channel_out, bias=bias)]
    else:
        sequence = [LinearUnit(channel_in, channel_out, bias=bias)]
    if bn:
        sequence.append(nn.BatchNorm1d(channel_out))
    if relu:
        sequence.append(HardQuant(0, 4))
    return nn.Sequential(*sequence)


class mobile_unit(nn.Module):
    dump_patches = True

    def __init__(self, channel_in, channel_out, stride=1, has_half_out=False, num3x3=2):
        print('Making SceneModel!')
        super(mobile_unit, self).__init__()
        self.stride = stride
        self.channel_in = channel_in
        self.channel_out = channel_out
        if num3x3 == 1:
            self.conv3x3 = nn.Sequential(
                conv_layer(channel_in, channel_in, ks=3, stride=stride, padding=1, group=channel_in),
            )
        else:
            self.conv3x3 = nn.Sequential(
                conv_layer(channel_in, channel_in, ks=3, stride=1, padding=1, group=channel_in),
                conv_layer(channel_in, channel_in, ks=3, stride=stride, padding=1, group=channel_in),
            )
        self.conv1x1 = conv_layer(channel_in, channel_out)
        self.has_half_out = has_half_out
        self.avg = AvgQuant()

    def forward(self, x):
        half_out = self.conv3x3(x)
        out = self.conv1x1(half_out)
        if self.stride == 1 and (self.channel_in == self.channel_out):
            out = self.avg(out, x)
        if self.has_half_out:
            return half_out, out
        else:
            return out


class Conv2dQuant(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dQuant, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                          groups=groups, bias=bias)
        self.register_buffer('weight_copy', torch.zeros(self.weight.size()))
        self.computation = 0

    def get_computation(self, x):
        feature_size = x.size()
        return feature_size[-1] * feature_size[-2] * self.kernel_size[0] ** 2 * self.in_channels * self.out_channels / (self.groups * 2 ** 20 * self.stride[0] ** 2)

    def forward(self, x):
        self.computation = self.get_computation(x)
        self.weight_copy[:] = self.weight.data[:]
        weight_max = \
            torch.abs(self.weight_copy).max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3,
                                                                                                    keepdim=True)[0]
        self.weight.data[:] = torch.round((self.weight.data[:] / (weight_max + 1e-12)) * 127) * weight_max / 127
        out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self.weight.data[:] = self.weight_copy[:]
        return out


class LinearQuant(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearQuant, self).__init__(in_features, out_features, bias=bias)
        self.register_buffer('weight_copy', torch.zeros(self.weight.size()))

    def forward(self, input):
        self.weight_copy[:] = self.weight.data[:]
        weight_max = \
            torch.abs(self.weight_copy).max(dim=1, keepdim=True)[0]
        self.weight.data[:] = torch.round((self.weight.data[:] / weight_max) * 127) * weight_max / 127
        out = F.linear(input, self.weight, self.bias)
        self.weight.data[:] = self.weight_copy[:]
        return out


class Conv2dUnit(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dUnit, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups=groups, bias=bias)

    def forward(self, input):
        weight_norm = torch.sqrt(
            (self.weight ** 2).sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True))
        weight_unit = self.weight/weight_norm
        out = F.conv2d(input, weight_unit, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out


class LinearUnit(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearUnit, self).__init__(in_features, out_features, bias=bias)
        self.weight_unit = None

    def forward(self, input):
        weight_norm = torch.sqrt((self.weight ** 2).sum(dim=1, keepdim=True))
        self.weight_unit = self.weight / weight_norm
        out = F.linear(input, self.weight_unit, self.bias)
        return out
