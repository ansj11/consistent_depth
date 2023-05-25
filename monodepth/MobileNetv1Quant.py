import sys
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable, Function
import numpy as np
from torch.nn.parameter import Parameter
from new_module import conv_layer, AvgQuant, mobile_unit, linear_layer, SliceMul, PixelMul, AdaptiveAvgPool
from config import *
from torchvision import models, transforms
import numpy as np
from IPython import embed
import new_module

weight_quant = False

def se_layer(channel, reduction=4):
    return nn.Sequential(
        AdaptiveAvgPool(1),
        nn.Conv2d(channel, channel//reduction, 1, 1, 0, bias=False),
        nn.BatchNorm2d(channel//reduction),
        nn.ReLU(inplace=True),
        nn.Conv2d(channel//reduction, channel, 1, 1, 0, bias=False),
        nn.BatchNorm2d(channel),
        nn.Sigmoid())

def se2_layer(channel, reduction=4):
    return nn.Sequential(
        nn.Conv2d(channel, channel//reduction, 1, 1, 0, bias=False),
        nn.BatchNorm2d(channel//reduction),
        nn.ReLU(inplace=True),
        nn.Conv2d(channel//reduction, 1, 1, 1, 0, bias=False),
        nn.BatchNorm2d(1),
        nn.Sigmoid())

def conv_bn_1x1(inp, oup, stride):
    return nn.Sequential(
        conv_layer(inp, oup, ks=1, stride=stride, padding=0, cut=False, weight_quant=weight_quant),
        #nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        #nn.BatchNorm2d(oup),
        #nn.ReLU(inplace=True)
        #HardQuant(0, 4)
    )

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        conv_layer(inp, oup, ks=3, stride=stride, padding=1, cut=False, weight_quant=weight_quant),
        #nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        #nn.BatchNorm2d(oup),
        #nn.ReLU(inplace=True)
        #HardQuant(0, 4)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        conv_layer(inp, inp, ks=3, stride=stride, padding=1, group=inp, cut=False, weight_quant=weight_quant),
        #nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        #nn.BatchNorm2d(inp),
        #nn.ReLU(inplace=True),

        conv_layer(inp, oup, ks=1, stride=1, padding=0, cut=False, weight_quant=weight_quant),
        #nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        #nn.BatchNorm2d(oup),
        #nn.ReLU(inplace=True),
    )

def smooth(inp, oup, stride):
    return nn.Sequential(
        conv_layer(inp, inp, ks=3, stride=stride, padding=1, group=inp, bias=False, bn=True, cut=False, weight_quant=weight_quant),
        #nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        #nn.BatchNorm2d(inp),
        #nn.ReLU(inplace=True),

        conv_layer(inp, oup, ks=1, stride=1, padding=0, cut=False, weight_quant=weight_quant),
        #nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        #nn.BatchNorm2d(oup),
        #nn.ReLU(inplace=True),
    )


def predict(in_planes, out_planes):
    return nn.Sequential(
        conv_layer(in_planes, out_planes, ks=3, stride=1, padding=1, bias=True, bn=False, cut=False, weight_quant=weight_quant),
        #nn.Conv2d(in_planes, out_planes, 3, 1, 1, bias=True),
        #nn.ReLU(inplace=True)
        )

def get_normal(in_planes, out_planes):
    return nn.Sequential(
        conv_layer(in_planes, out_planes, ks=1, stride=1, padding=0, bias=True, bn=False, relu=False, weight_quant=weight_quant),
        nn.Tanh()
        #nn.Conv2d(in_planes, out_planes, 3, 1, 1, bias=True),
        #nn.ReLU(inplace=True)
        )
def get_semantic(in_planes, out_planes):
    return nn.Sequential(
        conv_layer(in_planes, out_planes, ks=3, stride=1, padding=1, bias=False, bn=False, relu=False, weight_quant=weight_quant),
        nn.Softmax(1)
    )


def get_confidence(in_planes, out_planes):
    return nn.Sequential(
        conv_layer(in_planes, out_planes, ks=3, stride=1, padding=1, bias=False, bn=False, relu=False, weight_quant=weight_quant),
        nn.Sigmoid()
    )

def upshuffle(in_planes, out_planes, upscale_factor):
    return nn.Sequential(
        conv_layer(in_planes, out_planes*upscale_factor**2, ks=3, stride=1, padding=1, bias=False, bn=True, relu=False),
        nn.PixelShuffle(upscale_factor),
        nn.ReLU(inplace=True)
    )
class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.conv1 = conv_layer(channel, channel//reduction, ks=1, padding=0, relu=True, bn=True, cut=False, weight_quant=weight_quant)
        self.conv2 = conv_layer(channel//reduction, channel, ks=1, padding=0, relu=False, bn=True, weight_quant=weight_quant)
        self.pool = AdaptiveAvgPool(1)
        self.sigmod = nn.Sigmoid()
        self.slicemul = SliceMul(weight_quant=weight_quant)
        self.avgquant = AvgQuant()

    def forward(self, x):
        out = self.pool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        se_attention = self.sigmod(out)
        #return self.avgquant(x, self.slicemul(x, se_attention))
        return x + self.slicemul(x, se_attention)


class Net(nn.Module):
    def __init__(self, output_channel=1, semantic_channel=2, rate=1):
        super(Net, self).__init__()
        self.output_channel = output_channel
        self.semantic_channel = semantic_channel #semantic_channel is the total categories
        c = [i//rate for i in [32, 64, 128, 256, 512, 1024]]
        self.channels = c

        self.layer0 = nn.Sequential( conv_bn(3, c[0], 2),
                                     conv_dw(c[0], c[1], 1))
        self.layer1 = nn.Sequential( conv_dw(c[1], c[2], 2),
                                     conv_dw(c[2], c[2], 1))
        self.layer2 = nn.Sequential( conv_dw(c[2], c[3], 2),
                                     conv_dw(c[3], c[3], 1))
        self.layer3 = nn.Sequential( conv_dw(c[3], c[4], 2),
                                     conv_dw(c[4], c[4], 1),
                                     conv_dw(c[4], c[4], 1),
                                     conv_dw(c[4], c[4], 1),
                                     conv_dw(c[4], c[4], 1),
                                     conv_dw(c[4], c[4], 1))
        self.layer4 = nn.Sequential( conv_dw(c[4], c[5], 2),
                                     conv_dw(c[5], c[5], 1))    # top layer

        #self.top = nn.Sequential(AdaptiveAvgPool(1),
        #                         conv_bn_1x1(c[5], c[5], 1),
        #                         conv_bn_1x1(c[5], c[5], 1)
        #                        )

        # Smooth layers
        self.smooth1 = smooth(c[5], c[4], 1)
        self.smooth2 = smooth(c[4], c[3], 1)
        self.smooth3 = smooth(c[3], c[2], 1)
        self.smooth4 = smooth(c[2], c[1], 1)
        self.smooth4_n = smooth(c[2], c[1], 1)
        self.smooth4_s = smooth(c[2], c[1], 1)

        self.upconv1 = smooth(c[5], c[4], 1)
        self.upconv2 = nn.Sequential(smooth(c[4], (c[3]+c[4])//2, 1), smooth((c[3]+c[4])//2, c[3], 1))
        self.upconv3 = nn.Sequential(smooth(c[3], c[3], 1), smooth(c[3], (c[2]+c[3])//2, 1),smooth((c[3]+c[2])//2, c[2], 1))
        self.upconv4 = nn.Sequential(smooth(c[2], c[2], 1), smooth(c[2], (c[1]+c[2])//2, 1),smooth((c[2]+c[1])//2, c[1], 1))
        self.mnconv5 = nn.Sequential(smooth(c[1], c[1], 1), smooth(c[1], (c[0]+c[1])//2, 1),smooth((c[0]+c[1])//2, c[0], 1))
        self.mnconv5_n = nn.Sequential(smooth(c[1], c[1], 1), smooth(c[1], (c[0]+c[1])//2, 1),smooth((c[0]+c[1])//2, c[0], 1))
        self.mnconv5_s = nn.Sequential(smooth(c[1], c[1], 1), smooth(c[1], (c[0] + c[1]) // 2, 1),smooth((c[0] + c[1]) // 2, c[0], 1))
        self.mnconv5_c = nn.Sequential(smooth(c[1], c[1], 1), smooth(c[1], (c[0] + c[1]) // 2, 1),smooth((c[0] + c[1]) // 2, c[0], 1))


        # SE layer
        self.se1 = SELayer(c[1])
        self.se2 = SELayer(c[2])
        self.se3 = SELayer(c[3])
        self.se4 = SELayer(c[4])
        self.se5 = SELayer(c[5])

        #self.se1 = se_layer(c[4])
        #self.se2 = se_layer(c[3])
        #self.se9 = se_layer(c[5])
        #self.se3 = se_layer(c[2])
        #self.se4 = se_layer(c[1])

        # upsample layers
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        # SE2 layer
        #self.se5 = se2_layer(c[5])
        #self.se6 = se2_layer(c[4])
        #self.se7 = se2_layer(c[3])
        #self.se8 = se2_layer(c[2])
        #self.se0 = se2_layer(c[1])

        #self.mul1 = SliceMul()
        #self.mul2 = PixelMul()

        # Depth prediction
        self.predict = predict(c[0], self.output_channel)
        self.predict_confidence = get_confidence(c[0], 1)
        self.predict_norm = get_normal(c[0], self.output_channel*3)
        self.predict_semantic = get_semantic(c[0], self.semantic_channel)
        self.predict_p3 = predict(c[3], self.output_channel)
        self.predict_p2 = predict(c[2], self.output_channel)
        self.predict_p1 = predict(c[1], self.output_channel)

        self.predictl0 = predict(c[1], self.output_channel)
        self.predictl1 = predict(c[2], self.output_channel)
        self.predictl2 = predict(c[3], self.output_channel)
        self.predictl3 = predict(c[4], self.output_channel)
        self.predictl4 = predict(c[4], self.output_channel)


    def forward(self, x):

        # Bottom-up
        c1 = self.layer0(x)
        c1 = self.se1(c1) #(32,128,128)
        pre1 = self.predictl0(c1)
        pre1 = self.up(pre1)
        #se = self.se4(c1)
        #c1 = self.mul1(c1, se)

        c2 = self.layer1(c1)
        c2 = self.se2(c2)  #(64,64,64)
        pre2 = self.predictl1(c2)
        pre2 = self.up4(pre2)
        #se = selfse23(c2)
        #c2 = self.mul1(c2, se)

        c3 = self.layer2(c2)
        c3 = self.se3(c3)  #(128,32,32)
        pre3 = self.predictl2(c3)
        pre3 = self.up8(pre3)
        #se = self.se2(c3)
        #c3 = self.mul1(c3, se)

        c4 = self.layer3(c3)
        c4 = self.se4(c4)  #(256,16,16)
        pre4 = self.predictl3(c4)
        pre4 = self.up16(pre4)
        #se = self.se1(c4)
        #c4 = self.mul1(c4, se)

        c5 = self.layer4(c4)
        c5 = self.se5(c5)
        #top = self.top(c5)
        #c5 = c5 + top
        #se = self.se9(c5)
        #c5 = self.mul1(c5, se)

        # Top-down
        p4 = self.upconv1(c5)# (256,8,8)
        pre5 = self.predictl4(p4)
        pre5 = self.up32(pre5)

        p4 = self.up(p4)  #(256,16,16)
        p4 = torch.cat([p4,c4], dim=1) #(512,16,16)
        p4 = self.smooth1(p4)  #(512,16,16)
        #se = self.se6(p4)
        #p4 = self.mul2(p4, se)

        p3 = self.upconv2(p4) #(128,16,16)

        p3 = self.up(p3) #(128,32,32)
        p3 = torch.cat([p3, c3], dim=1) #(256,32,32)
        p3 = self.smooth2(p3) #(128,32,32)
        #se = self.se7(p3)
        #p3 = self.mul2(p3, se)

        p2 = self.upconv3(p3) #(64,32,32)

        p2 = self.up(p2) #(64,64,64)
        p2 = torch.cat([p2, c2], dim=1)  #(128,64,64)
        p2 = self.smooth3(p2) #(64,64,64)
        #se = self.se8(p2)
        #p2 = self.mul2(p2, se)

        p1 = self.upconv4(p2) #(32,64,64)

        p1 = self.up(p1) #(32,128,128)
        p1 = torch.cat([p1, c1], dim=1) #(64,128,128)
        p1_s = self.smooth4_s(p1) #(32,128,128)
        p1_n = self.smooth4_n(p1)  #(32,128,128)
        p1 = self.smooth4(p1) #(32,128,128)
        #se = self.se0(p1)
        #p1 = self.mul2(p1, se)

        p1_n = self.mnconv5_n(p1_n) #(16,128,128)
        p1_c = self.mnconv5_c(p1) #(16,128,128)
        p1 = self.mnconv5(p1) #(16,128,128)
        p1_s = self.mnconv5_s(p1_s) #(16,128,128)
        p = self.up(p1) #(16,256,256)
        p1_n = self.up(p1_n) #(16,256,256)
        p1_s = self.up(p1_s) #(16,256,256)
        p1_c = self.up(p1_c) #(16,256,256)
        norm = self.predict_norm(p1_n)
        semantic = self.predict_semantic(p1_s) #(8,256,256)
        confidence = self.predict_confidence(p1_c)
        #p = self.mnconv5(p)
        return self.predict(p),pre1,pre2,pre3,pre4,pre5,F.normalize(norm),semantic,confidence



def calc_computation(scene_model):
    torch.set_grad_enabled(False)
    _inputs = Variable(torch.rand((2, 3, IMAGE_HEIGHT, IMAGE_WIDTH)))
    _ = scene_model.forward(_inputs)
    #_ = scene_model.forward_small(_inputs)
    computation_be_used = []
    modules_conv = list(filter(lambda x: isinstance(x, new_module.Conv2dQuant), scene_model.modules()))
    for p in modules_conv:
        print(p)
        print('computation', p.computation, 'M')
        computation_be_used.append(p.computation)
    computation = sum(computation_be_used)
    print('MODEL TOTAL COMPUTATION:', computation, 'M')
    torch.set_grad_enabled(True)
    return computation


if __name__=='__main__':
    net = Net(rate=2)
    from thop import profile
    input = torch.randn(1, 3, 256, 256)
    macs, params = profile(net, inputs=(input, ))
    print (macs/1000000000)
    calc_computation(net)
