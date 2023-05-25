import cv2
import sys
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable, Function
import numpy as np
from torch.nn.parameter import Parameter
from new_module import conv_layer, AvgQuant, mobile_unit, linear_layer
from config import *
from torchvision import models, transforms
from torchvision.models.resnet import resnet101
import numpy as np
from IPython import embed
from hxwgcblock2 import ContextBlock

def se_layer(channel, reduction=4):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        #conv_layer(channel, channel//reduction, ks=1, padding=0, relu=True, bn=False, bias=True),                
        #conv_layer(channel//reduction, channel, ks=1, padding=0, relu=False, bn=False, bias=True),                
        nn.Conv2d(channel, channel//reduction, 1, 1, 0, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(channel//reduction, channel, 1, 1, 0, bias=True),
        nn.Sigmoid())

def se2_layer(channel, reduction=4):
    return nn.Sequential(
        #conv_layer(channel, channel//reduction, ks=1, padding=0, relu=True, bn=False, bias=True),
        #conv_layer(channel//reduction, 1, ks=1, padding=0, relu=False, bn=False, bias=True),
        nn.Conv2d(channel, channel//reduction, 1, 1, 0, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(channel//reduction, 1, 1, 1, 0, bias=True),
        nn.Sigmoid())

##mobilenet
def conv(inp, oup, stride):
    return nn.Sequential(
        #conv_layer(inp, oup, ks=3, stride=stride, padding=1, bias=True, bn=False),
        nn.Conv2d(inp, oup, 3, stride, 1, bias=True),
        nn.ReLU(inplace=True)
    )

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        #conv_layer(inp, oup, ks=3, stride=stride, padding=1, bias=False),
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        conv_layer(inp, inp, ks=3, stride=stride, padding=1, group=inp, bias=False),
        #nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        #nn.BatchNorm2d(inp),
        #nn.ReLU(inplace=True),

        conv_layer(inp, oup, ks=1, stride=1, padding=0, bias=False),
        #nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        #nn.BatchNorm2d(oup),
        #nn.ReLU(inplace=True),
    )

def smooth(inp, oup, stride):
    return nn.Sequential(
        #conv_layer(inp, inp, ks=3, stride=stride, padding=1, bias=False, bn=True),
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True)
        )


def predict(in_planes, out_planes):
    return nn.Sequential(
        #conv_layer(in_planes, out_planes, ks=3, stride=1, padding=1, bias=True, bn=False, relu=False),
        nn.Conv2d(in_planes, out_planes, 3, 1, 1, bias=True),
        nn.ReLU(inplace=True))

def upshuffle(in_planes, out_planes, upscale_factor):
    return nn.Sequential(
        #conv_layer(in_planes, out_planes*upscale_factor**2, ks=3, stride=1, padding=1, bias=False, bn=True, relu=False),
        nn.Conv2d(in_planes, out_planes*upscale_factor**2, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_planes*upscale_factor**2),
        nn.PixelShuffle(upscale_factor),
        nn.ReLU(inplace=True)
    )
#
class DFG(nn.Module):
    def __init__(self, inp, oup, mode='spatial', fuse='mul'):
        super(DFG, self).__init__()
        if mode == 'channel':
            self.conv_mask = nn.Conv2d(inp+oup, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
            self.att = nn.Sequential(
                                     nn.Conv2d(inp+oup, oup, kernel_size=1, stride=1, padding=0, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(oup, oup, kernel_size=1, stride=1, padding=0, bias=True),
                                     nn.ReLU(inplace=True)
                                    )
        elif mode == 'spatial':
            self.att = nn.Sequential(nn.Conv2d(inp+oup, inp//2, kernel_size=3, stride=1, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(inp//2, 1, kernel_size=1, stride=1, padding=0, bias=True),
                                     nn.ReLU(inplace=True)
                                    )
        else:
            self.att = nn.Sequential(nn.Conv2d(inp+oup, oup, kernel_size=3, stride=1, padding=1, bias=True),
                                     nn.ReLU(inplace=True)
                                    )
        self.fuse = fuse
        self.mode = mode

    def forward(self, g, x):
        g = torch.cat([g, x], dim=1)
        if self.mode == 'channel':
            N, C, H, W = g.size()
            input = g
            input = input.view(N, C, H*W)
            input = input.unsqueeze(1)
            context_mask = self.conv_mask(g)
            context_mask = context_mask.view(N, 1, H*W)
            context_mask = self.softmax(context_mask)
            context_mask = context_mask.unsqueeze(-1)
            context = torch.matmul(input, context_mask)
            g = context.view(N, C, 1, 1)
        att = self.att(g)
        if self.fuse == 'mul':
            out = att * x
        else:
            out = x + att
        return out

class Net(nn.Module):
    def __init__(self, pretrained=True, output_channel=1, rate=1, fixed_feature_weights=False):
        super(Net, self).__init__()
        self.output_channel = output_channel
       
        resnet = resnet101(pretrained=pretrained)
        
        # Freeze resnet weights
        if fixed_feature_weights:
            for p in resnet.parameters():
                p.requires_grad = False
        
        c = [i//rate for i in [64, 256, 512, 1024, 2048]]
        conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.layer0 = nn.Sequential(conv1, resnet.bn1, resnet.relu)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)
        
        # previous layers
        self.mid1 = conv_bn(c[0], c[0], 1)
        self.mid2 = conv_bn(c[1], c[1], 1)
        self.mid3 = conv_bn(c[2], c[2], 1)
        self.mid4 = conv_bn(c[3], c[3], 1)

        # toplayer
        self.top = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                                 nn.Conv2d(c[4], c[4], 1, 1, 0, bias=False),
                                 nn.BatchNorm2d(c[4]),
                                 nn.ReLU(inplace=True))#conv_layer(c[4], c[4], ks=1, padding=0, relu=True, bn=False, bias=False))

        # Smooth layers
        self.smooth1 = conv_bn(c[4], c[3], 1)
        self.smooth2 = conv_bn(c[3], c[2], 1)
        self.smooth3 = conv_bn(c[2], c[1], 1)
        self.smooth4 = conv_bn(c[0]*2, c[0], 1)
        self.smooth5 = conv(c[0]*4, c[0]*2, 1)

        self.upconv1 = conv_bn(c[4], c[3], 1)
        self.upconv2 = conv_bn(c[3], c[2], 1)
        self.upconv3 = conv_bn(c[2], c[1], 1)
        self.upconv4 = conv_bn(c[1], c[0], 1)
        self.upconv5 = conv(c[0]*2, c[0], 1)

        self.att1 = DFG(c[3], c[3])
        self.att2 = DFG(c[2], c[2])
        self.att3 = DFG(c[1], c[1])
        self.att4 = DFG(c[0], c[0])

        # upshuffle layers
        self.up1 = upshuffle(c[3], c[0], 8)
        self.up2 = upshuffle(c[2], c[0], 4)
        self.up3 = upshuffle(c[1], c[0], 2)

        # Context block
        self.gc2 = ContextBlock(c[1], 1/4)
        self.gc3 = ContextBlock(c[2], 1/4)
        self.gc4 = ContextBlock(c[3], 1/4)
        self.gc5 = ContextBlock(c[4], 1/4)

        # Depth prediction
        self.predict = predict(c[0], self.output_channel)

    def forward(self, x):
        # Bottom-up
        c1 = self.layer0(x)
        c2 = self.layer1(c1)
        c2 = self.gc2(c2)
        c3 = self.layer2(c2)
        c3 = self.gc3(c3)
        c4 = self.layer3(c3)
        c4 = self.gc4(c4)
        c5 = self.layer4(c4)
        p5 = self.gc5(c5)
        
        # top layer
        #top = self.top(c5)
        #p5 = c5 + top

        # Top-down
        _, _, H, W = c4.size()
        p4 = F.upsample(p5, size=(H, W), mode='bilinear')
        p4  = self.upconv1(p4)
        b  = self.att1(p4, self.mid4(c4))
        p4 = torch.cat([p4, b], dim=1)
        p4 = self.smooth1(p4)
        
        _, _, H, W = c3.size()
        p3 = F.upsample(p4, size=(H, W), mode='bilinear')
        p3  = self.upconv2(p3)
        b  = self.att2(p3, self.mid3(c3))
        p3 = torch.cat([p3, b], dim=1)
        p3 = self.smooth2(p3)

        _, _, H, W = c2.size()
        p2 = F.upsample(p3, size=(H, W), mode='bilinear')
        p2  = self.upconv3(p2)
        b  = self.att3(p2, self.mid2(c2))
        p2 = torch.cat([p2, b], dim=1)
        p2 = self.smooth3(p2)

        _, _, H, W = c1.size()
        p1 = F.upsample(p2, size=(H, W), mode='bilinear')
        p1  = self.upconv4(p1)
        b  = self.att4(p1, self.mid1(c1))
        p1 = torch.cat([p1, b], dim=1)
        p1 = self.smooth4(p1)

        # concatenate all branch
        d4, d3, d2, d1 = self.up1(p4), self.up2(p3), self.up3(p2), p1
        d = torch.cat([F.upsample(p, size=(H, W), mode='bilinear') for p in [d1, d2, d3, d4]], dim=1)
        d = self.smooth5(d)

        _, _, H, W = x.size()
        p = F.upsample(d, size=(H, W), mode='bilinear')
        p = self.upconv5(p)
        return self.predict(p)


def calc_computation_orig(scene_model):
    torch.set_grad_enabled(False)
    # just calculate conv's computation for now
    list_conv = []
    # declare hook function
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
        list_conv.append(flops)
    def calc(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, nn.Conv2d):
                net.register_forward_hook(conv_hook)
            return
        for c in childrens:  # Recursively step into the sub nn.Modules to find the nn.Conv2d
            calc(c)
    
    calc(scene_model)  # Recursively register the hook function in nn.Conv2d
    # y = scene_model.forward(Variable(torch.ones(1, 3, IMAGE_SIZE, IMAGE_SIZE)))         
    y = scene_model.forward(Variable(torch.ones(1, 5, IMAGE_HEIGHT, IMAGE_WIDTH))) 
    print('total_compuation:', sum(list_conv) / 1e6, 'M')
    #torch.set_grad_enabled(True)

def Resize(image):
    h, w, _ = image.shape
    if h >= w:
        h_out = 160
        w_mid = w / h * 160
        w_out = int(32 * np.round(w_mid / 32))
    else:
        w_out = 160
        h_mid = h / w *160
        h_out = int(32 * np.round(h_mid / 32))
    image = cv2.resize(image, (w_out, h_out), interpolation=cv2.INTER_LINEAR)
    return image

def Bgr2Yuv(image):
    yuv = np.clip(image.copy(), 0, 255).astype(np.float32)
    yuv[:,:,0] = 0.299*image[:,:,2] + 0.587*image[:,:,1] + 0.114*image[:,:,0]
    yuv[:,:,1] = 0.492*(image[:,:,0] - yuv[:,:,0]) + 128
    yuv[:,:,2] = 0.877*(image[:,:,2] - yuv[:,:,0]) + 128
    image = np.clip(yuv, 0, 255)
    '''
    # color trans image.dtype must be uint8
    if image.dtype == 'uint8':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        cv2.imwrite('yuv.jpg', image[:,:,(2,1,0)])
    else:
        print('error data type in Bgr2Yuv')
    '''
    return image.astype('uint8')

def Normalize(image):
    image = image.astype('float32')
    image /= 255.0
    return image

if __name__=='__main__':
    net = Net()
    net.eval()
    calc_computation_orig(net)
