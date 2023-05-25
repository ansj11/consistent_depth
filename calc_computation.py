import os
import cv2
import torch
import torch.nn as nn
import numpy as np
#import bit_pytorch.models as models
#import bit_pytorch.models2 as models2
#import model_aff as models
from monodepth.depth_model_registry import get_depth_model
from monodepth.mannequin_challenge.models.hourglass import HourglassModel



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
    y = scene_model.forward(torch.ones(1, 3, 256, 256)) 
    print('total_compuation:', sum(list_conv) / 1e6, 'M')
    #torch.set_grad_enabled(True)

if __name__=='__main__':
    #model = get_depth_model('mc')() # 无法计算
    model = HourglassModel(3)
    #model = get_depth_model('midas2')() # 45G 256x256
    #model = get_depth_model('monodepth2')() # 21G 1024x320
    from IPython import embed
    #embed()
    model.eval()
    calc_computation_orig(model)

