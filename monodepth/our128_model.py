#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.autograd as autograd

from utils.url_helpers import get_model_from_url

from .MobileNetv1Quant import Net
from .depth_model import DepthModel


class Our128Model(DepthModel):
    # Requirements and default settings
    align = 16
    learning_rate = 0.0004
    lambda_view_baseline = 0.1

    def __init__(self, support_cpu: bool = False, pretrained: bool = True):
        super().__init__()

        if support_cpu:
            # Allow the model to run on CPU when GPU is not available.
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            # Rather raise an error when GPU is not available.
            self.device = torch.device("cuda")

        try:
            self.model = torch.load('/home/anshijie/2021/consistent_depth/checkpoints/model-best.model')
        except:
            self.model = Net(rate=2)
            ckpt = torch.load('/home/anshijie/2021/consistent_depth/checkpoints/model-best.pth')
            self.model.load_state_dict(ckpt)
        # 兼容1.6
        for k, m in self.model.named_modules():
            m._non_persistent_buffers_set = set()

        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.device)


    def estimate_depth(self, images):
        # Reshape ...CHW -> XCHW
        shape = images.shape
        C, H, W = shape[-3:]
        input_ = images.reshape(-1, C, H, W).to(self.device)

        output = self.model(input_)



        return output

    def save(self, file_name):
        state_dict = self.model.state_dict()
        torch.save(state_dict, file_name)

    def load(self, file_name):
        state_dict = torch.load(file_name)
        self.model.load_state_dict(state_dict)
