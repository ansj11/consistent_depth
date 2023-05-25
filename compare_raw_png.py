#!/usr/bin/env python
# coding=utf-8
import os
import cv2
from IPython import embed

from utils import image_io

"""
path = '/home/anshijie/2021/consistent_depth/results/translate/color_down/frame_000000.raw'

im = image_io.load_raw_float32_image(path)
#im = im[:,:,::-1]

im2 = cv2.imread(path.replace('down', 'down_png')[:-3]+'png', cv2.IMREAD_UNCHANGED)

im2 = im2 / 255.0

cv2.imwrite('tp.jpg', (im*255).astype('uint8'))
embed()
"""

path = './results/translate/R_hierarchical2_our128/B0.1_R1.0_PL1-0_LR0.0004_BS4_Oadam/depth/frame_000000.raw'

d0 = image_io.load_raw_float32_image(path)

path = './results/translate/depth_our128/depth/frame_000000.raw'
d1 = image_io.load_raw_float32_image(path)

embed()




