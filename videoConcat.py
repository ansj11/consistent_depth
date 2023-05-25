#!/usr/bin/env python
# coding=utf-8
import os
import cv2
import numpy as np
from utils import (
    image_io,
    visualization,
    )
from IPython import embed

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fps = 30

paths = [#'results/02', 
         #'results/04', 
         #'results/06', 
         'results/08',
         'results/10']

for path in paths:
    print('processing: ', path)

    image_dir = os.path.join(path, 'color_down_png')
    depth_dir = 'R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS4_Oadam/depth/'

    i = 0
    for fname in os.listdir(image_dir):
        if not fname.endswith('png'):
            continue
        fname = 'frame_%06d.png' % i
        img_path = os.path.join(image_dir, fname)
        raw_path = img_path.replace('color_down_png', depth_dir)
        #raw_path = img_path.replace('color_down_png', depth_dir)[:-3] + 'raw'

        image = cv2.imread(img_path)
        depth = cv2.imread(raw_path)
        #depth = image_io.load_raw_float32_image(raw_path)
        if i == 0:
            h, w = image.shape[:2]
            videoWriter = cv2.VideoWriter(path + '.avi', fourcc, fps, (w*2, h))
        i += 1
        cat = np.concatenate([image, depth], axis=1)
        videoWriter.write(cat)

    image_dir = os.path.join(path+'s2', 'color_down_png')
    i = 0
    for fname in os.listdir(image_dir):
        if not fname.endswith('png'):
            continue
        fname = 'frame_%06d.png' % i
        img_path = os.path.join(image_dir, fname)
        raw_path = img_path.replace('color_down_png', depth_dir)
        #raw_path = img_path.replace('color_down_png', depth_dir)[:-3] + 'raw'

        image = cv2.imread(img_path)
        depth = cv2.imread(raw_path)
        #depth = image_io.load_raw_float32_image(raw_path)
        i += 1
        cat = np.concatenate([image, depth], axis=1)
        videoWriter.write(cat)
cv2.imwrite('tp.jpg', cat)

videoWriter.release()
