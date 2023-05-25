#!/usr/bin/env python
# coding=utf-8
import os
import cv2
import numpy as np

from utils import image_io

paths = [
        'results/mvs/',
        ]

for path in paths:
    print('processing: ', path)
    image_dir = os.path.join(path, 'color_down_png')
    metapath1 = os.path.join(path, 'R_hierarchical2_our320/metadata_scaled.npz')
    meta1 = np.load(metapath1)
    np.savetxt(os.path.join(image_dir, 'intrinsics.txt'), meta1['intrinsics'][0])
    
    i = 0
    for fname in os.listdir(image_dir):
        if not fname.endswith('png'):
            continue
        fname = 'frame_%06d.png' % i
        image_path = os.path.join(image_dir, fname)
        depth_path = image_path.replace('color_down_png', 'depth_our320/depth')[:-3] + 'raw'

        depth = image_io.load_raw_float32_image(depth_path)
        depth = 1.0 / (depth + 1e-6) * 1000
        out_path = image_path.replace('color_down_png', 'color_down_png/depth1')
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        pose_path = out_path[:-3] + 'txt'
        cv2.imwrite(out_path, depth.astype('uint16'))
        np.savetxt(pose_path, meta1['extrinsics'][i])

        i += 1
