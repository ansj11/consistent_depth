#!/usr/bin/env python
# coding=utf-8
import os
import cv2
import numpy as np
import shutil

from utils import image_io

paths = [
         #'results/ayush/',
         #'results/object/',
         #'results/selfie/',
         #'results/static/',
         #'results/translate/',
        #'results/01/',
        #'results/02/',
        #'results/03/',
        #'results/04/',
        #'results/05/',
        #'results/06/',
        #'results/07/',
        #'results/08/',
        #'results/09/',
        #'results/10/',
        #'results/11/',
        #'results/12/',
        #'results/22/',
        #'results/ap06/',
        #'results/scene0709_00/',
        #'results/scene0710_00/',
        #'results/scene0770_00/',
        'results/02s2/',
        'results/04s2/',
        'results/06s2/',
        'results/08s2/',
        'results/10s2/',
        ]

for path in paths:
    print('processing: ', path)
    image_dir = os.path.join(path, 'color_down_png')
    out_dir = image_dir.replace('results', 'recon3')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    metapath1 = os.path.join(path, 'R_hierarchical2_mc/metadata_scaled.npz')
    metapath2 = os.path.join(path, 'R_hierarchical2_our128/metadata_scaled.npz')
    meta1 = np.load(metapath1)
    #meta2 = np.load(metapath2)
    #assert (meta1['intrinsics'] == meta2['intrinsics']).all()
    np.savetxt(os.path.join(out_dir, 'intrinsics.txt'), meta1['intrinsics'][0])
    
    i = 0
    for fname in os.listdir(image_dir):
        if not fname.endswith('png'):
            continue
        fname = 'frame_%06d.png' % i
        image_path = os.path.join(image_dir, fname)
        shutil.copy(image_path, out_dir)

        depth_path = image_path.replace('color_down_png', 'R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS4_Oadam/depth')[:-3] + 'raw'

        depth = image_io.load_raw_float32_image(depth_path)
        depth = 1.0 / (depth + 1e-6) * 1000
        out_path = image_path.replace('results', 'recon3').replace('color_down_png', 'color_down_png/depth1')
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        pose_path = out_path[:-3] + 'txt'
        cv2.imwrite(out_path, depth.astype('uint16'))
        np.savetxt(pose_path, meta1['extrinsics'][i])
        """
        depth_path = image_path.replace('color_down_png', 'R_hierarchical2_our128/B0.1_R1.0_PL1-0_LR0.0004_BS4_Oadam/depth')[:-3] + 'raw'

        depth = image_io.load_raw_float32_image(depth_path)
        depth = 1.0 / (depth + 1e-6) * 1000
        out_path = image_path.replace('results', 'recon3').replace('color_down_png', 'color_down_png/depth2')
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        pose_path = out_path[:-3] + 'txt'
        cv2.imwrite(out_path, depth.astype('uint16'))
        np.savetxt(pose_path, meta1['extrinsics'][i])"""
        i += 1
