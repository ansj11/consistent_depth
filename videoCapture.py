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
fps = 25

path = './results/ayush'
image_dir = os.path.join(path, 'color_down_png')
depth_dir = 'R_hierarchical2_our128/B0.1_R1.0_PL1-0_LR0.0004_BS4_Oadam/depth/'

images, depths = [], []
i = 0
for fname in os.listdir(image_dir):
    if not fname.endswith('png'):
        continue
    fname = 'frame_%06d.png' % i
    img_path = os.path.join(image_dir, fname)
    raw_path = img_path.replace('color_down_png', depth_dir)[:-3] + 'raw'

    image = cv2.imread(img_path)
    depth = image_io.load_raw_float32_image(raw_path)
    depth = 1.0 / (depth + 1e-6)
    print(depth.min(), depth.max(), depth.mean(), np.median(depth), depth.std())
    images.append(image)
    depths.append(depth)
    i += 1

d = np.stack(depths, axis=0)

mmin, mmax, mean, std, median = d.min(), d.max(), d.mean(), d.std(), np.median(d)

print(mmin, mmax, mean, std, median)
mmin, mmax = max(median - 3*std, 0), median + 3*std
print(mmin, mmax)

videoWriter = cv2.VideoWriter(path + '.avi', fourcc, fps, (224*2, 384))

for (image, depth) in zip(images, depths):
    depth = (depth - mmin) / (mmax - mmin) * 255
    show = cv2.applyColorMap(depth.astype('uint8'), cv2.COLORMAP_JET)
    cat = np.concatenate([image, show], axis=1)
    videoWriter.write(cat)

cv2.imwrite('tp.jpg', cat)

videoWriter.release()
