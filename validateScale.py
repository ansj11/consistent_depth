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

path = './results/ap06'
image_dir = os.path.join(path, 'color_down_png')
depth_dir = 'depth_mc/depth'

images, depths, scales = [], [], []
i = 0
for fname in os.listdir(image_dir):
    if not fname.endswith('png'):
        continue
    fname = 'frame_%06d.png' % i
    img_path = os.path.join(image_dir, fname)
    raw_path = img_path.replace('color_down_png', depth_dir)[:-3] + 'raw'

    image = cv2.imread(img_path)
    disp1 = image_io.load_raw_float32_image(raw_path)
    depth = 1.0 / (disp1 + 1e-6)
    print(depth.min(), depth.max(), depth.mean(), np.median(depth), depth.std())

    raw_path = raw_path.replace(depth_dir, 'depth_colmap_dense/depth')
    disp2 = image_io.load_raw_float32_image(raw_path)
    h, w = image.shape[:2]
    disp2 = cv2.resize(disp2, (w, h))
    depth2 = 1.0 / (disp2 + 1e-6)
    ix = np.isfinite(disp2)
    scale = np.median(disp1 / disp2)
    images.append(image)
    depths.append([depth, depths/scale])
    scales.append(scale)
    i += 1

d = np.stack([i[0] for i in depths], axis=0)

mmin, mmax, mean, std, median = d.min(), d.max(), d.mean(), d.std(), np.median(d)

print(mmin, mmax, mean, std, median)
mmin, mmax = max(median - 3*std, 0), median + 3*std
print(mmin, mmax)

videoWriter = cv2.VideoWriter(path + '.avi', fourcc, fps, (224*3, 384))

for (image, dd) in zip(images, depths):
    depth, depth2 = dd
    depth = (depth - mmin) / (mmax - mmin) * 255
    show = cv2.applyColorMap(depth.astype('uint8'), cv2.COLORMAP_JET)
    depth2 = (depth2 - mmin) / (mmax - mmin) * 255
    show2 = cv2.applyColorMap(depth2.astype('uint8'), cv2.COLORMAP_JET)
    cat = np.concatenate([image, show, show2], axis=1)
    videoWriter.write(cat)

cv2.imwrite('tp.jpg', cat)

videoWriter.release()
