#!/usr/bin/env python
# coding=utf-8
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from utils.geometry import (
    pixel_grid,
    focal_length,
    project,
    pixels_to_points,
    reproject_points,
    sample,
    )
from utils import (
    image_io,
    visualization,
    )
from IPython import embed

#meta = np.load('./results/06/colmap_dense/metadata.npz')
#meta = np.load('./results/06/R_hierarchical2_our128/metadata_scaled.npz')
#meta = np.load('./results/scene0707_00/colmap_dense/metadata.npz')
#meta = np.load('./results/scene0707_00/R_hierarchical2_mc/metadata_scaled.npz')
meta = np.load('./results/ap06/R_hierarchical2_our128/metadata_scaled.npz')

intrinsics = meta['intrinsics']
extrinsics = meta['extrinsics']

path = './results/ap06/color_down_png/'
fnames = os.listdir(path)
num = len(fnames)

"""while True:
    i = np.random.randint(num)
    j = np.random.randint(num)
    if i != j and abs(i-j) < 2:
        break"""
error = 0
for i in range(num-1):
    interval = 2
    #i = np.random.randint(num - interval)
    j = i + interval # np.random.randint(num)

    imgpath1 = os.path.join(path, 'frame_%06d.png'%i)
    imgpath2 = os.path.join(path, 'frame_%06d.png'%j)

    depthpath1 = imgpath1.replace('color_down_png', 'depth_colmap_dense/depth')[:-3] + 'raw'
    depthpath2 = imgpath2.replace('color_down_png', 'depth_colmap_dense/depth')[:-3] + 'raw'

    img1 = cv2.imread(imgpath1)
    img2 = cv2.imread(imgpath2)

    #depth1 = cv2.imread(depthpath1, -1) / 1000.0
    #depth2 = cv2.imread(depthpath2, -1) / 1000.0

    disp1 = image_io.load_raw_float32_image(depthpath1)
    disp2 = image_io.load_raw_float32_image(depthpath2)
    depth1 = 1.0 / (disp1 + 1e-6)
    depth2 = 1.0 / (disp2 + 1e-6)
    depth1[np.where(np.isnan(depth1))] = 0
    depth2[np.where(np.isnan(depth2))] = 0

    h, w = img1.shape[:2]
    x1 = cv2.resize(img1, (w, h), cv2.INTER_AREA)
    x2 = cv2.resize(img2, (w, h), cv2.INTER_AREA)

    d1 = cv2.resize(depth1, (w, h), cv2.INTER_AREA)
    d2 = cv2.resize(depth2, (w, h), cv2.INTER_AREA)

    pose1 = extrinsics[i]
    pose2 = extrinsics[j]

    x1 = torch.from_numpy(x1.transpose(2, 0, 1)).unsqueeze(0).float().cuda()
    x2 = torch.from_numpy(x2.transpose(2, 0, 1)).unsqueeze(0).float().cuda()

    d1 = torch.from_numpy(d1).unsqueeze(0).unsqueeze(0).float().cuda()
    d2 = torch.from_numpy(d2).unsqueeze(0).unsqueeze(0).float().cuda()

    pose1 = torch.from_numpy(pose1).unsqueeze(0).float().cuda()
    pose2 = torch.from_numpy(pose2).unsqueeze(0).float().cuda()

    K = torch.from_numpy(intrinsics[0]).unsqueeze(0).float().cuda()

    H, W = h, w
    N = 1

    pixels = pixel_grid(1, (H, W))

    point_cam = pixels_to_points(K, d1, pixels)

    points_cam_tgt = reproject_points(
        point_cam, pose1, pose2
    )


    pixels_tgt = project(points_cam_tgt, K) # 投影-[x,y]

    proj_warped_image = sample(x2, pixels_tgt)
    err = torch.abs(proj_warped_image - x1).mean() / 255.0
    error += err
    warped = proj_warped_image.cpu().detach().numpy().transpose(0, 2, 3, 1)
    image_ref = x1.cpu().detach().numpy().transpose(0, 2, 3, 1)
    image_tgt = x2.cpu().detach().numpy().transpose(0, 2, 3, 1)

    if True:
        image_warp = cv2.addWeighted(image_ref[0], 0.5, warped[0], 0.5, 0)
        image_comb = cv2.addWeighted(image_ref[0], 0.5, image_tgt[0], 0.5, 0)
        cv2.imwrite('warp_arkit/%02d-warped.jpg'%i, image_warp.astype('uint8'))
        cv2.imwrite('warp_arkit/%02d-image1.jpg'%i, image_ref[0].astype('uint8'))
        cv2.imwrite('warp_arkit/%02d-image2.jpg'%i, image_tgt[0].astype('uint8'))
        cv2.imwrite('warp_arkit/%02d-imagec.jpg'%i, image_comb.astype('uint8'))

print(error / (i+1))
