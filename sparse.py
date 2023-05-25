#!/usr/bin/env python
# coding=utf-8
import os
import cv2
import json
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
from loaders.video_dataset import *
import matplotlib.pyplot as plt
from IPython import embed
from pdb import set_trace



def get_sparse_depth(x1, d1, pose1, x2, d2, pose2, K, flows, masks):
    """
    reprojection error, photometric error and coordinate error
    """
    H, W = x1.size()[-2:]
    pixels = pixel_grid(1, (H, W))

    point_cam = pixels_to_points(K, d1, pixels)

    points_cam_tgt = reproject_points(
        point_cam, pose1, pose2
    )
    
    matched_pixels_tgt = pixels + flows

    pixels_tgt = project(points_cam_tgt, K) # 投影-[x,y]

    dist = torch.norm(pixels_tgt - matched_pixels_tgt,
                    dim=1, keepdim=True) # * masks

    proj_warped_image = sample(x2, pixels_tgt)

    sample_depth = sample(d2, matched_pixels_tgt)

    derr = torch.abs(sample_depth - points_cam_tgt[:,-1:])

    err = torch.abs(proj_warped_image - x1).mean() / 255.0
    warped = proj_warped_image.cpu().detach().numpy().transpose(0, 2, 3, 1)
    image_ref = x1.cpu().detach().numpy().transpose(0, 2, 3, 1)
    image_tgt = x2.cpu().detach().numpy().transpose(0, 2, 3, 1)

    if False: # colmap位姿投影没问题
        for i in range(x2.size(0)):
            image_warp = cv2.addWeighted(image_ref[0], 0.5, warped[i], 0.5, 0)
            image_comb = cv2.addWeighted(image_ref[0], 0.5, image_tgt[i], 0.5, 0)
            cv2.imwrite('warp_colmap/%02d-warped.jpg'%i, image_warp.astype('uint8'))
            cv2.imwrite('warp_colmap/%02d-image1.jpg'%i, image_ref[0].astype('uint8'))
            cv2.imwrite('warp_colmap/%02d-image2.jpg'%i, image_tgt[i].astype('uint8'))
            cv2.imwrite('warp_colmap/%02d-imagec.jpg'%i, image_comb.astype('uint8'))
    
    mdist = dist.mean(dim=0,keepdim=True)
    mderr = derr.mean(dim=0,keepdim=True)
    merr = err.mean(dim=0,keepdim=True)
    tdist = mdist.mean()# + mdist.std()
    tderr = mderr.mean()# + mderr.std()
    terr = merr.mean() #+ merr.std()

    d1[mdist>tdist] = 0
    d1[mderr>tderr] = 0
    d1[merr>terr] = 0

    return d1


if __name__ == '__main__':
    PATH = './results/translate'
    OUT = './results/translate/sparse'
    if not os.path.exists(OUT):
        os.makedirs(OUT)
    meta = np.load(os.path.join(PATH, 'colmap_dense/metadata.npz'))
    
    intrinsics = meta['intrinsics']
    extrinsics = meta['extrinsics']
    #set_trace()
    #path = os.path.join(PATH, 'color_down_png/') # 小图光度误差不可信,只过滤了远处深度
    path = os.path.join(PATH, 'color_full')
    fnames = os.listdir(path)
    num = len(fnames)

    flow_list = os.path.join(PATH, 'flow_list.json')
    with open(flow_list) as f:
        flow_indices = json.load(f)
    flow_array = np.array(flow_indices)
    flow_sort = sorted(flow_array, key=lambda x: x[0])
    flow_dict = {}
    for item in flow_sort:
        k, v = item
        if k in flow_dict:
            flow_dict[k].append(v)
        else:
            flow_dict[k] = [v]
    
    K = torch.from_numpy(intrinsics[0]).unsqueeze(0).float().cuda() / 384 * 1920
    print(K)
    for i, v in flow_dict.items():
        imgpath1 = os.path.join(path, 'frame_%06d.png'%i)
        #depthpath1 = imgpath1.replace('color_down_png', 'depth_colmap_dense/depth')[:-3] + 'raw'
        depthpath1 = imgpath1.replace('color_full', 'depth_colmap_dense/depth')[:-3] + 'raw'

        img1 = cv2.imread(imgpath1)

        disp1 = image_io.load_raw_float32_image(depthpath1)
        depth1 = 1.0 / disp1
        depth1[np.isnan(depth1)] = 0

        h, w = img1.shape[:2]

        x1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA)

        d1 = cv2.resize(depth1, (w, h), interpolation=cv2.INTER_NEAREST)

        pose1 = extrinsics[i]

        x1 = torch.from_numpy(x1.transpose(2, 0, 1)).unsqueeze(0).float().cuda()

        d1 = torch.from_numpy(d1).unsqueeze(0).unsqueeze(0).float().cuda()

        pose1 = torch.from_numpy(pose1).unsqueeze(0).float().cuda()

        x2, d2, pose2, flows, masks = [], [], [], [], []
        for j in v:
            imgpath2 = os.path.join(path, 'frame_%06d.png'%j)
            #depthpath2 = imgpath2.replace('color_down_png', 'depth_colmap_dense/depth')[:-3] + 'raw'
            depthpath2 = imgpath2.replace('color_full', 'depth_colmap_dense/depth')[:-3] + 'raw'
            flowpath = os.path.join(PATH,'flow/flow_%06d_%06d.raw'%(i, j))
            maskpath = flowpath.replace('flow', 'mask')[:-3] + 'png'
            
            flow = load_flow(flowpath, channels_first=True)
            mask = load_mask(maskpath, channels_first=True)
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_NEAREST) / 384 * 1920
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
            img2 = cv2.imread(imgpath2)
    
            disp2 = image_io.load_raw_float32_image(depthpath2)
            depth2 = 1.0 / disp2
            depth2[np.isnan(depth2)] = 0

            x = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
    
            d = cv2.resize(depth2, (w, h), interpolation=cv2.INTER_NEAREST)
    
            pose = extrinsics[j]

            x2.append(x)
            d2.append(d)
            pose2.append(pose)
            flows.append(flow)
            masks.append(mask)

        x2 = np.stack(x2, axis=0)
        d2 = np.stack(d2, axis=0)
        pose2 = np.stack(pose2, axis=0)
        flows = torch.stack(flows, dim=0).cuda()
        masks = torch.stack(masks, dim=0).cuda()
    
        x2 = torch.from_numpy(x2.transpose(0, 3, 1, 2)).float().cuda()
    
        d2 = torch.from_numpy(d2).unsqueeze(1).float().cuda()
    
        pose2 = torch.from_numpy(pose2).float().cuda()

        d1 = get_sparse_depth(x1, d1, pose1, x2, d2, pose2, K, flows, masks)
        cv2.imwrite(os.path.join(OUT, '%0d3-image.jpg'%i), img1)
        plt.imsave(os.path.join(OUT, '%03d-depth.jpg'%i), depth1, vmax=d1.cpu().numpy().max(), cmap='jet')
        plt.imsave(os.path.join(OUT, '%03d-dspar.jpg'%i), d1.cpu().numpy()[0,0], cmap='jet')
        print(imgpath1)
