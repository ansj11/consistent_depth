#!/usr/bin/env python
# coding=utf-8
import os
import cv2
import numpy as np
import shutil
from utils import image_io
from tools import make_video as mk
from IPython import embed
from scipy.spatial.transform import Rotation as R
from utils import visualization
from utils.helpers import SuppressedStdout


ROT_COLMAP_TO_NORMAL = np.diag([1, -1, -1])

def make_video(dirpath, outpath):
    num = min(100, len(os.listdir(dirpath)))
    for i in range(num):
        image_path = os.path.join(dirpath, '%d.jpg'%i)
        image = cv2.imread(image_path)
        if i == 0:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            fps = 30
            h, w = image.shape[:2]
            videoWriter = cv2.VideoWriter(outpath + '.avi', fourcc, fps, (w, h))
        videoWriter.write(image)

    videoWriter.release()
    print('video generate done!')

def get_extrinsics(dirpath):
    num = min(100, len(os.listdir(dirpath)))
    poses = []
    for i in range(num):
        pose_path = os.path.join(dirpath, '%d.txt'%i)
        pose = np.loadtxt(pose_path)
        R = ROT_COLMAP_TO_NORMAL.dot(pose[:-1,:-1]).dot(ROT_COLMAP_TO_NORMAL.T)
        t = ROT_COLMAP_TO_NORMAL.dot(pose[:-1,-1:])
        poses.append(np.concatenate([R, t], axis=1))
    
    print('pose generate done!')
    return np.stack(poses, axis=0)

PATH = '/share/group_guoxiaoyan/group_depth/dataset/scannet_parse/scans_test/'

for dirname in os.listdir(PATH):
    print('processing: ', dirname)
    dirpath = os.path.join(PATH, dirname, 'color')
    dirpath2 = os.path.join(PATH, dirname, 'depth')
    if not os.path.exists(dirpath):
        continue
    outpath = os.path.join('data/videos/', dirname)
    extrinsics = get_extrinsics(os.path.join(PATH, dirname, 'pose'))
    intrinsics = np.loadtxt(os.path.join(PATH, dirname, 'intrinsic/intrinsic_color.txt'))
    intrinsics_scale = np.array([intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]]) / 1296 * 384
    intrinsics_new = np.tile(intrinsics_scale, (len(extrinsics), 1))
    """
    print('combine videos...')
    #mk.make_video('ffmpeg', os.path.join(dirpath, 'vio_%08d.jpg'), outpath)
    make_video(dirpath, outpath)

    print('generate depths...')
    num = min(100, len(os.listdir(dirpath2)))
    for i in range(num):
        depth_path = os.path.join(dirpath2, '%d.png'%i)
        depth = cv2.imread(depth_path, -1) / 1000.0
        # arkit有效深度是4m
        index = i
        save_path = os.path.join('./results/', dirname, 'depth_colmap_dense/depth/frame_%06d.png'%index)
        inv_depth = 1.0 / depth
        ix = np.isinf(inv_depth) | (inv_depth < 0)
        inv_depth[ix] = float('nan')
        inv_depth = cv2.resize(inv_depth, (216, 384), cv2.INTER_LINEAR)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        image_io.save_raw_float32_image(save_path[:-3]+'raw', inv_depth)
    with SuppressedStdout():
        src_path = os.path.join('results', dirname, 'depth_colmap_dense/depth')
        visualization.visualize_depth_dir(src_path, src_path, force=True, min_percentile=0, max_percentile=99)
    """
    meta_path = os.path.join('./results/', dirname, 'colmap_dense/metadata.npz')
    if not os.path.exists(os.path.dirname(meta_path)):
        os.makedirs(os.path.dirname(meta_path))
    np.savez(meta_path, intrinsics=intrinsics_new, extrinsics=extrinsics)


