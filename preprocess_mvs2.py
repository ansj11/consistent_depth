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
from pdb import set_trace


ROT_COLMAP_TO_NORMAL = np.diag([1, -1, -1])

def make_video(dirpath, outpath):
    num = min(100, len(os.listdir(dirpath)))
    for i in range(num):
        image_path = os.path.join(dirpath, 'vio_%08d.jpg'%i)
        image = cv2.imread(image_path)
        if i == 0:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            fps = 30
            h, w = image.shape[:2]
            videoWriter = cv2.VideoWriter(outpath + '.avi', fourcc, fps, (w, h))
        videoWriter.write(image)

    videoWriter.release()
    print('video generate done!')

def read_depth(path):
    array = np.empty(0)
    with open(path, "rb") as fid:
        byte = fid.read(4)
        types = int.from_bytes(byte, byteorder='little', signed=True)
        byte = fid.read(4)
        height = int.from_bytes(byte, byteorder='little', signed=True)
        byte = fid.read(4)
        width = int.from_bytes(byte, byteorder='little', signed=True)
        byte = fid.read(4)
        channels = int.from_bytes(byte, byteorder='little', signed=True)

        array = np.fromfile(fid, np.float32)
        array = array.reshape((width, height, channels), order="F")
        array = np.transpose(array, (1, 0, 2)).squeeze()
                            
        # print maxval and minval of depth map
        # print("depth max and min ", array.max(), " ", array.min())
        
        # filter the outliers
        # min_depth, max_depth = np.percentile(array, [5, 95])
        # array[array < min_depth] = min_depth
        # array[array > max_depth] = max_depth
        # to visualize the depth map
        # plt.figure(dpi=300)
        # plt.imshow(array, cmap='gray')
        # plt.show()
        # print("depth max and min after filter", array.max(), " ", array.min())

    return array

def get_extrinsics(pose_path):
    with open(pose_path) as f:
        pose = [x.strip() for x in f.readlines()]
    extrinsics = np.array([[float(y) for y in x.split()] for x in pose[1:5]])
    intrinsics = np.array([[float(y) for y in x.split()] for x in pose[7:10]])
    R = ROT_COLMAP_TO_NORMAL.dot(extrinsics[:-1,:-1]).dot(ROT_COLMAP_TO_NORMAL.T)
    t = ROT_COLMAP_TO_NORMAL.dot(extrinsics[:-1,-1:])
    R = R.transpose(1,0)
    t = -np.dot(R, t)
    #R = extrinsics[:-1,:-1] # 投影更不对
    #t = extrinsics[:-1,-1:]
    extrinsics = np.concatenate([R, t], axis=1)
    intrinsics = np.stack([intrinsics[0,0], intrinsics[1,1], intrinsics[0,-1], intrinsics[1,-1]], axis=0) / 1280 * 384
    
    return extrinsics, intrinsics

PATH = '/intern-share/zhangwenhong/images/'

make_video(PATH, './data/videos/mvs')

dirname = 'mvs'
extrinsics = []
intrinsics = []
maxval = 0
for i, fname in enumerate(os.listdir(PATH)):
    print('processing: ', fname, i)
    image_path = os.path.join(PATH, fname)
    if not os.path.exists(image_path):
        continue
    extrinsic, intrinsic = get_extrinsics(os.path.join(PATH.replace('images', 'cams'), '%08d_cam.txt' % i))
    print(extrinsic)
    extrinsics.append(extrinsic)
    intrinsics.append(intrinsic)

    depth_path = os.path.join(PATH.replace('images', 'ACMM_filter_idx'), '%d.dmb'%i)
    depth = read_depth(depth_path)
    depth[np.isnan(depth)] = 0
    depth[np.isinf(depth)] = 0
    min_depth, max_depth = np.percentile(depth, [5, 95])
    if max_depth > maxval: 
        maxval = max_depth
    depth[depth < min_depth] = min_depth
    depth[depth > max_depth] = max_depth
    #print(depth.min(), depth.max(), min_depth, max_depth)
    index = i
    save_path = os.path.join('./results/', dirname, 'depth_colmap_dense/depth/frame_%06d.png'%index)
    inv_depth = 1.0 / depth
    #ix = np.isinf(inv_depth) | (inv_depth < 0)
    #inv_depth[ix] = float('nan')
    inv_depth[np.isnan(inv_depth)] = 0
    inv_depth[np.isinf(inv_depth)] = 0
    inv_depth = cv2.resize(inv_depth, (216, 384), interpolation=cv2.INTER_NEAREST) # 会出现小值, 用nearest
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    #print(inv_depth.min(), inv_depth.max(), np.unique(inv_depth))
    image_io.save_raw_float32_image(save_path[:-3]+'raw', inv_depth)
with SuppressedStdout():
    src_path = os.path.join('results', dirname, 'depth_colmap_dense/depth')
    visualization.visualize_depth_dir(src_path, src_path, force=True, min_percentile=0, max_percentile=99)

meta_path = os.path.join('./results/', dirname, 'colmap_dense/metadata.npz')
if not os.path.exists(os.path.dirname(meta_path)):
    os.makedirs(os.path.dirname(meta_path))

extrinsics = np.stack(extrinsics, axis=0)
intrinsics = np.stack(intrinsics, axis=0)
np.savez(meta_path, intrinsics=intrinsics, extrinsics=extrinsics)

print(maxval)
