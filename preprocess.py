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

PATH = './normal_motion2/'
#PATH = './new/'

MIN = 200
NUM = 10000

def make_video(dirpath, outpath):
    num = min(NUM, len(os.listdir(dirpath)))
    for i in range(MIN, num):
        image_path = os.path.join(dirpath, 'vio_%08d.jpg'%i)
        image = cv2.imread(image_path)
        if i == MIN:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            fps = 30
            h, w = image.shape[:2]
            videoWriter = cv2.VideoWriter(outpath + '.avi', fourcc, fps, (w, h))
        videoWriter.write(image)

    videoWriter.release()
    print('video generate done!')

for dirname in os.listdir(PATH):
    if dirname not in ['02', '04', '06', '08', '10']:
        continue
    print('processing: ', dirname)
    dirpath = os.path.join(PATH, dirname, 'images')
    dirpath2 = os.path.join(PATH, dirname, 'depths')
    outpath = os.path.join('data/videos/', dirname+'s2')
    extrinsics = np.loadtxt(os.path.join(PATH, dirname, 'camera_poses.txt'))
    intrinsics = np.loadtxt(os.path.join(PATH, dirname, 'camera_intrinsic.txt'))
    intrinsics_scale = intrinsics[:4][[0,1,3,2]] / intrinsics[4] * 384
    #r = R.from_quat(extrinsics[0][[3,4,5,2]]).as_matrix()
    RT_matrixs = []
    NUM = len(extrinsics)
    for i, it in enumerate(extrinsics):
        if i < MIN:
            continue
        if i >= NUM and 'new' not in PATH:
            break
        r = R.from_quat(it[[3,4,5,2]])
        try:
            rotate_matrix = r.as_matrix()
        except:
            embed()
        trans_vector = it[6:].reshape(3, 1)
        #rotate_matrix = ROT_COLMAP_TO_NORMAL.dot(rotate_matrix).dot(ROT_COLMAP_TO_NORMAL.T)
        #trans_vector = ROT_COLMAP_TO_NORMAL.dot(trans_vector)
        RT_matrixs.append(np.concatenate([rotate_matrix, trans_vector], axis=1))
        
    RT = np.array(RT_matrixs)
    intrinsics_new = np.tile(intrinsics_scale, (len(RT), 1))
    
    print('combine videos...')
    #mk.make_video('ffmpeg', os.path.join(dirpath, 'vio_000000%02d.jpg'), outpath)
    make_video(dirpath, outpath)

    print('generate depths...')
    for i, fname in enumerate(os.listdir(dirpath2)):
        if not fname.startswith('depth'):
            continue
        depth_path = os.path.join(dirpath2, fname)
        depth = cv2.imread(depth_path, -1) / 1000.0
        # arkit有效深度是4m
        depth = np.clip(depth, 0, 4)
        index = int(fname.split('.')[0].split('_')[1])
        if (index < MIN or index >= NUM) and 'new' not in PATH:
            continue
        save_path = os.path.join('./results/', dirname+'s2', 'depth_colmap_dense/depth/frame_%06d.png'%(index-MIN))
        inv_depth = 1.0 / depth
        ix = np.isinf(inv_depth) | (inv_depth < 0)
        inv_depth[ix] = float('nan')
        inv_depth = cv2.resize(inv_depth, (216, 384), cv2.INTER_LINEAR)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        image_io.save_raw_float32_image(save_path[:-3]+'raw', inv_depth)
    with SuppressedStdout():
        src_path = os.path.join('results', dirname+'s2', 'depth_colmap_dense/depth')
        visualization.visualize_depth_dir(src_path, src_path, force=True, min_percentile=0, max_percentile=99)
    
    meta_path = os.path.join('./results/', dirname+'s2', 'colmap_dense/metadata.npz')
    if not os.path.exists(os.path.dirname(meta_path)):
        os.makedirs(os.path.dirname(meta_path))
    np.savez(meta_path, intrinsics=intrinsics_new, extrinsics=RT)


