import os
import cv2
import random
import numpy as np
from scipy.spatial.transform import Rotation as R

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython import embed
#from tools import runtime
import time


PATH = './normal_motion2/08'
intrinsics = np.loadtxt(os.path.join(PATH, 'camera_intrinsic.txt'))

fx = fy = intrinsics[0] #518.8579
cx = intrinsics[3]
cy = intrinsics[2]

ROT_COLMAP_TO_NORMAL = np.array([[1.0, 0.0, 0.0, 0.03],
                                 [0.0, -1.0, 0.0, 0.02],
                                 [0.0, 0.0, -1.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]])

class Projection(object):
    def __init__(self, height, width, eps=1e-12):
        super(Projection, self).__init__()
        self.height = height
        self.width = width
        self.eps=eps
        
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)

        self.ones = np.ones((1, self.height * self.width))
        
        self.pix_coords = np.concatenate([self.id_coords[0].reshape(1,-1),
                                          self.id_coords[1].reshape(1,-1), self.ones
                                         ], 0)
        
        self.K = np.array([[fx, 0, cx, 0], 
                           [0, fy, cy, 0],
                           [0, 0, 1, 0], [0, 0, 0, 1]])
        self.inv_K = np.linalg.inv(self.K)
    
    def depth2Mesh(self, depth):
        cam_points = np.matmul(self.inv_K[:3, :3], self.pix_coords)
        cam_points = depth.reshape(1, -1) * cam_points
        cam_points = np.concatenate([cam_points, np.ones_like(cam_points[:1])], 0)

        return cam_points
    
    def project(self, points, K, T):
        P = np.matmul(K, T)[:3, :]
        cam_points = np.matmul(P, points)

        return cam_points
        
    #@runtime
    def __call__(self, image, depth, mask, T):
        image = image.astype('float32')
        h, w = depth.shape
        pts0 = self.depth2Mesh(depth)   # 4 x hw
        pts1 = self.project(pts0, self.K, T)    # 3 x hw
        #depth0 = pts1[2,:].reshape(h, w)
        
        coord = pts1[:2] / (pts1[2,:] + self.eps)   # 2 x hw

        # find top-left, top-right, down-left and down-right, 2 x hw
        tl = np.floor(coord[:2]).astype('int32')
        tr = np.concatenate([np.ceil(coord[:1]), np.floor(coord[1:2])], 0).astype('int32')
        dl = np.concatenate([np.floor(coord[:1]), np.ceil(coord[1:2])], 0).astype('int32')
        dr = np.ceil(coord[:2]).astype('int32')
        #nr = np.round(coord[:2]).astype('int32')

        dtl = 2 - np.sqrt(np.sum((coord[:2] - tl)**2, axis=0)).reshape(-1,1)
        dtr = 2 - np.sqrt(np.sum((coord[:2] - tr)**2, axis=0)).reshape(-1,1)
        ddl = 2 - np.sqrt(np.sum((coord[:2] - dl)**2, axis=0)).reshape(-1,1)
        ddr = 2 - np.sqrt(np.sum((coord[:2] - dr)**2, axis=0)).reshape(-1,1)

        start = time.time()
        image0 = image.reshape(-1, 3)
        depth0 = depth.reshape(-1, 1)
        mask0 = mask.reshape(-1, 1)
        image1 = np.zeros_like(image0)
        depth1 = np.zeros_like(depth0)
        #depth2 = np.zeros_like(depth0)
        mask1 = np.zeros_like(mask0)
        coef = np.zeros_like(depth0)
        # 矩阵操作，取索引，幅值
        tl[0,:] = np.clip(tl[0,:], 0, w-1)
        tl[1,:] = np.clip(tl[1,:], 0, h-1)
        dr[0,:] = np.clip(dr[0,:], 0, w-1)
        dr[1,:] = np.clip(dr[1,:], 0, h-1)
        tr[0,:] = np.clip(tr[0,:], 0, w-1)
        tr[1,:] = np.clip(tr[1,:], 0, h-1)
        dl[0,:] = np.clip(dl[0,:], 0, w-1)
        dl[1,:] = np.clip(dl[1,:], 0, h-1)
        #nr[0,:] = np.clip(nr[0,:], 0, w-1)
        #nr[1,:] = np.clip(nr[1,:], 0, h-1)
        
        image1[tl[1,:]*w+tl[0,:]] += dtl * image0
        depth1[tl[1,:]*w+tl[0,:]] += dtl * depth0
        mask1[tl[1,:]*w+tl[0,:]] += dtl * mask0
        coef[tl[1,:]*w+tl[0,:]] += dtl

        image1[tr[1,:]*w+tr[0,:]] += dtr * image0
        depth1[tr[1,:]*w+tr[0,:]] += dtr * depth0
        mask1[tr[1,:]*w+tr[0,:]] += dtr * mask0
        coef[tr[1,:]*w+tr[0,:]] += dtr

        image1[dl[1,:]*w+dl[0,:]] += ddl * image0
        depth1[dl[1,:]*w+dl[0,:]] += ddl * depth0
        mask1[dl[1,:]*w+dl[0,:]] += ddl * mask0
        coef[dl[1,:]*w+dl[0,:]] += ddl

        image1[dr[1,:]*w+dr[0,:]] += ddr * image0
        depth1[dr[1,:]*w+dr[0,:]] += ddr * depth0
        mask1[dr[1,:]*w+dr[0,:]] += ddr * mask0
        coef[dr[1,:]*w+dr[0,:]] += ddr

        image1 /= (coef + self.eps)
        depth1 /= (coef + self.eps)
        mask1 /= (coef + self.eps)
        #depth2[nr[1,:]*w+nr[0,:]] = depth0
        image1 = image1.reshape(h,w,-1)
        depth1 = depth1.reshape(h,w)
        #depth2 = depth2.reshape(h,w)
        mask1 = mask1.reshape(h,w)
        #print(time.time() - start)
        # 显示结果
        #s = '_'.join(['%.2f'%i for i in T[:3].flatten()])
        #self.save_img_show('show/image0.jpg', image)
        #self.save_img_show('show/depth0.jpg', depth)
        #self.save_img_show('show/image1{}.jpg'.format(s), image1)
        #self.save_img_show('show/depth1{}.jpg'.format(s), depth1)
        image2 = self.inpaint(image1, depth1)
        mask1[mask1<=0.5] = 0
        mask1[mask1>0.5] = 1
        #self.save_img_show('show/image2{}.jpg'.format(s), image2)
        return image2, depth1, mask1
    
    def inpaint(self, image, depth):
        image = image.astype('uint8')
        _, mask = cv2.threshold(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),10,255,cv2.THRESH_BINARY_INV)
        #_, mask = cv2.threshold(depth, 10, 255, cv2.THRESH_BINARY_INV)
        dst = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        #self.save_img_show('show/image1_fill.jpg', dst)
        return dst

    def get_matrix(self, angle):
        # 分别绕xyz轴旋转
        seed = np.random.randint(3)
        tx = (np.random.rand() - 0.5) / 5
        ty = (np.random.rand() - 0.5) / 5
        tz = (np.random.rand() - 0.5) / 5
        if seed == 0:
            ax = angle * np.pi / 180
            R = np.array([[1, 0, 0, 0], [0, np.cos(ax), -np.sin(ax), 0], [0, np.sin(ax), np.cos(ax), 0], [0, 0, 0, 1]])
        elif seed == 1:
            ay = angle * np.pi / 180
            R = np.array([[np.cos(ay), 0, -np.sin(ay), 0], [0, 1, 0, 0], [np.sin(ay), 0, np.cos(ay), 0], [0, 0, 0, 1]])
        else:
            az = angle * np.pi / 180
            R = np.array([[np.cos(az), -np.sin(az), 0, 0], [np.sin(az), np.cos(az), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        R[0,-1] = tx
        R[1,-1] = ty
        R[2,-1] = tz
        #print(R, seed)
        return R.astype('float32'), seed
        
    def save_img_show(self, fname, imgs):
        if imgs.ndim == 3:
            imgs = imgs[:,:,::-1]
        plt.imsave(fname, (imgs-imgs.min())/(imgs.max()-imgs.min()), cmap='jet')


if __name__ == '__main__':
    idx = 50
    path1 = './normal_motion2/08/images/vio_00000000.jpg'
    path2 = './normal_motion2/08/images/vio_00000%03d.jpg'%(idx)
    image1 = cv2.imread(path1).astype('float32')
    image2 = cv2.imread(path2).astype('float32')
    depth1 = cv2.imread(path1.replace('images','depths').replace('vio','depth')[:-3]+'png', -1) / 1000.
    depth2 = cv2.imread(path2.replace('images','depths').replace('vio','depth')[:-3]+'png', -1) / 1000.
    h, w = image1.shape[:2]
    depth1 = cv2.resize(depth1, (w,h), interpolation=cv2.INTER_LINEAR)
    depth2 = cv2.resize(depth2, (w,h), interpolation=cv2.INTER_LINEAR)
    mask1 = (depth1 > 0).astype('float32')
    mask2 = (depth2 > 0).astype('float32')

    extrinsics = np.loadtxt(os.path.join(PATH, 'camera_poses.txt'))
    pose1, pose2 = np.eye(4), np.eye(4)
    r = R.from_quat(extrinsics[0][[3,4,5,2]])
    pose1[:3,:3] = r.as_matrix()
    pose1[:3,3:] = extrinsics[0][6:].reshape(3, 1)
    print(pose1)
    #pose1 = np.dot(pose1, ROT_COLMAP_TO_NORMAL)
    #pose1 = ROT_COLMAP_TO_NORMAL.dot(pose1).dot(ROT_COLMAP_TO_NORMAL.T)
    pose1 = np.linalg.inv(ROT_COLMAP_TO_NORMAL).dot(pose1).dot(ROT_COLMAP_TO_NORMAL)
    #pose1 = ROT_COLMAP_TO_NORMAL.dot(pose1).dot(np.linalg.inv(ROT_COLMAP_TO_NORMAL))
    r = R.from_quat(extrinsics[idx][[3,4,5,2]])
    pose2[:3,:3] = r.as_matrix()
    pose2[:3,3:] = extrinsics[idx][6:].reshape(3, 1)
    #pose2 = np.dot(pose2, ROT_COLMAP_TO_NORMAL)
    #pose2 = ROT_COLMAP_TO_NORMAL.dot(pose2).dot(ROT_COLMAP_TO_NORMAL.T) # better ??
    pose2 = np.linalg.inv(ROT_COLMAP_TO_NORMAL).dot(pose2).dot(ROT_COLMAP_TO_NORMAL) # same
    #pose2 = ROT_COLMAP_TO_NORMAL.dot(pose2).dot(np.linalg.inv(ROT_COLMAP_TO_NORMAL)) # little better
    print(pose1)
    T = np.dot(np.linalg.inv(pose2), pose1) # 重投影不对
    #T = np.dot(pose2, np.linalg.inv(pose1)) # 重投影更不对
    #T = np.load('/home/anshijie/dataset/T1-10.npy', allow_pickle=True).item()[5]#.astype('float32')

    p = Projection(h, w)
    image1_2, depth1_2, mask1_2 = p(image1, depth1, mask1, T)
    p.save_img_show('show2/image1.jpg', image1)
    p.save_img_show('show2/image2.jpg', image2)
    image_warp = cv2.addWeighted(image2.astype('uint8'), 0.5, image1_2.astype('uint8'), 0.5, 0)
    p.save_img_show('show2/image2_1v3.jpg', image1_2)
    p.save_img_show('show2/image2_warpv3.jpg', image_warp)
    p.save_img_show('show2/depth1.jpg', depth1)
    p.save_img_show('show2/depth2.jpg', depth2)
    p.save_img_show('show2/depth1_2.jpg', depth1_2)
    p.save_img_show('show2/mask1.jpg', mask1)
    p.save_img_show('show2/mask2.jpg', mask2)
    p.save_img_show('show2/mask1_2.jpg', mask1_2)

