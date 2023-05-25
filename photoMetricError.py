import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

fx = fy = 518.8579

class pMError(nn.Module):
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(pMError, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps
        
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False) # u,v 1
        self.K = nn.Parameter(torch.Tensor([[fx*self.width/640, 0, self.width/2, 0],
                                            [0, fy*self.height/480, self.height/2, 0],
                                            [0, 0, 1, 0], [0, 0, 0, 1]]), requires_grad=False)
        self.inv_K = self.K.inverse()
        self.K = nn.Parameter(self.K.repeat(batch_size, 1, 1), requires_grad=False)
        self.inv_K = nn.Parameter(self.inv_K.repeat(batch_size, 1, 1), requires_grad=False)

    def depth2Mesh(self, depth):
        cam_points = torch.matmul(self.inv_K[:, :3, :3], self.pix_coords)
        cam_points = cam_points.to(depth.device)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, torch.ones_like(cam_points[:,-1:])], 1)

        return cam_points
    
    def project(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)
        
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords

    def forward(self, image0, image1, pred0, pred1, mask0, mask1, T0, T1):
        
        # pixel coorinate transform from 1-0
        cam_points0 = self.depth2Mesh(pred0)
         
        pix_coords0_1 = self.project(cam_points0, self.K.to(T0.device), T0)
        pred1_0 = F.grid_sample(pred1, pix_coords0_1, padding_mode="zeros")
        image1_0 = F.grid_sample(image1, pix_coords0_1, padding_mode="zeros")
        mask0 = mask0 * (pred1_0>0).float()
        diff1_0 = mask0*torch.abs(pred1_0 - 0)
        loss1_0 = torch.sum(diff1_0)/torch.sum(mask0)
        loss1_0.backward()
        
        # pixel coorinate transform from 1-2
        cam_points1 = self.depth2Mesh(pred1)
        
        pix_coords1_0 = self.project(cam_points1, self.K.to(T0.device), T1)
        pred0_1 = F.grid_sample(pred0, pix_coords1_0, padding_mode="zeros")
        image0_1 = F.grid_sample(image0, pix_coords1_0, padding_mode="zeros")
        mask1 = mask1 * (pred0_1>0).float()
        diff0_1 = mask1*torch.abs(pred0_1 - 0)
        loss0_1 = torch.sum(diff0_1)/torch.sum(mask1)
        self.save_img_show('show/',pred0_1, 'pred0_1.jpg')
        self.save_img_show('show/',pred1_0, 'pred1_0.jpg')
        self.save_img_show('show/',pred0, 'pred0.jpg')
        self.save_img_show('show/',pred1, 'pred1.jpg')
        self.save_img_show('show/',(image0_1+image1)/2.0, 'image0_1.jpg')
        self.save_img_show('show/',(image1_0+image0)/2.0, 'image1_0.jpg')
        self.save_img_show('show/',image0, 'image0.jpg')
        self.save_img_show('show/',image1, 'image1.jpg')
        self.save_img_show('show/',pred0.grad, 'grad0.jpg')
        #self.save_img_show('show/',pred1.grad, 'grad1.jpg')
        return loss1_0 + loss0_1
        
    def save_img_show(self, prefix, data, suffix):
        if data.size(1) == 3:
            imgs = data.cpu().detach().numpy().transpose(0,2,3,1)[:,:,:,::-1]
        else:
            imgs = data.cpu().detach().numpy().transpose(0,2,3,1).squeeze(3)
        for i in range(len(imgs)):
            plt.imsave(prefix+'%02d'%i+suffix, imgs[i]/imgs[i].max(), cmap='jet')
        
        
if __name__ == '__main__':
    path = '/home/anshijie/dataset/scannet_parse/rgb/scene0000_00'
    image0 = cv2.imread(os.path.join(path, '4.jpg')).astype('float32')
    image1 = cv2.imread(os.path.join(path, '9.jpg')).astype('float32')

    depth0 = cv2.imread(os.path.join(path.replace('rgb', 'depth'), '4.png'), -1).astype('float32')/1000
    depth1 = cv2.imread(os.path.join(path.replace('rgb', 'depth'), '9.png'), -1).astype('float32')/1000

    pose0 = np.loadtxt(os.path.join(path.replace('rgb', 'pose'), '4.txt')).astype('float32')
    pose1 = np.loadtxt(os.path.join(path.replace('rgb', 'pose'), '9.txt')).astype('float32')
    
    h, w = depth0.shape
    image0 = cv2.resize(image0, (w,h), interpolation=cv2.INTER_AREA)
    image1 = cv2.resize(image1, (w,h), interpolation=cv2.INTER_AREA)
    
    images = torch.stack([torch.from_numpy(image0), torch.from_numpy(image1)]).cuda(0).permute(0, 3, 1, 2)
    image0 = torch.autograd.Variable(images[:1], requires_grad=True)
    image1 = torch.autograd.Variable(images[1:], requires_grad=True)
    depths = torch.stack([torch.from_numpy(depth0), torch.from_numpy(depth1)]).cuda(0).unsqueeze(1)
    depth0 = torch.autograd.Variable(depths[:1], requires_grad=True)
    depth1 = torch.autograd.Variable(depths[1:], requires_grad=True)
    T0_1 = torch.matmul(torch.from_numpy(pose1).inverse(), torch.from_numpy(pose0))
    T0 = T0_1.cuda(0).unsqueeze(0)
    T1 = T0_1.inverse().cuda(0).unsqueeze(0)
    mask = (depths > 0).float()
    pmerror = pMError(1, 480, 640)
    pmerror(image0, image1, depth0, depth1, mask[:1], mask[1:], T0, T1)
#     embed()
