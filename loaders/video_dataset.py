#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import os
import cv2
from os.path import join as pjoin
import json
import math
import numpy as np
import torch.utils.data as data
import torch
from typing import Optional
import matplotlib.pyplot as plt

from utils import image_io, frame_sampling as sampling
from sparse import get_sparse_depth
from IPython import embed
from pdb import set_trace


_dtype = torch.float32


def Bgr2Yuv(image):
    yuv = np.clip(image.copy(),0,255).astype(np.float32)
    yuv[:,:,0] = 0.299*image[:,:,2] + 0.587*image[:,:,1] + 0.114*image[:,:,0]
    yuv[:,:,1] = 0.492*(image[:,:,0] - yuv[:,:,0]) + 128
    yuv[:,:,2] = 0.877*(image[:,:,2] - yuv[:,:,0]) + 128
    image = np.clip(yuv, 0, 255)
    return image.astype('uint8')

def load_image(
    path: str,
    channels_first: bool,
    check_channels: Optional[int] = None,
    post_proc_raw=lambda x: x,
    post_proc_other=lambda x: x,
) -> torch.FloatTensor:
    if os.path.splitext(path)[-1] == ".raw":
        im = image_io.load_raw_float32_image(path)
        im = post_proc_raw(im)
    else:
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        im = post_proc_other(im)
    im = im.reshape(im.shape[:2] + (-1,))

    if check_channels is not None:
        assert (
            im.shape[-1] == check_channels
        ), "receive image of shape {} whose #channels != {}".format(
            im.shape, check_channels
        )

    if channels_first:
        im = im.transpose((2, 0, 1))
    # to torch
    return torch.tensor(im, dtype=_dtype)


def load_color(path: str, channels_first: bool) -> torch.FloatTensor:
    """
    Returns:
        torch.tensor. color in range [0, 1]
    """
    im = load_image(
        path,
        channels_first,
        post_proc_raw=lambda im: im[..., [2, 1, 0]] if im.ndim == 3 else im,
        post_proc_other=lambda im: im / 255,
    )
    return im


def load_flow(path: str, channels_first: bool) -> torch.FloatTensor:
    """
    Returns:
        flow tensor in pixels.
    """
    flow = load_image(path, channels_first, check_channels=2)
    return flow


def load_mask(path: str, channels_first: bool) -> torch.ByteTensor:
    """
    Returns:
        mask takes value 0 or 1
    """
    mask = load_image(path, channels_first, check_channels=1) > 0
    return mask.to(_dtype)


class VideoDataset(data.Dataset):
    """Load 3D video frames and related metadata for optimizing consistency loss.
    File organization of the corresponding 3D video dataset should be
        color_down/frame_{__ID__:06d}.raw
        flow/flow_{__REF_ID__:06d}_{__TGT_ID__:06d}.raw
        mask/mask_{__REF_ID__:06d}_{__TGT_ID__:06d}.png
        metadata.npz: {'extrinsics': (N, 3, 4), 'intrinsics': (N, 4)}
        <flow_list.json>: [[i, j], ...]
    """

    def __init__(self, path: str, meta_file: str = None, model_type: str = None):
        """
        Args:
            path: folder path of the 3D video
        """
        self.model_type = model_type
        self.color_fmt = pjoin(path, "color_down", "frame_{:06d}.raw")
        if not os.path.isfile(self.color_fmt.format(0)):
            self.color_fmt = pjoin(path, "color_down", "frame_{:06d}.png")

        self.mask_fmt = pjoin(path, "mask", "mask_{:06d}_{:06d}.png")
        self.flow_fmt = pjoin(path, "flow", "flow_{:06d}_{:06d}.raw")

        if meta_file is not None:
            with open(meta_file, "rb") as f:
                meta = np.load(f)
                self.extrinsics = torch.tensor(meta["extrinsics"], dtype=_dtype)
                self.intrinsics = torch.tensor(meta["intrinsics"], dtype=_dtype)
            assert (
                self.extrinsics.shape[0] == self.intrinsics.shape[0]
            ), "#extrinsics({}) != #intrinsics({})".format(
                self.extrinsics.shape[0], self.intrinsics.shape[0]
            )

        flow_list_fn = pjoin(path, "flow_list.json")
        if os.path.isfile(flow_list_fn):
            with open(flow_list_fn, "r") as f:
                self.flow_indices = json.load(f)
        else:
            names = os.listdir(os.path.dirname(self.flow_fmt))
            self.flow_indices = [
                self.parse_index_pair(name)
                for name in names
                if os.path.splitext(name)[-1] == os.path.splitext(self.flow_fmt)[-1]
            ]
            self.flow_indices = sampling.to_in_range(self.flow_indices)
        self.flow_indices = list(sampling.SamplePairs.to_one_way(self.flow_indices))

    def parse_index_pair(self, name):
        strs = os.path.splitext(name)[0].split("_")[-2:]
        return [int(s) for s in strs]

    def __getitem__(self, index: int):
        """Fetch tuples of data. index = i * (i-1) / 2 + j, where i > j for pair (i,j)
        So [-1+sqrt(1+8k)]/2 < i <= [1+sqrt(1+8k))]/2, where k=index. So
            i = floor([1+sqrt(1+8k))]/2)
            j = k - i * (i - 1) / 2.

        The number of image frames fetched, N, is not the 1, but computed
        based on what kind of consistency to be measured.
        For instance, geometry_consistency_loss requires random pairs as samples.
        So N = 2.
        If with more losses, say triplet one from temporal_consistency_loss. Then
            N = 2 + 3.

        Returns:
            stacked_images (N, C, H, W): image frames
            targets: {
                'extrinsics': torch.tensor (N, 3, 4), # extrinsics of each frame.
                                Each (3, 4) = [R, t].
                                    point_wolrd = R * point_cam + t
                'intrinsics': torch.tensor (N, 4), # (fx, fy, cx, cy) for each frame
                'geometry_consistency':
                    {
                        'indices':  torch.tensor (2),
                                    indices for corresponding pairs
                                        [(ref_index, tgt_index), ...]
                        'flows':    ((2, H, W),) * 2 in pixels.
                                    For k in range(2) (ref or tgt),
                                        pixel p = pixels[indices[b, k]][:, i, j]
                                    correspond to
                                        p + flows[k][b, :, i, j]
                                    in frame indices[b, (k + 1) % 2].
                        'masks':    ((1, H, W),) * 2. Masks of valid flow matches
                                    to compute the consistency in training.
                                    Values are 0 or 1.
                    }
            }

        """
        pair = self.flow_indices[index]

        indices = torch.tensor(pair)
        intrinsics = torch.stack([self.intrinsics[k] for k in pair], dim=0)
        extrinsics = torch.stack([self.extrinsics[k] for k in pair], dim=0)

        if 'our' in self.model_type:
            imgs = []
            for k in pair:
                image = cv2.imread(self.color_fmt.format(k).replace('down', 'down_png')[:-3]+'png', cv2.IMREAD_UNCHANGED)
                #image = cv2.resize(image, (128,128),interpolation=cv2.INTER_LINEAR)
                image = Bgr2Yuv(image) if '320' not in self.model_type else image
                image = image.astype(np.float32)
                image /= 255.0
                if '320' in self.model_type: image *= 4.0
                image = torch.from_numpy(image.astype('float32')).permute(2, 0, 1)
                imgs.append(image)
            images = torch.stack(imgs, dim=0)
        else:
            images = torch.stack(
                [load_color(self.color_fmt.format(k), channels_first=True) for k in pair],
                dim=0,
            )
        flows = [
            load_flow(self.flow_fmt.format(k_ref, k_tgt), channels_first=True)
            for k_ref, k_tgt in [pair, pair[::-1]]
        ]
        masks = [
            load_mask(self.mask_fmt.format(k_ref, k_tgt), channels_first=True)
            for k_ref, k_tgt in [pair, pair[::-1]]
        ]

        metadata = {
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "geometry_consistency": {
                "indices": indices,
                "flows": flows,
                "masks": masks,
            },
        }

        if getattr(self, "scales", None):
            if isinstance(self.scales, dict):
                metadata["scales"] = torch.stack(
                    [torch.Tensor([self.scales[k]]) for k in pair], dim=0
                )
            else:
                metadata["scales"] = torch.Tensor(
                    [self.scales, self.scales]).reshape(2, 1)
        depths, guides = [], []
        for k in pair:
            if 'down_png' in self.color_fmt:
                depth_path = self.color_fmt.format(k).replace('color_down_png', 'depth_colmap_dense/depth')
            else:
                depth_path = self.color_fmt.format(k).replace('color_down', 'depth_colmap_dense/depth')

            if os.path.exists(depth_path[:-3]+'raw'):
                depth = image_io.load_raw_float32_image(depth_path[:-3]+'raw')
                h, w = images[0].shape[-2:]
                depth = 1.0 / depth
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST) #/ 10
                depth[np.isnan(depth)] = 0
                depth[np.isinf(depth)] = 0
                depth[depth<0.5] = 0
                depth[depth>15] = 0
                # print(np.unique(depth))
                # print(depth.max())
                depth = torch.from_numpy(depth.astype('float32')).unsqueeze(0) / 5
                gmask = torch.zeros_like(depth)
                """num = np.random.randint(16, 25)
                for ii in range(num):
                    x, y = np.random.randint(w-7), np.random.randint(h-7)
                    gmask[:,y:y+7,x:x+7] = 1
                gmask[depth==0] = 0
                #print((depth*gmask).max())"""
                gmask[depth>0] = 1
                #gmask = 1 - gmask # 模拟大模型补全
                guides.append(torch.cat([depth*gmask, gmask], dim=0))
                depths.append(depth)
        depths = torch.stack(depths, dim=0) if len(depths) > 0 else None
        guides = torch.stack(guides, dim=0) if len(guides) > 0 else None
        metadata['depth'] = depths
        metadata['guide'] = guides
        #set_trace()
        return (images, metadata)

    def __len__(self):
        return len(self.flow_indices)


class VideoFrameDataset(data.Dataset):
    """Load video frames from
        color_fmt.format(frame_id)
    """

    def __init__(self, color_fmt, frames=None, model_type=''):
        """
        Args:
            color_fmt: e.g., <video_dir>/frame_{:06d}.raw
        """
        self.color_fmt = color_fmt
        self.model_type = model_type

        if frames is None:
            files = os.listdir(os.path.dirname(self.color_fmt))
            self.frames = range(len(files))
        else:
            self.frames = frames

    def __getitem__(self, index):
        """Fetch image frame.
        Returns:
            image (C, H, W): image frames
        """
        frame_id = self.frames[index]
        if 'our' in self.model_type:
            image_path = self.color_fmt.format(frame_id).replace('down', 'down_png')[:-3]+'png' if 'down_png' not in self.color_fmt else self.color_fmt.format(frame_id)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            #image = cv2.resize(image, (128,128),interpolation=cv2.INTER_LINEAR)
            try:
                image = Bgr2Yuv(image) if '320' not in self.model_type else image
            except:
                set_trace()
            image = image.astype(np.float32)
            image /= 255.0
            if '320' in self.model_type: image *= 4.0
            image = torch.from_numpy(image.astype('float32')).permute(2, 0, 1)
        else:
            image = load_color(self.color_fmt.format(frame_id), channels_first=True)

        if 'down_png' in self.color_fmt:
            depth_path = self.color_fmt.format(frame_id).replace('color_down_png', 'depth_colmap_dense/depth')
        else:
            depth_path = self.color_fmt.format(frame_id).replace('color_down', 'depth_colmap_dense/depth')

        meta = {"frame_id": frame_id}
        if os.path.exists(depth_path[:-3]+'raw'):
            depth = image_io.load_raw_float32_image(depth_path[:-3]+'raw')
            h, w = image.shape[-2:]
            depth = 1.0 / depth
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST) #/ 10
            depth[np.isnan(depth)] = 0
            depth[np.isinf(depth)] = 0
            depth[depth<0.5] = 0
            depth[depth>15] = 0
            #print(np.unique(depth), frame_id)
            # print(depth.max(), frame_id)
            depth = torch.from_numpy(depth.astype('float32')).unsqueeze(0) / 5
            gmask = torch.zeros_like(depth)
            """num = np.random.randint(16, 25)
            for ii in range(num):
                x, y = np.random.randint(w-7), np.random.randint(h-7)
                gmask[:,y:y+7,x:x+7] = 1
            gmask[depth==0] = 0
            print((depth*gmask).max())"""
            gmask[depth>0] = 1
            #plt.imsave('guide/%d.jpg'%frame_id, (depth*gmask)[0].cpu().numpy(), cmap='jet')
            #gmask = 1 - gmask # 模拟大模型补全
            guide = torch.cat([depth*gmask, gmask], dim=0)
            meta['depth'] = depth
            meta['guide'] = guide
        #set_trace()
        return image, meta

    def __len__(self):
        return len(self.frames)
