#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import cv2
import torch
import torch.nn as nn
from utils.torch_helpers import _device
from utils.geometry import (
    pixel_grid,
    focal_length,
    project,
    pixels_to_points,
    reproject_points,
    sample,
)
from pdb import set_trace


def select_tensors(x):
    """
    x (B, N, C, H, W) -> (N, B, C, H, W)
    Each batch (B) is composed of a pair or more samples (N).
    """
    return x.transpose(0, 1)


def weighted_mse_loss(input, target, weights, dim=1, eps=1e-6):
    """
        Args:
            input (B, C, H, W)
            target (B, C, H, W)
            weights (B, 1, H, W)

        Returns:
            scalar
    """
    assert (
        input.ndimension() == target.ndimension()
        and input.ndimension() == weights.ndimension()
    )
    # normalize to sum=1
    B = weights.shape[0]
    weights_sum = torch.sum(weights.view(B, -1), dim=-1).view(B, 1, 1, 1)
    weights_sum = torch.clamp(weights_sum, min=eps)
    weights_n = weights / weights_sum

    sq_error = torch.sum((input - target) ** 2, dim=dim, keepdim=True)  # BHW
    return torch.sum((weights_n * sq_error).reshape(B, -1), dim=1)


def weighted_rmse_loss(input, target, weights, dim=1, eps=1e-6):
    """
        Args:
            input (B, C, H, W)
            target (B, C, H, W)
            weights (B, 1, H, W)

        Returns:
            scalar = weighted_mean(rmse_along_dim)
    """
    assert (
        input.ndimension() == target.ndimension()
        and input.ndimension() == weights.ndimension()
    )
    # normalize to sum=1
    B = weights.shape[0]
    weights_sum = torch.sum(weights.view(B, -1), dim=-1).view(B, 1, 1, 1)
    weights_sum = torch.clamp(weights_sum, min=eps)
    weights_n = weights / weights_sum

    diff = torch.norm(input - target, dim=dim, keepdim=True)
    return torch.sum((weights_n * diff).reshape(B, -1), dim=1)


def weighted_mean_loss(x, weights, eps=1e-6):
    """
        Args:
            x (B, ...)
            weights (B, ...)

        Returns:
            a scalar
    """
    assert x.ndimension() == weights.ndimension() and x.shape[0] == weights.shape[0]
    # normalize to sum=1
    B = weights.shape[0]
    weights_sum = torch.sum(weights.view(B, -1), dim=-1).view(B, 1, 1, 1)
    weights_sum = torch.clamp(weights_sum, min=eps)
    weights_n = weights / weights_sum

    return torch.sum((weights_n * x).reshape(B, -1), dim=1)


class ConsistencyLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.dist = torch.abs

    def geometry_consistency_loss(self, points_cam, metadata, pixels, images):
        """Geometry Consistency Loss.

        For each pair as specified by indices,
            geom_consistency = reprojection_error + disparity_error
        reprojection_error is measured in the screen space of each camera in the pair.

        Args:
            points_cam (B, N, 3, H, W): points in local camera coordinate.
            pixels (B, N, 2, H, W)
            metadata: dictionary of related metadata to compute the loss. Here assumes
                metadata include entries as below.
                {
                    'extrinsics': torch.tensor (B, N, 3, 4), # extrinsics of each frame.
                                    Each (3, 4) = [R, t]
                    'intrinsics': torch.tensor (B, N, 4), # (fx, fy, cx, cy)
                    'geometry_consistency':
                        {
                            'flows':    (B, 2, H, W),) * 2 in pixels.
                                        For k in range(2) (ref or tgt),
                                            pixel p = pixels[indices[b, k]][:, i, j]
                                        correspond to
                                            p + flows[k][b, :, i, j]
                                        in frame indices[b, (k + 1) % 2].
                            'masks':    ((B, 1, H, W),) * 2. Masks of valid flow
                                        matches. Values are 0 or 1.
                        }
                }
        """
        geom_meta = metadata["geometry_consistency"]
        points_cam_pair = select_tensors(points_cam) # N, B, 3, H, W
        extrinsics = metadata["extrinsics"] # N, B, 3, 4
        extrinsics_pair = select_tensors(extrinsics)
        intrinsics = metadata["intrinsics"]
        intrinsics_pair = select_tensors(intrinsics)
        pixels_pair = select_tensors(pixels)
        images_pair = select_tensors(images)

        flows_pair = (flows for flows in geom_meta["flows"])
        masks_pair = (masks for masks in geom_meta["masks"])

        reproj_losses, disp_losses = [], []
        inv_idxs = [1, 0]

        for (
            points_cam_ref,
            tgt_points_cam_tgt,
            image_ref,
            image_tgt,
            pixels_ref,
            flows_ref,
            masks_ref,
            intrinsics_ref,
            intrinsics_tgt,
            extrinsics_ref,
            extrinsics_tgt,
        ) in zip(
            points_cam_pair,
            points_cam_pair[inv_idxs],
            images_pair,
            images_pair[inv_idxs],
            pixels_pair,
            flows_pair,
            masks_pair,
            intrinsics_pair,
            intrinsics_pair[inv_idxs],
            extrinsics_pair,
            extrinsics_pair[inv_idxs],
        ):
            # change to camera space for target_camera
            points_cam_tgt = reproject_points(
                points_cam_ref, extrinsics_ref, extrinsics_tgt
            )
            matched_pixels_tgt = pixels_ref + flows_ref
            pixels_tgt = project(points_cam_tgt, intrinsics_tgt) # 投影-[x,y]

            if self.opt.lambda_reprojection > 0:
                reproj_dist = torch.norm(pixels_tgt - matched_pixels_tgt,
                    dim=1, keepdim=True) # 关于维度1求模长 == L2范数
                masks_dist = (reproj_dist < 10).float()
                reproj_losses.append(
                    weighted_mean_loss(self.dist(reproj_dist), masks_ref*masks_dist) # mask均值误差
                )
            
            if True:
                flow_warped_image = sample(image_tgt, matched_pixels_tgt)
                proj_warped_image = sample(image_tgt, pixels_tgt)
                warped1 = flow_warped_image.cpu().detach().numpy().transpose(0, 2, 3, 1) * 255 / 4
                warped2 = proj_warped_image.cpu().detach().numpy().transpose(0, 2, 3, 1) * 255 / 4
                image_ref = image_ref.cpu().detach().numpy().transpose(0, 2, 3, 1) * 255 / 4
                image_tgt = image_tgt.cpu().detach().numpy().transpose(0, 2, 3, 1) * 255 / 4
                for i, (warp1, warp2, image1, image2) in enumerate(zip(warped1, warped2, image_ref, image_tgt)):
                    image_flow = cv2.addWeighted(image1, 0.5, warp1, 0.5, 0)
                    image_warp = cv2.addWeighted(image1, 0.5, warp2, 0.5, 0)
                    image_comb = cv2.addWeighted(image1, 0.5, image2, 0.5, 0)
                    cv2.imwrite('warped/%02d-warped1.jpg'%i, image_flow.astype('uint8'))
                    cv2.imwrite('warped/%02d-warped2.jpg'%i, image_warp.astype('uint8'))
                    cv2.imwrite('warped/%02d-warped.jpg'%i, warp2.astype('uint8'))
                    cv2.imwrite('warped/%02d-image1.jpg'%i, image1.astype('uint8'))
                    cv2.imwrite('warped/%02d-image2.jpg'%i, image2.astype('uint8'))
                    cv2.imwrite('warped/%02d-imagec.jpg'%i, image_comb.astype('uint8'))
                    set_trace()
            # set_trace()
            if self.opt.lambda_view_baseline > 0:
                # disparity consistency
                f = torch.mean(focal_length(intrinsics_ref)) # mean(fx, fy)
                # warp points in target image grid target camera coordinates to
                # reference image grid
                warped_tgt_points_cam_tgt = sample(
                    tgt_points_cam_tgt, matched_pixels_tgt
                ) # grid sample采样目标点云

                disp_diff = 1.0 / points_cam_tgt[:, -1:, ...] \
                    - 1.0 / warped_tgt_points_cam_tgt[:, -1:, ...] # 视差误差

                masks_dist = (reproj_dist < 3).float()
                disp_losses.append(
                    f * weighted_mean_loss(self.dist(disp_diff), masks_ref*masks_dist)
                )

        B = points_cam_pair[0].shape[0]
        dtype = points_cam_pair[0].dtype
        reproj_loss = (
            self.opt.lambda_reprojection
            * torch.mean(torch.stack(reproj_losses, dim=-1), dim=-1)
            if len(reproj_losses) > 0
            else torch.zeros(B, dtype=dtype, device=_device) # 求均值
        )
        disp_loss = (
            self.opt.lambda_view_baseline
            * torch.mean(torch.stack(disp_losses, dim=-1), dim=-1)
            if len(disp_losses) > 0
            else torch.zeros(B, dtype=dtype, device=_device)
        )

        batch_losses = {"reprojection": reproj_loss, "disparity": disp_loss}
        return torch.mean(reproj_loss + disp_loss), batch_losses

    def __call__(
        self,
        depths,
        metadata,
        images
    ):
        """Compute total loss.

        The network predicts a set of depths results. The number of samples, N, is
        not the batch_size, but computed based on the loss.
        For instance, geometry_consistency_loss requires pairs as samples, then
            N = 2 .
        If with more losses, say triplet one from temporal_consistency_loss. Then
            N = 2 + 3.

        Args:
            depths (B, N, H, W):   predicted_depths
            metadata: dictionary of related metadata to compute the loss. Here assumes
                metadata include data as below. But each loss assumes more.
                {
                    'extrinsics': torch.tensor (B, N, 3, 4), # extrinsics of each frame.
                                    Each (3, 4) = [R, t]
                    'intrinsics': torch.tensor (B, N, 4),
                                  # (fx, fy, cx, cy) for each frame in pixels
                }

        Returns:
            loss: python scalar. And set self.total_loss
        """

        def squeeze(x):
            return x.reshape((-1,) + x.shape[2:])

        def unsqueeze(x, N):
            return x.reshape((-1, N) + x.shape[1:])

        depths = depths.unsqueeze(-3) # B, N, 1, H, W
        intrinsics = metadata["intrinsics"]
        B, N, C, H, W = depths.shape
        pixels = pixel_grid(B * N, (H, W)) # B*N, 2, H, W; [x, y]
        points_cam = pixels_to_points(squeeze(intrinsics), squeeze(depths), pixels) # [X, Y, Z]
        pixels = unsqueeze(pixels, N) # B, N, 2, H, W
        points_cam = unsqueeze(points_cam, N) # B, N, 3, H, W
        images = images.view(B, N, -1, H, W)

        return self.geometry_consistency_loss(points_cam, metadata, pixels, images)
