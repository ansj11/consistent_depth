#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import os
from os.path import join as pjoin
import shutil
import time

from depth_fine_tuning import DepthFineTuner
from flow import Flow
from scale_calibration import calibrate_scale
from tools import make_video as mkvid
from utils.frame_range import FrameRange, OptionalSet
from utils.helpers import print_banner, print_title
from video import (Video, sample_pairs)
from IPython import embed


class DatasetProcessor:
    def __init__(self, writer=None):
        self.writer = writer

    def create_output_path(self, params):
        range_tag = f"R{params.frame_range.name}"
        flow_ops_tag = "-".join(params.flow_ops)
        name = f"{range_tag}_{flow_ops_tag}_{params.model_type}"

        out_dir = pjoin(self.path, name)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def extract_frames(self, params):
        print_banner("Extracting PTS")
        self.video.extract_pts() # 解析frame.txt

        print_banner("Extracting frames")
        self.video.extract_frames() # 解帧

    def pipeline(self, params):
        self.extract_frames(params)

        print_banner("Downscaling frames (raw)") # 下采样
        self.video.downscale_frames("color_down", params.size, "raw")

        print_banner("Downscaling frames (png)") # 长边384
        self.video.downscale_frames("color_down_png", params.size, "png")

        print_banner("Downscaling frames (for flow)") # 长边1024原图for
        self.video.downscale_frames("color_flow", Flow.max_size(), "png", align=64)

        frame_range = FrameRange(
            frame_range=params.frame_range.set, num_frames=self.video.frame_count,
        ) # 帧数
        frames = frame_range.frames()

        print_banner("Compute initial depth")

        ft = DepthFineTuner(self.out_dir, frames, params)
        initial_depth_dir = pjoin(self.path, f"depth_{params.model_type}")
        t0 = time.time()
        if not self.video.check_frames(pjoin(initial_depth_dir, "depth"), "raw"):
            ft.save_depth(initial_depth_dir) # 预测初始深度保存raw和png
        eval_time = time.time() - t0
        print('Eval Depth time: ', eval_time)

        valid_frames = calibrate_scale(self.video, self.out_dir, frame_range, params) # colmap重建合理帧
        # frame range for finetuning:
        ft_frame_range = frame_range.intersection(OptionalSet(set(valid_frames)))
        print("Filtered out frames",
            sorted(set(frame_range.frames()) - set(ft_frame_range.frames())))

        print_banner("Compute flow")

        t0 = time.time()
        frame_pairs = sample_pairs(ft_frame_range, params.flow_ops)
        self.flow.compute_flow(frame_pairs, params.flow_checkpoint)
        flow_time = time.time() - t0

        print_banner("Compute flow masks, Time: %.f" % flow_time)

        self.flow.mask_valid_correspondences()

        flow_list_path = self.flow.check_good_flow_pairs(
            frame_pairs, params.overlap_ratio
        )
        shutil.copyfile(flow_list_path, pjoin(self.path, "flow_list.json"))

        print_banner("Visualize flow")

        self.flow.visualize_flow(warp=True)
        
        reload = False
        #reload = True
        t0 = time.time()
        if not reload:
            print_banner("Fine-tuning")
            ft.fine_tune(writer=self.writer)
            print_banner("Compute final depth")
        ft_time = time.time() - t0
        print('Finetune Time: ', ft_time)

        if not self.video.check_frames(pjoin(ft.out_dir, "depth"), "raw", frames):
            ft.save_depth(ft.out_dir, frames, reload=reload)

        if params.make_video:
            print_banner("Export visualization videos")
            self.make_videos(params, ft.out_dir)

        print('Eval Time: ', eval_time)
        print('Flow Time:', flow_time)
        print('Finetune Time: ', ft_time)

        return initial_depth_dir, ft.out_dir, frame_range.frames()

    def process(self, params):
        self.path = params.path # 输入输出路径results/*
        os.makedirs(self.path, exist_ok=True)

        self.video_file = params.video_file # 视频路径data/videos/*.mp4

        self.out_dir = self.create_output_path(params) # results/*/R_hierarchical2_mc

        self.video = Video(params.path, params.video_file)
        self.flow = Flow(params.path, self.out_dir)

        print_title(f"Processing dataset '{self.path}'")

        print(f"Output directory: {self.out_dir}")

        if params.op == "all": # 整体流程
            return self.pipeline(params)
        elif params.op == "extract_frames":
            return self.extract_frames(params)
        else:
            raise RuntimeError("Invalid operation specified.")

    def make_videos(self, params, ft_depth_dir):
        args = [
            "--color_dir", pjoin(self.path, "color_down_png"),
            "--out_dir", pjoin(self.out_dir, "videos"),
            "--depth_dirs",
            pjoin(self.path, f"depth_{params.model_type}"),
            pjoin(self.path, "depth_colmap_dense"),
            pjoin(ft_depth_dir, "depth"),
        ]
        gt_dir = pjoin(self.path, "depth_gt")
        if os.path.isdir(gt_dir):
            args.append(gt_dir)

        vid_params = mkvid.MakeVideoParams().parser.parse_args(
            args,
            namespace=params
        )
        logging.info("Make videos {}".format(vid_params))
        mkvid.main(vid_params)
