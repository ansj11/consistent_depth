#!/usr/bin/env python
# coding=utf-8

import os
import cv2
import numpy as np
from tools.colmap_processor import COLMAPParams, COLMAPProcessor
from scale_calibration import ScaleCalibrationParams

import argparse
from pdb import set_trace



class ParamsParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--path", type=str,
            help="Path to all the input (except for the video) and output files are stored")
        #self.parser.add_argument("image_path", help="image path")
        #self.parser.add_argument("workspace_path", help="workspace path")
        self.parser.add_argument(
            "--mask_path",
            help="path for mask to exclude feature extration from those regions",
            default=None, # 排除区域不提取特征
        )
        self.parser.add_argument(
            "--dense_max_size", type=int, help='Max size for dense COLMAP', default=1280,
        ) # 最长边384
        self.parser.add_argument(
            "--colmap_bin_path",
            help="path to colmap bin. COLMAP 3.6 is required to enable mask_path",
            default='colmap'
        )
        self.parser.add_argument(
            "--sparse", help="disable dense reconstruction", action='store_true'
        )
        self.parser.add_argument(
            "--initialize_pose", help="Intialize Pose", action='store_true'
        )
        self.parser.add_argument(
            "--camera_params", help="prior camera parameters", default=None
        )
        self.parser.add_argument(
            "--camera_model", help="camera_model", default='SIMPLE_PINHOLE'
        )
        self.parser.add_argument(
            "--refine_intrinsics",
            help="refine camera parameters. Not used when camera_params is None",
            action="store_true"
        )
        self.parser.add_argument(
            "--matcher", choices=["exhaustive", "sequential"], default="exhaustive",
            help="COLMAP matcher ('exhaustive' or 'sequential')"
        )
        self.parser.add_argument(
            "--dense_frame_ratio", type=float, default=0.95,
            help="threshold on percentage of successully computed dense depth frames."
        )
        self.parser.add_argument("--dense_pixel_ratio", type=float, default=0.3,
            help="ratio of valid dense depth pixels for that frame to valid")


    def parse(self, args=None, namespace=None):
        return self.parser.parse_args(args, namespace=namespace)


def run_colmap():
    print('COLMAP reconstruction.')
    parser = ParamsParser()  # 参数配置
    args = parser.parse()
    path = args.path

    colmap = COLMAPProcessor()
    
    color_dir = os.path.join(path, 'color_full')
    colmap_dir = os.path.join(path, 'colmap_dense2')
    path_args = [color_dir, colmap_dir]
    mask_path = os.path.join(path, 'colmap_mask')
    if os.path.isdir(mask_path):
        path_args.extend(['--mask_path', mask_path])
    colmap_args = COLMAPParams().parse_args(
        args=path_args + ['--dense_max_size', str(1280)], # 生成大图输出1280x720
        namespace=args
    )

    colmap.process(colmap_args) # colmap计算内外参
    print('COLMAP finish!')


if __name__ == '__main__':
    run_colmap()
