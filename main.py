#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from params import Video3dParamsParser
from process import DatasetProcessor


if __name__ == "__main__":
    parser = Video3dParamsParser()  # 参数配置
    params = parser.parse()

    dp = DatasetProcessor() # 数据加载
    dp.process(params)  # 在线训练
