#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from .depth_model import DepthModel
from .mannequin_challenge_model import MannequinChallengeModel
from .midas_v2_model import MidasV2Model
from .monodepth2_model import Monodepth2Model
from .our128_model import Our128Model
from .our256_model import Our256Model
from .our320_model import Our320Model

from typing import List


def get_depth_model_list() -> List[str]:
    return ["mc", "midas2", "monodepth2", "our128", "our256", 'our320']


def get_depth_model(type: str) -> DepthModel:
    if type == "mc":
        return MannequinChallengeModel
    elif type == "midas2":
        return MidasV2Model
    elif type == "monodepth2":
        return Monodepth2Model
    elif type == "our128":
        return Our128Model()
    elif type == "our256":
        return Our256Model()
    elif type == "our320":
        return Our320Model()
    else:
        raise ValueError(f"Unsupported model type '{type}'.")


def create_depth_model(type: str) -> DepthModel:
    model_class = get_depth_model(type)
    return model_class()
