#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os, sys
from loguru import logger

import cv2
import numpy as np
from pycocotools.coco import COCO

# USER imports
sys.path.append("/home/varun/work/all_yolo/src/yolox/yolox/data")
from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset
from .coco import COCODataset


# Create Dataloader
class fourWeeds(COCODataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

# obj = fourWeeds(
#     data_dir="/home/varun/work/all_yolo/src/yolox/datasets/data",
#     json_file="/home/varun/work/all_yolo/src/yolox/datasets/train.json",
#     name="", # keep empty, used to get file path -> path.join(data_dir+name+filename)
#     img_size=(416,416),
#     preproc=None,
#     cache=False,
#     )

# sum = 0
# for i, data in enumerate(obj):
#     sum += 1
# print(sum)