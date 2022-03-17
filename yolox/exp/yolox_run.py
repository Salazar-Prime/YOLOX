#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os

import torch
import torch.distributed as dist
import torch.nn as nn

from yolox.exp import Exp as MyExp
from yolox.data.datasets.fourWeedsDL import fourWeeds

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 4
        
        # ## for yolox-l
        # self.depth = 1.0
        # self.width = 1.0
        ## yolox-s
        self.depth = 0.33
        self.width = 0.50

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_num_workers = 8
        self.input_size = (640,640)
        self.multiscale_range = 0
        self.data_dir = "/home/varun/work/all_yolo/src/yolox/datasets/data"
        self.train_ann = "/home/varun/work/all_yolo/src/yolox/datasets/train.json"
        self.val_ann = "/home/varun/work/all_yolo/src/yolox/datasets/val.json"
        self.test_ann = "/home/varun/work/all_yolo/src/yolox/datasets/val.json"
        # self.shear = 0.0
        self.eval_interval = 10

        # --------------- transform config ----------------- #
        # prob of applying mosaic aug
        self.mosaic_prob = 1.0
        # prob of applying mixup aug
        self.mixup_prob = 1.0
        # prob of applying hsv aug
        self.hsv_prob = 1.0
        # prob of applying flip aug
        self.flip_prob = 0.5
        # rotation angle range, for example, if set to 2, the true range is (-2, 2)
        self.degrees = 10.0
        # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        # apply mixup aug or not
        self.enable_mixup = True
        self.mixup_scale = (0.5, 1.5)
        # shear angle range, for example, if set to 2, the true range is (-2, 2)
        self.shear = 2.0

        # -----------------  testing config ------------------ #
        # output image size during evaluation/test
        self.test_size = (640, 640)
        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.01
        # nms threshold
        self.nmsthre = 0.05
        # self.output_dir = "."

        # yolox-nano
        self.depth = 0.33
        self.width = 0.375
        self.input_size = (416, 416)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (416, 416)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False


    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            COCODataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            # dataset = COCODataset(
            #     data_dir=self.data_dir,
            #     json_file=self.train_ann,
            #     img_size=self.input_size,
            #     preproc=TrainTransform(
            #         max_labels=50,
            #         flip_prob=self.flip_prob,
            #         hsv_prob=self.hsv_prob),
            #     cache=cache_img,
            # )
            dataset = fourWeeds(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                name="", # keep empty, used to get file path -> path.join(data_dir+name+filename)
                img_size=self.input_size,
                preproc=TrainTransform(
                                max_labels=50,
                                flip_prob=self.flip_prob,
                                hsv_prob=self.hsv_prob),
                cache=cache_img,
                )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader
    
    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import COCODataset, ValTransform

        valdataset = fourWeeds(
                data_dir=self.data_dir,
                json_file=self.val_ann,
                name="", # keep empty, used to get file path -> path.join(data_dir+name+filename)
                img_size=self.input_size,
                preproc=ValTransform(legacy=legacy),
                )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader
 
