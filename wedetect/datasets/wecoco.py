# Copyright (c) Tencent Inc. All rights reserved.

from .yolov5_coco import YOLOv5CocoDataset
from mmdet.registry import DATASETS
import json
import os.path as osp


@DATASETS.register_module()
class WeCocoDataset(YOLOv5CocoDataset):
    def __init__(self, debug_mode=False, **kwargs):
        if "metainfo" in kwargs:
            metainfo = kwargs["metainfo"]
            if "classes" in metainfo and isinstance(metainfo["classes"], str) and osp.isfile(metainfo["classes"]):
                with open(metainfo["classes"], "r") as f:
                    classes = json.load(f)
                metainfo["classes"] = classes
                kwargs["metainfo"] = metainfo
        self.debug_mode = debug_mode
        super().__init__(**kwargs)

    def __len__(self):
        if self.debug_mode:
            return 100
        else:
            return super().__len__()
