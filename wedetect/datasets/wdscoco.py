# Copyright (c) Tencent Inc. All rights reserved.

from .yolov5_coco import YOLOv5CocoDataset
from mmdet.registry import DATASETS
import json
import os.path as osp
import webdataset as wds
from typing import List
from mmengine.dataset.base_dataset import BaseDataset, Compose, force_full_init
import cv2
from PIL import Image
import numpy as np
from mmengine.logging import print_log
import torch.distributed as dist
from .weref import NegQueue

# BGR
def pil2cv2(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


@DATASETS.register_module()
class WDSCoco(YOLOv5CocoDataset):
    def __init__(
        self,
        data_root,
        ann_key="annotations",
        label_key="text_ch",
        length=100,
        class_text_path=None,
        use_negative_queue=False,
        en_zh_map=None,
        **kwargs,
    ):
        kwargs["serialize_data"] = False
        if "metainfo" in kwargs:
            metainfo = kwargs["metainfo"]
            if (
                "classes" in metainfo
                and isinstance(metainfo["classes"], str)
                and osp.isfile(metainfo["classes"])
            ):
                with open(metainfo["classes"], "r") as f:
                    classes = json.load(f)
                metainfo["classes"] = classes
                kwargs["metainfo"] = metainfo
        super().__init__(data_root=data_root, **kwargs)
        self.length = length
        self.max_retry = 3
        self.ann_key = ann_key
        self.label_key = label_key
        self.use_negative_queue=use_negative_queue
        # ---------新增--------------
        self.en_zh_map = en_zh_map
        if en_zh_map is not None:
            self.en_zh_map = json.load(open(en_zh_map, "r"))
        # --------------------------
        if class_text_path is not None:
            self.class_texts = json.load(open(class_text_path, "r"))
        else:
            self.class_texts = None
        if use_negative_queue:
            self.neg_queue = NegQueue(size=80)

    def load_data_list(self) -> List[dict]:
        return (
            wds.WebDataset(
                self.data_root, resampled=True, nodesplitter=wds.split_by_node
            )
            .shuffle(1000)
            .decode("pil", handler=wds.handlers.warn_and_continue)
            .to_tuple("jpg", "json")
        )

    def __len__(self):
        return self.length

    def filter_data(self):
        return self.data_list

    @force_full_init
    def get_data_info(self, idx: int, retry=0) -> dict:
        if retry > self.max_retry:
            raise ValueError(f"retry {retry} times, still failed")
        if not hasattr(self, "data_iter"):
            self.data_list = self.load_data_list()
            self.data_iter = iter(self.data_list)
        try:
            data = next(self.data_iter)
        except StopIteration:
            print_log(
                f"Loaded All Data in {self.data_root}, idx: {idx}, reinitializing"
            )
            self.data_iter = iter(self.data_list)
            data = next(self.data_iter)
        img = pil2cv2(data[0])
        js = data[1]
        results = dict()
        results["img"] = img
        results["img_path"] = js["meta"]["image_name"]
        # for debug:
        # if dist.is_initialized():
        #     rank = dist.get_rank()
        # else:
        #     rank = 0
        # with open("test.txt", "a") as f:
        #     f.write(f"Rank {rank}: {results['img_path']}\n")
        results["img_shape"] = img.shape[:2]
        results["ori_shape"] = img.shape[:2]
        instances = []
        class_texts = []
        if self.class_texts is not None:
            class_texts = self.class_texts
            text2catid = {}
            for i, v_list in enumerate(class_texts):
                for v in v_list:
                    text2catid[v] = i
        else:
            class_texts = []
            text2catid = {}
        instances = []
        annotations = js[self.ann_key]
        for i in range(len(annotations)):
            ann = annotations[i]
            if "bbox" not in ann:
                # print('bbox_error: ', js)
                return self.get_data_info(idx, retry + 1)
            if self.label_key=='vlm':
                try:
                    tags=ann['vlm']['tags']
                except:
                    tags=[]
                if not tags:
                    # print('not tags_error: ', js)
                    return self.get_data_info(idx, retry + 1)
                else:
                    text=tags[-1]
            else:
                text = ann[self.label_key]
            # --------新增英文到中文映射-----------
            if self.en_zh_map is not None:
                # text = self.en_zh_map[text]
                text = self.en_zh_map.get(text, text)
            # ----------------------------------
            if text not in text2catid:
                text2catid[text] = len(class_texts)
                class_texts.append([text])
            bbox_label = text2catid[text]
            bbox = ann["bbox"]
            x1, y1, w, h = bbox
            bbox = [x1, y1, x1 + w, y1 + h]
            instances.append(dict(ignore_flag=0, bbox=bbox, bbox_label=bbox_label))
        if len(instances) == 0:
            # print('nothing anno: ', js)
            return self.get_data_info(idx, retry + 1)
        results["instances"] = instances
        if self.use_negative_queue:
            class_texts = self.neg_queue.enrich(class_texts)
            self.neg_queue.update(class_texts)
        results["texts"] = class_texts
        return results
