# Copyright (c) Tencent Inc. All rights reserved.
import os.path as osp
from typing import List, Union
from mmengine.logging import print_log
import logging
from typing import Callable, List, Union
from mmengine.fileio import join_path
from mmengine.utils import is_abs
from mmdet.datasets import BaseDetDataset
from mmengine.dataset.base_dataset import BaseDataset, Compose, force_full_init
from mmdet.registry import DATASETS
import tqdm
import copy
import json
import os
from .mm_dataset import MultiModalDataset
import random
from mmengine.logging import print_log
import logging


class NegQueue:
    def __init__(self, size=80):
        self.size = size
        self.queue = set()

    def update(self, data: List[str]):
        if not isinstance(data[0],str):
            data=[xx for x in data for xx in x]
            for val in data:
                assert isinstance(val,str),str(data)
        self.queue.update(set(data))
        if len(self.queue) > self.size:
            self.queue = set(random.sample(list(self.queue), self.size))
        if "object" in self.queue:
            self.queue.remove("object")

    def enrich(self, class_texts):        
        if isinstance(class_texts[0], str):
            append=list(self.queue-set(class_texts))            
        else:
            flatten=[xx for x in class_texts for xx in x]
            append=[[s] for s in self.queue-set(flatten)]
        return class_texts+append


@DATASETS.register_module()
class WeRefDataset(MultiModalDataset):
    @property
    def metainfo(self) -> dict:
        return copy.deepcopy({"classes": ("object",), "palette": [(220, 20, 60)]})

    def __init__(
        self,
        ref_root: str = None,
        mixed_ratio=0.5,
        use_negative_queue=True,
        use_sam_box=True,
        **kwargs,
    ):
        self.ref_root = ref_root
        self.mixed_ratio = mixed_ratio
        self.use_negative_queue = use_negative_queue
        self.use_sam_box = use_sam_box
        self.success_ids = set()
        self.error_ids = set()
        if self.use_negative_queue:
            self.neg_queue = NegQueue(80)
        super().__init__(**kwargs)

    def full_init(self) -> None:
        """``full_init`` dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        self._ori_len = len(self.dataset)
        self._fully_initialized = True
        self.ref_infos = dict()
        for file in tqdm.tqdm(list(os.listdir(self.ref_root)), "loading ref infos"):
            if file.endswith(".jsonl"):
                with open(os.path.join(self.ref_root, file), "r") as f:
                    for line in f.readlines():
                        data = json.loads(line.strip())
                        anns = data["annotations"]
                        for ann in anns:
                            if ann["vlm"] == "ERROR":
                                print(f"ERROR vlm text for key: {data['key']}")
                                ann["vlm"] = {"tags": ["object"], "neg_tags": []}
                        self.ref_infos[data["key"]] = anns

    def report_error(self, error_msg, idx):
        print_log(
            error_msg,
            logger="current",
            level=logging.INFO,
        )
        self.error_ids.add(idx)
        print_log(
            f"{self.ref_root}: success/error: {len(self.success_ids)}/{len(self.error_ids)}",
            logger="current",
            level=logging.INFO,
        )
        if self.success_ids:
            idx = random.choice(tuple(self.success_ids))
            return self.get_data_info(idx)
        else:
            return self.get_data_info(0)

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        if random.randint(0, 100) > self.mixed_ratio * 100:
            return super().get_data_info(idx)
        """Get annotation by index."""
        data_info = self.dataset.get_data_info(idx)
        image_name = os.path.basename(data_info["img_path"])
        key = image_name.split(".")[0]
        if key not in self.ref_infos:
            return self.report_error(f"{key} not found in ref infos.", idx)
        ref_info = self.ref_infos[key]
        class_texts = []
        text2catid = {}
        instances = []
        for i in range(len(ref_info)):
            info = ref_info[i]
            if "bbox" not in info:
                return self.report_error(f"no bbox found for {key}:{i}", idx)
            try:
                tags = info["vlm"]["tags"]
            except:
                tags = []
            if not tags:
                return self.report_error(f"no tags found for {key}:{i}", idx)
            else:
                text = tags[-1]
                assert isinstance(text,str),str(text)
            if text not in text2catid:
                text2catid[text] = len(class_texts)
                class_texts.append(text)
            bbox_label = text2catid[text]
            if self.use_sam_box:
                bbox = info["sam2_bbox"]
            else:
                bbox = info["bbox"]
            x1, y1, w, h = bbox
            bbox = [x1, y1, x1 + w, y1 + h]
            instances.append(dict(ignore_flag=0, bbox=bbox, bbox_label=bbox_label))        
        if len(instances) == 0:
            return self.report_error(f"no instances found for {key}", idx)
        if self.use_negative_queue:
            class_texts = self.neg_queue.enrich(class_texts)
            self.neg_queue.update(class_texts)
        data_info["instances"] = instances
        data_info["texts"] = [[c] for c in class_texts]
        self.success_ids.add(idx)
        return data_info
