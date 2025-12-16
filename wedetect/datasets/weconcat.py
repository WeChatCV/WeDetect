# Copyright (c) Tencent Inc. All rights reserved.

from mmengine.dataset import ConcatDataset
from mmdet.registry import DATASETS
from mmengine.logging import print_log
import logging
import os
import os.path as osp
from typing import List, Sequence, Tuple, Union, TypedDict
from mmengine.dataset import BaseDataset, force_full_init
from .mm_dataset import MultiModalDataset
from .weref import WeRefDataset
import numpy as np
from mmengine.dist import get_rank
import json
import tqdm
import random
from collections import defaultdict
import torch
from torch.utils.data import get_worker_info
from transformers import AutoModel, AutoTokenizer
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

def find_data_root(dataset):
    if isinstance(dataset, dict):
        if 'data_root' in dataset:
            return dataset['data_root']
        else:
            return find_data_root(dataset['dataset'])
    else:
        if hasattr(dataset,'data_root'):
            return dataset.data_root
        else:
            return find_data_root(dataset.data_root)
@DATASETS.register_module()
class WeConcatDataset(ConcatDataset):
    def __init__(
        self,
        datasets: Sequence[Union[BaseDataset, dict]],
        name="WeConcatDataset",
        language_model="/mnt/csp/nj1/home/yukunsu/code/enire3.0",
        num_cluster=100,
        lazy_init: bool = False,
        ignore_keys: Union[str, List[str], None] = None,
        parallel=False,
    ):
        self.name = name
        self.language_model = language_model
        self.num_cluster = num_cluster
        self.datasets: List[BaseDataset] = []
        if parallel:
            with ThreadPoolExecutor() as executor:
                futures = []
                for dataset in datasets:
                    if isinstance(dataset, dict):
                        data_root=find_data_root(dataset)
                        print_log(
                            f"Building dataset in {data_root}",
                            logger="current",
                            level=logging.INFO,
                        )
                        futures.append(executor.submit(DATASETS.build, dataset))
                    elif isinstance(dataset, BaseDataset):
                        futures.append(executor.submit(lambda x: x, dataset))
        else:
            for dataset in datasets:
                if isinstance(dataset, dict):
                    data_root=find_data_root(dataset)
                    print_log(
                        f"Building dataset in {data_root}",
                        logger="current",
                        level=logging.INFO,
                    )
                    self.datasets.append(DATASETS.build(dataset))
                elif isinstance(dataset, BaseDataset):
                    self.datasets.append(dataset)
        if ignore_keys is None:
            self.ignore_keys = []
        elif isinstance(ignore_keys, str):
            self.ignore_keys = [ignore_keys]
        elif isinstance(ignore_keys, list):
            self.ignore_keys = ignore_keys
        else:
            raise TypeError(
                "ignore_keys should be a list or str, " f"but got {type(ignore_keys)}"
            )

        meta_keys: set = set()
        for dataset in self.datasets:
            meta_keys |= dataset.metainfo.keys()
        # Only use metainfo of first dataset.
        self._metainfo = self.datasets[0].metainfo
        for i, dataset in enumerate(self.datasets, 1):
            for key in meta_keys:
                if key in self.ignore_keys:
                    continue
                if key not in dataset.metainfo:
                    raise ValueError(
                        f"{key} does not in the meta information of "
                        f"the {i}-th dataset"
                    )
                first_type = type(self._metainfo[key])
                cur_type = type(dataset.metainfo[key])
                if first_type is not cur_type:  # type: ignore
                    raise TypeError(
                        f"The type {cur_type} of {key} in the {i}-th dataset "
                        "should be the same with the first dataset "
                        f"{first_type}"
                    )
                if (
                    isinstance(self._metainfo[key], np.ndarray)
                    and not np.array_equal(self._metainfo[key], dataset.metainfo[key])
                    or (
                        not isinstance(self._metainfo[key], np.ndarray)
                        and self._metainfo[key] != dataset.metainfo[key]
                    )
                ):
                    raise ValueError(
                        f"The meta information of the {i}-th dataset does not "
                        "match meta information of the first dataset"
                    )

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

        rank = get_rank()
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        np.random.seed(rank * 999 + worker_id)
        random.seed(rank * 999 + worker_id)

    def full_init(self):
        super().full_init()
        self.init_texts()

    @force_full_init
    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids of class balanced dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            List[int]: All categories in the image of specified index.
        """
        data_info = self.get_data_info(idx)
        data_texts = data_info["texts"]
        labels = [instance["bbox_label"] for instance in data_info["instances"]]
        text_lists = [data_texts[label] for label in labels]
        flatten_text_list = [text for text_list in text_lists for text in text_list]
        text_ids = [
            self.text2textid.get(t, random.randint(0, len(self.texts) - 1))
            for t in flatten_text_list
        ]
        return text_ids

    def extract_embeddings(self, texts):
        text_encoder = AutoModel.from_pretrained(self.language_model).cuda()
        tokenizer = AutoTokenizer.from_pretrained(self.language_model)
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device="cuda")
        outputs = text_encoder(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings

    def init_texts(self):
        texts = []
        for i, dataset in enumerate(self.datasets):
            if isinstance(dataset, WeRefDataset):
                pass
            elif isinstance(dataset, MultiModalDataset):
                # 闭集OD数据集，用MultiModalDataset包装
                for t in dataset.class_texts:
                    texts.extend(t)            
            else:
                print(type(dataset))
                pass
        self.texts = texts
        self.text2textid = {text: i for i, text in enumerate(texts)}
        print_log(
            f"Initialized {len(texts)} texts", logger="current", level=logging.INFO
        )
        return texts
