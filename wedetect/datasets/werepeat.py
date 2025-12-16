# Copyright (c) Tencent Inc. All rights reserved.

from mmengine.dataset import RepeatDataset
from mmdet.registry import DATASETS

@DATASETS.register_module()
class WeRepeat(RepeatDataset):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        if hasattr(self.dataset, "data_root"):
            self.data_root = self.dataset.data_root
        elif hasattr(self.dataset.dataset, "data_root"):
            self.data_root = self.dataset.dataset.data_root
        if hasattr(self.dataset, "class_texts"):
            self.class_texts = self.dataset.class_texts
        elif hasattr(self.dataset.dataset, "class_texts"):
            self.class_texts = self.dataset.dataset.class_texts
