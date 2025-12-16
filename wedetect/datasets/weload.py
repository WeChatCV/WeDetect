# Copyright (c) Tencent Inc. All rights reserved.

from mmcv.transforms import LoadImageFromFile
from mmdet.registry import TRANSFORMS
@TRANSFORMS.register_module()
class WeLoadImg(LoadImageFromFile):
    def transform(self, results: dict) -> dict:
        if 'img' in results:
            return results
        return super().transform(results)