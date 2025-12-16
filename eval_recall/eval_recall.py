import json
import numpy as np
from pycocotools.coco import COCO
from lvis import LVIS
from recall import eval_recalls




ds_collections = {
    'coco': {
        'ann_path': 'data/coco/annotations/instances_val2017.json',
        'image_path': 'data/coco/val2017/',
    },
    'lvis': {
        'ann_path': 'data/lvis/lvis_v1_val.json',
        'image_path': 'data/coco/',
    },
    'paco': {
        'ann_path': 'data/PACO/paco_lvis_v1_test.json',
        'image_path': 'data/coco/',
    },
}



def fast_eval_recall(dataset, proposals):
    """Evaluate proposal recall with COCO's fast_eval_recall.

    Args:
        results (List[dict]): Results of the dataset.
        proposal_nums (Sequence[int]): Proposal numbers used for
            evaluation.
        iou_thrs (Sequence[float]): IoU thresholds used for evaluation.
        logger (MMLogger, optional): Logger used for logging the recall
            summary.
    Returns:
        np.ndarray: Averaged recall results.
    """
    
    iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    
    if dataset == 'coco':
        coco = COCO(ds_collections['coco']['ann_path'])
        gt_bboxes = []
        pred_bboxes = []
        for proposal in proposals:
            single_proposals = proposal["boxes"]
            # print(single_proposals.shape)
            pred_bboxes.append(np.array(single_proposals, dtype=np.float32))
            ann_ids = coco.getAnnIds(imgIds=proposal["image_id"])
            ann_info = coco.loadAnns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(gt_bboxes, pred_bboxes, [100, 300], iou_thrs)
        print(sum(recalls[0]) / len(recalls[0]))
        print(sum(recalls[1]) / len(recalls[1]))

    elif dataset == 'lvis':
        lvis = LVIS(ds_collections['lvis']['ann_path'])

        gt_bboxes = []
        pred_bboxes = []

        for proposal in proposals:
            img_id = proposal["image_id"]
            single_proposals = proposal["boxes"]

            # 预测框：转为 [x1, y1, x2, y2] 格式（如果是 xywh 要转换）
            pred_boxes = np.array(single_proposals, dtype=np.float32)
            pred_bboxes.append(pred_boxes)

            # 获取该图像的所有 GT annotations（只包括被标注的物体）
            ann_ids = lvis.get_ann_ids(img_ids=[img_id])
            anns = lvis.load_anns(ann_ids)

            if len(anns) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue

            bboxes = []
            for ann in anns:
                # 忽略 crowd 和 ignore 标注
                if ann.get('ignore', False):
                    continue
                x, y, w, h = ann['bbox']
                bboxes.append([x, y, x + w, y + h])  # 转为 x1y1x2y2

            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        # 评估 recall
        recalls = eval_recalls(gt_bboxes, pred_bboxes, [100, 300], iou_thrs)
        print(sum(recalls[0]) / len(recalls[0]))
        print(sum(recalls[1]) / len(recalls[1]))

    if dataset == 'paco':
        lvis = LVIS(ds_collections['paco']['ann_path'])

        gt_bboxes = []
        pred_bboxes = []

        for proposal in proposals:
            img_id = proposal["image_id"]
            single_proposals = proposal["boxes"]

            # 预测框：转为 [x1, y1, x2, y2] 格式（如果是 xywh 要转换）
            pred_boxes = np.array(single_proposals, dtype=np.float32)
            pred_bboxes.append(pred_boxes)

            # 获取该图像的所有 GT annotations（只包括被标注的物体）
            ann_ids = lvis.get_ann_ids(img_ids=[img_id])
            anns = lvis.load_anns(ann_ids)

            if len(anns) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue

            bboxes = []
            for ann in anns:
                # 忽略 crowd 和 ignore 标注
                if ann.get('ignore', False):
                    continue
                x, y, w, h = ann['bbox']
                bboxes.append([x, y, x + w, y + h])  # 转为 x1y1x2y2

            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        # 评估 recall
        recalls = eval_recalls(gt_bboxes, pred_bboxes, [100, 300], iou_thrs)
        print(sum(recalls[0]) / len(recalls[0]))
        print(sum(recalls[1]) / len(recalls[1]))


import os
import time
import json
import random
import argparse
import itertools
import subprocess
import torch

from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO



import argparse
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import math
from torch import Tensor
import numpy as np
from typing import Tuple, Union, List, Optional
from torch.nn.modules.utils import _pair
import torchvision
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import json
import random
additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]


from collections import OrderedDict
from typing import Sequence
import itertools
from typing import List, Sequence, Tuple

try:
    from transformers import AutoTokenizer, AutoConfig, XLMRobertaModel
except ImportError:
    AutoTokenizer = None
    HFBertModel = None


def letterbox(
    img: Image.Image,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=False,
    scale_fill=False,
    scale_up=True
):
    """
    将 PIL 图像进行 letterbox 处理，用于 YOLOv5 推理前的预处理
    来源：YOLOv5 utils/augmentations.py

    参数:
        img (PIL.Image): 输入图像
        new_shape (tuple): 模型输入尺寸 (height, width)
        color (tuple): 填充颜色 (R, G, B)
        auto (bool): 是否自动选择最小步长（如 YOLOv5 的 auto-anchor）
        scale_fill (bool): 是否拉伸填充（不保持比例）
        scale_up (bool): 是否放大图像

    返回:
        image (PIL.Image): letterbox 后的图像
        ratio (float): 缩放比例
        (dw, dh) (float, float): 左/右 和 上/下 填充的一半（用于还原 bbox）
    """
    shape = img.size  # (w, h)
    new_shape = (new_shape[1], new_shape[0])  # YOLOv5 使用 (w, h)

    # 计算缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:  # 只缩小，不放大
        r = min(r, 1.0)

    # 计算缩放后的新尺寸
    new_unpad = (int(round(shape[0] * r)), int(round(shape[1] * r)))

    # 缩放图像
    img_resized = img.resize(new_unpad, Image.Resampling.BILINEAR)

    # 填充颜色
    top, left = 0, 0
    # if auto:  # YOLOv5 自动模式：保证尺寸是 32 的倍数
    #     dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]
    #     dw %= 32
    #     dh %= 32
    # elif scale_fill:
    #     # 拉伸填充（不推荐，破坏宽高比）
    #     img_resized = img.resize(new_shape, Image.Resampling.BILINEAR)
    #     dw, dh = 0, 0
    #     new_unpad = new_shape
    # else:
    # 正常 letterbox：居中填充
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]
    left = dw // 2
    top = dh // 2

    # 创建新图像并填充背景色
    img_letterboxed = Image.new("RGB", new_shape, color)
    img_letterboxed.paste(img_resized, (left, top))

    # 计算填充量（用于还原检测框）
    ratio = r
    dw /= 2  # 左侧填充
    dh /= 2  # 上侧填充

    return img_letterboxed, ratio, (dw, dh)


def filter_scores_and_topk(scores, score_thr, topk, results=None):
    """Filter results using score threshold and topk candidates.

    Args:
        scores (Tensor): The scores, shape (num_bboxes, K).
        score_thr (float): The score filter threshold.
        topk (int): The number of topk candidates.
        results (dict or list or Tensor, Optional): The results to
           which the filtering rule is to be applied. The shape
           of each item is (num_bboxes, N).

    Returns:
        tuple: Filtered results

            - scores (Tensor): The scores after being filtered, \
                shape (num_bboxes_filtered, ).
            - labels (Tensor): The class labels, shape \
                (num_bboxes_filtered, ).
            - anchor_idxs (Tensor): The anchor indexes, shape \
                (num_bboxes_filtered, ).
            - filtered_results (dict or list or Tensor, Optional): \
                The filtered results. The shape of each item is \
                (num_bboxes_filtered, N).
    """
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)

    num_topk = min(topk, valid_idxs.size(0))
    # torch.sort is actually faster than .topk (at least on GPUs)
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs.unbind(dim=1)

    filtered_results = None
    if results is not None:
        if isinstance(results, dict):
            filtered_results = {k: v[keep_idxs] for k, v in results.items()}
        elif isinstance(results, list):
            filtered_results = [result[keep_idxs] for result in results]
        elif isinstance(results, torch.Tensor):
            filtered_results = results[keep_idxs]
        else:
            raise NotImplementedError(f'Only supports dict or list or Tensor, '
                                      f'but get {type(results)}.')
    return scores, labels, keep_idxs, filtered_results


#################################################
# ConvNext Backbone
#################################################

class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXt(nn.Module):
    r"""ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        model_name
    ):
        super().__init__()

        if model_name == "base":
            depths = [3, 3, 27, 3]
            dims = [128, 256, 512, 1024]
        if model_name == "large":
            depths = [3, 3, 27, 3]
            dims = [192, 384, 768, 1536]
        if model_name == "small":
            depths = [3, 3, 27, 3]
            dims = [96, 192, 384, 768]
            
        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, 0.0, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=1e-6,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]



    def forward(self, x):

        outputs = []
        c1 = self.downsample_layers[0](x)
        c1 = self.stages[0](c1)
        outputs.append(c1)

        c2 = self.downsample_layers[1](c1)
        c2 = self.stages[1](c2)
        outputs.append(c2)

        c3 = self.downsample_layers[2](c2)
        c3 = self.stages[2](c3)
        outputs.append(c3)

        c4 = self.downsample_layers[3](c3)
        c4 = self.stages[3](c4)
        outputs.append(c4)

        return (c1, c2, c3, c4)






#################################################
# neck
#################################################


activation_table = {'relu':nn.ReLU(),
                    'silu':nn.SiLU(),
                    'hardswish':nn.Hardswish()
                    }


class ConvModule_torch(nn.Module):
    '''A combination of Conv + BN + Activation'''
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation_type, padding=None, groups=1, bias=False):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if activation_type is not None:
            self.act = activation_table.get(activation_type)
        self.activation_type = activation_type

    def forward(self, x):
        if self.activation_type is None:
            return self.bn(self.conv(x))
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        if self.activation_type is None:
            return self.conv(x)
        return self.act(self.conv(x))


class ConvBNReLU(nn.Module):
    '''Conv and BN with ReLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        super().__init__()
        self.block = ConvModule_torch(in_channels, out_channels, kernel_size, stride, 'relu', padding, groups, bias)

    def forward(self, x):
        return self.block(x)
    

class ConvBNSiLU(nn.Module):
    '''Conv and BN with SiLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        super().__init__()
        self.block = ConvModule_torch(in_channels, out_channels, kernel_size, stride, 'silu', padding, groups, bias)

    def forward(self, x):
        return self.block(x)



class RepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''
    def __init__(self, in_channels, out_channels, block, basic_block, n=1):
        super().__init__()

        self.conv1 = BottleRep(in_channels, out_channels, basic_block=basic_block, weight=True)
        n = n // 2
        self.block = nn.Sequential(*(BottleRep(out_channels, out_channels, basic_block=basic_block, weight=True) for _ in range(n - 1))) if n > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class BottleRep(nn.Module):

    def __init__(self, in_channels, out_channels, basic_block, weight=False):
        super().__init__()
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha = Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs


class BepC3(nn.Module):
    '''CSPStackRep Block'''
    def __init__(self, in_channels, out_channels, n=1, e=0.5):
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = ConvBNReLU(in_channels, c_, 1, 1)
        self.cv2 = ConvBNReLU(in_channels, c_, 1, 1)
        self.cv3 = ConvBNReLU(2 * c_, out_channels, 1, 1)
        self.cv1 = ConvBNSiLU(in_channels, c_, 1, 1)
        self.cv2 = ConvBNSiLU(in_channels, c_, 1, 1)
        self.cv3 = ConvBNSiLU(2 * c_, out_channels, 1, 1)

        self.m = RepBlock(in_channels=c_, out_channels=c_, n=n, block=BottleRep, basic_block=ConvBNSiLU)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class Transpose(nn.Module):
    '''Normal Transpose, default for upsampling'''
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.upsample_transpose = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True
        )

    def forward(self, x):
        return self.upsample_transpose(x)
    

class BiFusion(nn.Module):
    '''BiFusion Block in PAN'''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cv1 = ConvBNReLU(in_channels[0], out_channels, 1, 1)
        self.cv2 = ConvBNReLU(in_channels[1], out_channels, 1, 1)
        self.cv3 = ConvBNReLU(out_channels * 3, out_channels, 1, 1)

        self.upsample = Transpose(
            in_channels=out_channels,
            out_channels=out_channels,
        )
        self.downsample = ConvBNReLU(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2
        )

    def forward(self, x):
        x0 = self.upsample(x[0])
        x1 = self.cv1(x[1])
        x2 = self.downsample(self.cv2(x[2]))
        return self.cv3(torch.cat((x0, x1, x2), dim=1))




class CSPRepBiFPANNeck(nn.Module):
    """
    CSPRepBiFPANNeck module.
    """

    def __init__(self, scale_factor):
        super().__init__()

        channels_list=[64, 128, 256, 512, 1024, 256, 128, 128, 256, 256, 512]
        num_repeats=[1, 6, 12, 18, 6, 12, 12, 12, 12]
        csp_e=float(1)/2

        assert channels_list is not None
        assert num_repeats is not None

        stage_block = BepC3
        

        self.reduce_layer0 = ConvBNReLU(
            in_channels=int(channels_list[4] * scale_factor), # 1024
            out_channels=int(channels_list[5] * scale_factor), # 256
            kernel_size=1,
            stride=1
        )

        self.Bifusion0 = BiFusion(
            in_channels=[int(channels_list[3] * scale_factor), int(channels_list[2] * scale_factor)], # 512, 256
            out_channels=int(channels_list[5] * scale_factor), # 256
        )

        self.Rep_p4 = stage_block(
            in_channels=int(channels_list[5] * scale_factor), # 256
            out_channels=int(channels_list[5] * scale_factor), # 256
            n=num_repeats[5],
            e=csp_e,
        )

        self.reduce_layer1 = ConvBNReLU(
            in_channels=int(channels_list[5] * scale_factor), # 256
            out_channels=int(channels_list[6] * scale_factor), # 128
            kernel_size=1,
            stride=1
        )

        self.Bifusion1 = BiFusion(
            in_channels=[int(channels_list[2] * scale_factor), int(channels_list[1] * scale_factor)], # 256, 128
            out_channels=int(channels_list[6] * scale_factor), # 128
        )

        self.Rep_p3 = stage_block(
            in_channels=int(channels_list[6] * scale_factor), # 128
            out_channels=int(channels_list[6] * scale_factor), # 128
            n=num_repeats[6],
            e=csp_e,
        )

        self.downsample2 = ConvBNReLU(
            in_channels=int(channels_list[6] * scale_factor), # 128
            out_channels=int(channels_list[7] * scale_factor), # 128
            kernel_size=3,
            stride=2
        )

        self.Rep_n3 = stage_block(
            in_channels=int(channels_list[6] * scale_factor) + int(channels_list[7] * scale_factor), # 128 + 128
            out_channels=int(channels_list[8] * scale_factor), # 256
            n=num_repeats[7],
            e=csp_e,
        )

        self.downsample1 = ConvBNReLU(
            in_channels=int(channels_list[8] * scale_factor), # 256
            out_channels=int(channels_list[9] * scale_factor), # 256
            kernel_size=3,
            stride=2
        )


        self.Rep_n4 = stage_block(
            in_channels=int(channels_list[5] * scale_factor) + int(channels_list[9] * scale_factor), # 256 + 256
            out_channels=int(channels_list[10] * scale_factor), # 512
            n=num_repeats[8],
            e=csp_e,
        )


    def forward(self, input):

        (x3, x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]


        return outputs









#################################################
# head
#################################################

class BNContrastiveHead(nn.Module):
    """ Batch Norm Contrastive Head for YOLO-World
    using batch norm instead of l2-normalization
    Args:
        embed_dims (int): embed dim of text and image features
        norm_cfg (dict): normalization params
    """

    def __init__(self,
                 embed_dims: int,
                 use_einsum: bool = True) -> None:

        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims, momentum=0.03, eps=0.001)
        self.bias = nn.Parameter(torch.zeros([]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))
        self.use_einsum = use_einsum

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        if self.use_einsum:
            x = torch.einsum('bchw,bkc->bkhw', x, w)
        else:
            batch, channel, height, width = x.shape
            _, k, _ = w.shape   
            x = x.permute(0, 2, 3, 1)  # bchw->bhwc
            x = x.reshape(batch, -1, channel)  # bhwc->b(hw)c
            w = w.permute(0, 2, 1)  # bkc->bck
            x = torch.matmul(x, w)
            x = x.reshape(batch, height, width, k)
            x = x.permute(0, 3, 1, 2)

        x = x * self.logit_scale.exp() + self.bias
        return x
    

class YOLOWorldHeadModule(nn.Module):
    """Head Module for YOLO-World

    Args:
        embed_dims (int): embed dim for text feautures and image features
        use_bn_head (bool): use batch normalization head
    """

    def __init__(self,
                 embed_dims: int,
                 in_channels,
                 use_bn_head: bool = False,
                 use_einsum: bool = True,
                 freeze_all: bool = False, ) -> None:
        self.embed_dims = embed_dims
        self.use_bn_head = use_bn_head
        self.use_einsum = use_einsum
        self.freeze_all = freeze_all
        self.in_channels = in_channels
        self.reg_max = 16        
        super().__init__()
        self._init_layers()


    def _init_layers(self) -> None:
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.cls_contrasts = nn.ModuleList()
        cls_out_channels = 256
        
        self.featmap_strides = [8, 16, 32]
        self.num_levels = len(self.in_channels)

        reg_out_channels = max(
            (16, self.in_channels[0] // 4, self.reg_max * 4))

        for i in range(self.num_levels):
            self.reg_preds.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channels[i],
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False,),
                    nn.BatchNorm2d(reg_out_channels, momentum=0.03, eps=0.001),
                    nn.SiLU(),
                    nn.Conv2d(in_channels=reg_out_channels,
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False,),
                    nn.BatchNorm2d(reg_out_channels, momentum=0.03, eps=0.001),
                    nn.SiLU(),
                    nn.Conv2d(in_channels=reg_out_channels,
                              out_channels=4 * self.reg_max,
                              kernel_size=1)))
            self.cls_preds.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channels[i],
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False,),
                    nn.BatchNorm2d(cls_out_channels, momentum=0.03, eps=0.001),
                    nn.SiLU(),
                    nn.Conv2d(in_channels=cls_out_channels,
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False,),
                    nn.BatchNorm2d(cls_out_channels, momentum=0.03, eps=0.001),
                    nn.SiLU(),
                    nn.Conv2d(in_channels=cls_out_channels,
                              out_channels=self.embed_dims,
                              kernel_size=1)))

            self.cls_contrasts.append(BNContrastiveHead(self.embed_dims, use_einsum=self.use_einsum))



        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)


    def forward(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        assert len(img_feats) == self.num_levels
        txt_feats = [txt_feats for _ in range(self.num_levels)]
        outputs = []
        for i in range(self.num_levels):
            outputs.append(
                self.forward_single(img_feats[i], txt_feats[i],
                                    self.cls_preds[i], self.reg_preds[i],
                                    self.cls_contrasts[i]))
        return tuple(outputs)

    def forward_single(self, img_feat: Tensor, txt_feat: Tensor,
                       cls_pred: nn.ModuleList, reg_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed, txt_feat)
        bbox_dist_preds = reg_pred(img_feat)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds









#################################################
# prior generator
#################################################



class MlvlPointGenerator:
    """Standard points generator for multi-level (Mlvl) feature maps in 2D
    points-based detectors.

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels in order (w, h).
        offset (float): The offset of points, the value is normalized with
            corresponding stride. Defaults to 0.5.
    """

    def __init__(self,
                 strides: Union[List[int], List[Tuple[int, int]]],
                 offset: float = 0.5) -> None:
        self.strides = [_pair(stride) for stride in strides]
        self.offset = offset

    @property
    def num_levels(self) -> int:
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    @property
    def num_base_priors(self) -> List[int]:
        """list[int]: The number of priors (points) at a point
        on the feature grid"""
        return [1 for _ in range(len(self.strides))]

    def _meshgrid(self,
                  x: Tensor,
                  y: Tensor,
                  row_major: bool = True) -> Tuple[Tensor, Tensor]:
        yy, xx = torch.meshgrid(y, x)
        if row_major:
            # warning .flatten() would cause error in ONNX exporting
            # have to use reshape here
            return xx.reshape(-1), yy.reshape(-1)

        else:
            return yy.reshape(-1), xx.reshape(-1)

    def grid_priors(self,
                    featmap_sizes: List[Tuple],
                    dtype: torch.dtype = torch.float32,
                    device = 'cuda',
                    with_stride: bool = False) -> List[Tensor]:
        """Generate grid points of multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            dtype (:obj:`dtype`): Dtype of priors. Defaults to torch.float32.
            device (str | torch.device): The device where the anchors will be
                put on.
            with_stride (bool): Whether to concatenate the stride to
                the last dimension of points.

        Return:
            list[torch.Tensor]: Points of  multiple feature levels.
            The sizes of each tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """

        assert self.num_levels == len(featmap_sizes)
        multi_level_priors = []
        for i in range(self.num_levels):
            priors = self.single_level_grid_priors(
                featmap_sizes[i],
                level_idx=i,
                dtype=dtype,
                device=device,
                with_stride=with_stride)
            multi_level_priors.append(priors)
        return multi_level_priors

    def single_level_grid_priors(self,
                                 featmap_size: Tuple[int],
                                 level_idx: int,
                                 dtype: torch.dtype = torch.float32,
                                 device = 'cuda',
                                 with_stride: bool = False) -> Tensor:
        """Generate grid Points of a single level.

        Note:
            This function is usually called by method ``self.grid_priors``.

        Args:
            featmap_size (tuple[int]): Size of the feature maps, arrange as
                (h, w).
            level_idx (int): The index of corresponding feature map level.
            dtype (:obj:`dtype`): Dtype of priors. Defaults to torch.float32.
            device (str | torch.device): The device the tensor will be put on.
                Defaults to 'cuda'.
            with_stride (bool): Concatenate the stride to the last dimension
                of points.

        Return:
            Tensor: Points of single feature levels.
            The shape of tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (torch.arange(0, feat_w, device=device) +
                   self.offset) * stride_w
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        shift_x = shift_x.to(dtype)

        shift_y = (torch.arange(0, feat_h, device=device) +
                   self.offset) * stride_h
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        shift_y = shift_y.to(dtype)
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        if not with_stride:
            shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        else:
            # use `shape[0]` instead of `len(shift_xx)` for ONNX export
            stride_w = shift_xx.new_full((shift_xx.shape[0], ),
                                         stride_w).to(dtype)
            stride_h = shift_xx.new_full((shift_yy.shape[0], ),
                                         stride_h).to(dtype)
            shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h],
                                 dim=-1)
        all_points = shifts.to(device)
        return all_points

    def valid_flags(self,
                    featmap_sizes: List[Tuple[int, int]],
                    pad_shape: Tuple[int],
                    device = 'cuda') -> List[Tensor]:
        """Generate valid flags of points of multiple feature levels.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            pad_shape (tuple(int)): The padded shape of the image,
                arrange as (h, w).
            device (str | torch.device): The device where the anchors will be
                put on.

        Return:
            list(torch.Tensor): Valid flags of points of multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            point_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / point_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / point_stride[0])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w),
                                                  device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self,
                                 featmap_size: Tuple[int, int],
                                 valid_size: Tuple[int, int],
                                 device = 'cuda') -> Tensor:
        """Generate the valid flags of points of a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps, arrange as
                as (h, w).
            valid_size (tuple[int]): The valid size of the feature maps.
                The size arrange as as (h, w).
            device (str | torch.device): The device where the flags will be
            put on. Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each points in a single level \
                feature map.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid

    def sparse_priors(self,
                      prior_idxs: Tensor,
                      featmap_size: Tuple[int],
                      level_idx: int,
                      dtype: torch.dtype = torch.float32,
                      device = 'cuda') -> Tensor:
        """Generate sparse points according to the ``prior_idxs``.

        Args:
            prior_idxs (Tensor): The index of corresponding anchors
                in the feature map.
            featmap_size (tuple[int]): feature map size arrange as (w, h).
            level_idx (int): The level index of corresponding feature
                map.
            dtype (obj:`torch.dtype`): Date type of points. Defaults to
                ``torch.float32``.
            device (str | torch.device): The device where the points is
                located.
        Returns:
            Tensor: Anchor with shape (N, 2), N should be equal to
            the length of ``prior_idxs``. And last dimension
            2 represent (coord_x, coord_y).
        """
        height, width = featmap_size
        x = (prior_idxs % width + self.offset) * self.strides[level_idx][0]
        y = ((prior_idxs // width) % height +
             self.offset) * self.strides[level_idx][1]
        prioris = torch.stack([x, y], 1).to(dtype)
        prioris = prioris.to(device)
        return prioris
    

def distance2bbox(
    points: Tensor,
    distance: Tensor,
    max_shape = None
) -> Tensor:
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (B, N, 2) or (N, 2).
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom). Shape (B, N, 4) or (N, 4)
        max_shape (Union[Sequence[int], Tensor, Sequence[Sequence[int]]],
            optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If priors shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.

    Returns:
        Tensor: Boxes with shape (N, 4) or (B, N, 4)
    """

    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    if max_shape is not None:
        if bboxes.dim() == 2 and not torch.onnx.is_in_onnx_export():
            # speed up
            bboxes[:, 0::2].clamp_(min=0, max=max_shape[1])
            bboxes[:, 1::2].clamp_(min=0, max=max_shape[0])
            return bboxes

        if not isinstance(max_shape, torch.Tensor):
            max_shape = x1.new_tensor(max_shape)
        max_shape = max_shape[..., :2].type_as(x1)
        if max_shape.ndim == 2:
            assert bboxes.ndim == 3
            assert max_shape.size(0) == bboxes.size(0)

        min_xy = x1.new_tensor(0)
        max_xy = torch.cat([max_shape, max_shape],
                           dim=-1).flip(-1).unsqueeze(-2)
        bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
        bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

    return bboxes



class SimpleYOLOWorldDetector(nn.Module):
    """Implementation of YOLO World Series"""

    def __init__(self,
                 backbone_size,
                 prompt_dim=768,
                 num_prompts=512,) -> None:
        super().__init__()
        self.backbone = ConvNeXt(backbone_size)
        if backbone_size == 'base':
            scale_factor = 1.0
            in_channels = [128, 256, 512]
            self.img_size = (640, 640)
            self.grid_size = [6400, 1600, 400]
        elif backbone_size == 'large':
            scale_factor = 1.5
            in_channels = [192, 384, 768]
            self.img_size = (1280, 1280)
            self.grid_size = [6400*4, 1600*4, 400*4]
        self.neck = CSPRepBiFPANNeck(scale_factor)
        self.bbox_head = YOLOWorldHeadModule(embed_dims=prompt_dim, in_channels=in_channels, use_bn_head=True, use_einsum=True)

        embeddings = nn.functional.normalize(torch.randn(
                    (num_prompts, prompt_dim)), dim=-1)
        self.embeddings = nn.Parameter(embeddings)
        self.prior_generator = MlvlPointGenerator(strides=[8, 16, 32], offset=0.5)


    def forward(self, image_paths: List[str], rescale=True):
        inputs = []
        ratios = []
        offsets = []
        ori_shapes = []
        for image_path in image_paths:
            img = image_path
            width, height = img.size
            ori_shape = (height, width)
            ori_shapes.append(ori_shape)
            img, ratio, offset = letterbox(img, self.img_size)
            img = torch.tensor(np.array(img)).permute(2, 0, 1).to(self.embeddings.data.dtype)  # HWC, RGB
            img = img / 255.0
            inputs.append(img)
            ratios.append(ratio)
            offsets.append(offset)
        inputs = torch.stack(inputs, dim=0).cuda()
        img_feats = self.backbone(inputs)
        img_feats = self.neck(img_feats)
        results = self.head_predict(img_feats)

        for i in range(len(results)):
            # print(results[i]['bboxes'])
            results[i]['bboxes'] -= results[i]['bboxes'].new_tensor([
                        offsets[i][0], offsets[i][1], offsets[i][0], offsets[i][1]
                    ])
            # print(results[i]['bboxes'])
            if rescale:
                results[i]['bboxes'] /= ratios[i]
            results[i]['bboxes'][:, 0::2] = results[i]['bboxes'][:, 0::2].clamp_(0, ori_shapes[i][1])
            results[i]['bboxes'][:, 1::2] = results[i]['bboxes'][:, 1::2].clamp_(0, ori_shapes[i][0])
            # print(results[i]['bboxes'])
        return results

    def head_module_forward_single(
            self,
            img_feat: Tensor,
            cls_pred: nn.ModuleList,
            reg_pred: nn.ModuleList,
            cls_contrast: nn.ModuleList,
    ):
        module=self.bbox_head
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_embed = cls_contrast.norm(cls_embed)
        cls_logits = torch.einsum('bchw,kc->bkhw', cls_embed, self.embeddings)
        cls_logits = cls_logits * cls_contrast.logit_scale.exp() + cls_contrast.bias
        bbox_dist_preds = reg_pred(img_feat)
        if module.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, module.reg_max, h * w]
            ).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = (
                bbox_dist_preds.softmax(3).matmul(module.proj.view([-1, 1])).squeeze(-1)
            )
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        return cls_embed, bbox_preds,cls_logits
    

    def head_predict(self, img_feats):
        scales = torch.cat([
            torch.full([self.grid_size[0]], self.bbox_head.cls_contrasts[0].logit_scale.data, device=img_feats[0].device),
            torch.full([self.grid_size[1]], self.bbox_head.cls_contrasts[1].logit_scale.data, device=img_feats[0].device),
            torch.full([self.grid_size[2]], self.bbox_head.cls_contrasts[2].logit_scale.data, device=img_feats[0].device),
        ])
        bias = torch.cat([
            torch.full([self.grid_size[0]], self.bbox_head.cls_contrasts[0].bias.data, device=img_feats[0].device),
            torch.full([self.grid_size[1]], self.bbox_head.cls_contrasts[1].bias.data, device=img_feats[0].device),
            torch.full([self.grid_size[2]], self.bbox_head.cls_contrasts[2].bias.data, device=img_feats[0].device),
        ])

        bbox_embed, bbox_preds, cls_scores = [], [], []
        for i in range(len(img_feats)):
            box_embed, bbox_pred, cls_score = self.head_module_forward_single(
                img_feats[i],
                self.bbox_head.cls_preds[i],
                self.bbox_head.reg_preds[i],
                self.bbox_head.cls_contrasts[i],
            )
            bbox_embed.append(box_embed)
            bbox_preds.append(bbox_pred)
            cls_scores.append(cls_score)
        
        # objectness = None, with_objectness=False, multi_label=False
        txt_channel = bbox_embed[0].shape[1]
        num_imgs = bbox_embed[0].shape[0]
        featmap_sizes = [x.shape[2:] for x in bbox_preds]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, dtype=bbox_embed[0].dtype, device=bbox_embed[0].device
        )
        flatten_priors = torch.cat(mlvl_priors)
        mlvl_strides = [
            flatten_priors.new_full((featmap_size.numel(),), stride)
            for featmap_size, stride in zip(featmap_sizes, self.bbox_head.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)
        flatten_bbox_embed = [
            x.permute(0, 2, 3, 1).reshape(num_imgs, -1, txt_channel) for x in bbox_embed
        ]
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  cls_scores[0].shape[1])
            for cls_score in cls_scores
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_bbox_embed = torch.cat(flatten_bbox_embed, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_bbox_preds = flatten_bbox_preds * flatten_stride[None, :, None]
        flatten_decoded_bbox = distance2bbox(
            flatten_priors[None], flatten_bbox_preds
        )
        results_list = []
        for bbox, embed,scores in zip(
            flatten_decoded_bbox, flatten_bbox_embed,flatten_cls_scores
        ):
            scores, labels, keep_idxs, _ = filter_scores_and_topk(
                scores, 0.0, 30000)

            
            bbox = bbox[keep_idxs]
            embed = embed[keep_idxs]
            scales = scales[keep_idxs]
            bias = bias[keep_idxs]
            idx = torchvision.ops.batched_nms(bbox.float(), scores.float(), labels, 0.7)[:300]
            bbox = bbox[idx]
            embed = embed[idx]
            labels = labels[idx]
            results_list.append({
                'bboxes': bbox,
                'embeddings': embed,
                'scores': scores[idx],
                'labels': labels,
                'scales': scales[idx],
                'bias': bias[idx],
            })

        return results_list




class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: str,
    ):
        super().__init__()
        self.dataset = dataset
        self.images = []

        if dataset == 'coco' or dataset == 'paco':
            images = json.load(open(ds_collections[dataset]['ann_path']))['images']
            for ann in images:
                item = {
                    'id': ann['id'],
                    'image': ds_collections[dataset]['image_path'] + ann['file_name'],
                }
                self.images.append(item)
        elif dataset == "lvis":
            images = json.load(open(ds_collections[dataset]['ann_path']))['images']
            for ann in images:
                item = {
                    'id': ann['id'],
                    'image': ds_collections[dataset]['image_path'] + ann['coco_url'].replace('http://images.cocodataset.org/', ''),
                }
                self.images.append(item)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        ann = self.images[idx]

        data = {}
        data['id'] = int(ann['id'])
        data['image'] = Image.open(ann['image']).convert('RGB')

        return data


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def collate_fn(inputs):
    return inputs




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--wedetect_uni_checkpoint', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    if 'base' in args.wedetect_uni_checkpoint:
        model = SimpleYOLOWorldDetector(backbone_size='base', prompt_dim=768, num_prompts=256)
        checkpoint = torch.load(args.wedetect_uni_checkpoint, map_location='cpu')
    elif 'large' in args.wedetect_uni_checkpoint:
        model = SimpleYOLOWorldDetector(backbone_size='large', prompt_dim=768, num_prompts=256)
        checkpoint = torch.load(args.wedetect_uni_checkpoint, map_location='cpu')
    else:
        print("Please name the ckpt properly")
        assert NotImplementedError

    keys = list(checkpoint.keys())
    for key in keys:
        if 'backbone' in key:
            new_key = key.replace('backbone.image_model.model.', 'backbone.')
            checkpoint[new_key] = checkpoint.pop(key)
    # head
    keys = list(checkpoint.keys())
    for key in keys:
        if 'bbox_head' in key:
            new_key = key.replace('bbox_head.head_module.', 'bbox_head.')
            new_key = new_key.replace('0.2.', '0.6.')
            new_key = new_key.replace('1.2.', '1.6.')
            new_key = new_key.replace('2.2.', '2.6.')
            new_key = new_key.replace('1.bn', '4')
            new_key = new_key.replace('1.conv', '3')
            new_key = new_key.replace('0.bn', '1')
            new_key = new_key.replace('0.conv', '0')
            checkpoint[new_key] = checkpoint.pop(key)
    msg = model.load_state_dict(checkpoint, strict=False)
    print(msg)
    model = model.cuda()
    model.eval()


    random.seed(args.seed)
    dataset = ImageDataset(args.dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    image_ids = []
    all_boxes = []
    with torch.no_grad():
        for inputs in tqdm(dataloader, disable=torch.distributed.get_rank() != 0):
            image_ids.append(inputs[0]['id'])

            images = [inputs[0]['image']]
            outputs = model(images)
            boxes = outputs[0]['bboxes']
            
            all_boxes.append(boxes.cpu())

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_ids = [None for _ in range(world_size)]
    merged_all_boxes = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_ids, image_ids)
    torch.distributed.all_gather_object(merged_all_boxes, all_boxes)

    merged_ids = [_ for _ in itertools.chain.from_iterable(merged_ids)]
    merged_all_boxes = [_ for _ in itertools.chain.from_iterable(merged_all_boxes)]

    if torch.distributed.get_rank() == 0:
        print(f"Evaluating {args.dataset} ...")

        results = []
        for image_id, boxes in zip(merged_ids, merged_all_boxes):
            results.append({
                'image_id': int(image_id),
                'boxes': boxes,
            })
        # torch.save(results, f"{args.dataset}_{args.model}.pth")
        fast_eval_recall(args.dataset, results)
    torch.distributed.barrier()

