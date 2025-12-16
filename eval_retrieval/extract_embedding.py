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




class XLMRobertaLanguageBackbone(nn.Module):

    def __init__(
        self,
        ckpt_path,
        frozen_modules: Sequence[str] = (),
        dropout: float = 0.0,
        init_cfg= None,
    ) -> None:

        super().__init__()
        if 'base' in ckpt_path:
            self.head = nn.Linear(768, 768, bias=True) # XLarge
            model_name = "../xlm-roberta-base/"
        elif 'large' in ckpt_path:
            self.head = nn.Linear(1024, 768, bias=True) # XLarge
            model_name = "../xlm-roberta-large/"

        self.frozen_modules = frozen_modules
        cfg = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = XLMRobertaModel(cfg)
        self.language_dim = cfg.hidden_size
        
        
        # 加载 model 权重
        new_state_dict = OrderedDict()
        state_dict = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        )['state_dict']
        for k, v in state_dict.items():
            if k.startswith('backbone.text_model.'):
                name = k.split("backbone.text_model.")[-1]
                new_state_dict[name] = v
        msg = self.load_state_dict(new_state_dict, strict=True)
        print(msg)

        print("EXT-ENCODER xlm-roberta-base LOADING WEIGHTS !!!!")



    def forward(self, text: List[str]) -> Tensor:
        text = self.tokenizer(text=text, return_tensors="pt", padding=True)
        text = text.to(device=self.model.device)
        print(text['input_ids'].shape)

        txt_feats = self.model(**text)["last_hidden_state"][:, 0]
        print(txt_feats.shape)
        txt_feats = self.head(txt_feats)
        # txt_feats = txt_feats.reshape(-1, num_per_batch[0], txt_feats.shape[-1])
    
        return txt_feats



ds_collections = {
    'coco': {
        'ann_path': 'data/coco/annotations/instances_val2017.json',
        'image_path': 'data/coco/val2017/',
        "name_chinese": [x[0] for x in [["人"], ["自行车"], ["汽车"], ["摩托车"], ["飞机"], ["公共汽车"], ["火车"], ["卡车"], ["船"], ["交通灯"], ["消防栓"], ["停车标志"], ["停车计费表"], ["长凳"], ["鸟"], ["猫"], ["狗"], ["马"], ["羊"], ["牛"], ["大象"], ["熊"], ["斑马"], ["长颈鹿"], ["背包"], ["雨伞"], ["手提包"], ["领带"], ["手提箱"], ["飞盘"], ["滑雪板"], ["滑雪板"], ["运动球"], ["风筝"], ["棒球棒"], ["棒球手套"], ["滑板"], ["冲浪板"], ["网球拍"], ["瓶子"], ["酒杯"], ["杯子"], ["叉子"], ["刀"], ["勺子"], ["碗"], ["香蕉"], ["苹果"], ["三明治"], ["橙子"], ["西兰花"], ["胡萝卜"], ["热狗"], ["披萨"], ["甜甜圈"], ["蛋糕"], ["椅子"], ["沙发"], ["盆栽植物"], ["床"], ["餐桌"], ["厕所"], ["电视显示器"], ["笔记本电脑"], ["鼠标"], ["遥控器"], ["键盘"], ["手机"], ["微波炉"], ["烤箱"], ["烤面包机"], ["水槽"], ["冰箱"], ["书"], ["时钟"], ["花瓶"], ["剪刀"], ["泰迪熊小熊"], ["吹风机"], ["牙刷"]]],
        "name_english": ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train','truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep','cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella','handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard','sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard','surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork','knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange','broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair','couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv','laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    },
    'lvis': {
        'ann_path': 'data/lvis/lvis_v1_val.json',
        'image_path': 'data/coco/',
        "name_chinese": ['喷雾罐', '空调', '飞机', '闹钟', '酒精', '短吻鳄', '杏仁', '救护车', '放大器', '脚链', '天线', '苹果', '苹果酱', '杏子', '围裙', '水族馆', '防寒鞋', '臂章', '扶手椅', '衣柜', '盔甲', '洋蓟', '垃圾桶', '烟灰缸', '芦笋', '喷雾器', '鳄梨', '奖品', '遮阳篷', '斧子', '狒狒', '婴儿车', '篮球篮板', '背包', '手提包', '手提箱', '百吉饼', '风笛', '法棍面包', '诱饵', '球', '芭蕾舞裙', '气球', '竹子', '香蕉', '创可贴', '绷带', '印花大手帕', '班卓琴', '横幅', '杠铃', '驳船', '桶', '发夹', '手推车', '棒球垒', '棒球', '棒球棒', '棒球帽', '棒球手套', '篮子', '篮球', '苏萨号', '蝙蝠', '浴室地垫', '浴巾', '浴袍', '浴缸', '面糊', '电池', '沙滩球', '珠子', '豆腐', '豆袋坐垫', '无檐小便帽', '熊', '床', '便盆', '床罩', '奶牛', '牛肉', '传呼机', '啤酒瓶', '啤酒罐', '甲虫', '铃铛', '甜椒', '皮带', '皮带扣', '长凳', '贝雷帽', '围嘴', '圣经', '自行车', '帽舌', '广告牌', '活页夹', '双筒望远镜', '鸟', '喂鸟器', '水盆', '鸟笼', '鸟舍', '生日蛋糕', '生日贺卡', '海盗旗', '黑羊', '黑莓', '黑板', '毛毯', '运动夹克', '搅拌机', '软式飞艇', '闪光灯', '衬衫', '蓝莓', '游戏板', '船', '浮子', '线轴', '发夹', '煮鸡蛋', '饰扣式领带', '插锁', '螺栓', '引擎盖', '书', '书架', '小册子', '书签', '吊杆式麦克风', '靴子', '瓶子', '开瓶器', '花束', '弓', '蝴蝶结', '领结', '碗', '烟斗的斗', '圆顶礼帽', '保龄球', '盒子', '拳击手套', '吊裤带', '手镯', '黄铜牌匾', '胸罩', '面包箱', '面包', '缠腰布', '新娘礼服', '公文包', '西兰花', '胸针', '扫帚', '蛋糕', '抱子甘蓝', '泡泡糖', '桶', '马车', '公牛', '斗牛犬', '推土机', '子弹头列车', '公告板', '防弹背心', '扩音器', '小圆面包', '双层床', '浮标', '墨西哥卷饼', '公共汽车', '名片', '黄油', '蝴蝶', '纽扣', '出租车', '小屋', '守车', '橱柜', '储物柜', '蛋糕', '计算器', '日历', '小牛', '便携式摄像机', '骆驼', '相机', '相机镜头', '露营车', '罐头', '开罐器', '蜡烛', '烛台', '块状糖果', '拐杖糖', '手杖', '罐', '独木舟', '香瓜', '食堂', '帽子', '瓶盖', '披风', '卡布奇诺', '汽车', '铁路车辆', '电梯厢', '汽车电池', '身份证', '卡片', '开襟羊毛衫', '货船', '康乃馨', '马车', '胡萝卜', '大手提袋', '手推车', '纸板箱', '收银机', '砂锅', '磁带', '石膏模型', '猫', '花椰菜', '辣椒', 'CD播放机', '芹菜', '移动电话', '锁子甲', '椅子', '躺椅', '圣杯', '枝形吊灯', '家伙', '支票簿', '棋盘', '樱桃', '棋盘', '鸡肉', '鹰嘴豆', '辣椒', '钟声', '瓷器', '薯片', '扑克筹码', '巧克力棒', '巧克力蛋糕', '巧克力牛奶', '巧克力慕斯', '项圈', '案板', '筷子', '圣诞树', '滑梯', '苹果酒', '雪茄盒', '香烟', '香烟盒', '水箱', '单簧管', '扣子', '清洁剂', '防滑钉', '小柑橘', '夹子', '写字板', '剪刀', '披风', '时钟', '钟楼', '脏衣篮', '晾衣夹', '手拿包', '杯垫', '外套', '衣架', '衣帽架', '公鸡', '蟑螂', '可可粉', '椰子', '咖啡机', '咖啡桌', '咖啡壶', '线圈', '硬币', '滤器', '卷心菜', '着色材料', '密码锁', '奶嘴', '漫画书', '指南针', '电脑键盘', '调味品', '圆锥体', '控制', '敞篷车', '沙发床', '炉灶', '饼干', '烹饪用具', '冷藏箱', '软木塞', '软木板', '开瓶器', '可食用玉米', '玉米面包', '短号', '檐口', '玉米粉', '紧身胸衣', '服装', '美洲狮', '工作服', '牛铃', '牛仔帽', '螃蟹', '蟹肉', '薄脆饼干', '黑纱', '板条箱', '蜡笔', '奶油壶', '新月形面包', '婴儿床', '慢炖锅', '横杆', '油炸面包块', '乌鸦', '撬棍', '王冠', '十字架', '游轮', '警用巡逻车', '面包屑', '拐杖', '幼兽', '立方体', '黄瓜', '袖扣', '杯子', '奖杯', '橱柜', '纸杯蛋糕', '卷发器', '卷发棒', '窗帘', '垫子', '圆柱体', '钹', '匕首', '犬', '飞镖靶', '枣', '折叠躺椅', '鹿', '牙线', '书桌', '洗涤剂', '尿布', '日记', '骰子', '小艇', '餐桌', '男士晚礼服', '盘子', '碟形天线', '洗碗布', '擦碗布', '洗碗机', '洗碗机洗涤剂', '自动售货机', '跳水板', '一次性纸杯', '狗', '狗项圈', '玩偶', '美元', '玩具屋', '海豚', '驴', '门把手', '门口地垫', '甜甜圈', '鸽子', '蜻蜓', '抽屉', '内裤', '连衣裙', '礼帽', '礼服套装', '梳妆台', '钻头', '无人机', '滴管', '鼓状物', '鼓槌', '鸭子', '小鸭', '强力胶带', '行李袋', '哑铃', '垃圾桶', '簸箕', '鹰', '耳机', '耳塞', '耳环', '画架', '泡芙', '鳗鱼', '蛋', '蛋卷', '蛋黄', '打蛋器', '茄子', '电椅', '冰箱', '大象', '驼鹿', '信封', '橡皮擦', '蜗牛', '眼罩', '猎鹰', '风扇', '水龙头', '帽', '雪貂', '摩天轮', '渡船', '无花果', '战斗机', '雕像', '文件柜', '文件夹', '火灾报警器', '消防车', '灭火器', '消防水带', '壁炉', '消防栓', '急救箱', '鱼', '鱼饲料', '鱼缸', '鱼竿', '旗帜', '旗杆', '火烈鸟', '法兰绒', '襟翼', '闪光', '手电筒', '羊毛', '夹趾拖鞋', '鳍状肢', '插花', '香槟酒杯', '驹', '折叠椅', '食品加工机', '橄榄球', '橄榄球头盔', '脚凳', '餐叉', '叉车', '货车车厢', '法式吐司', '清新剂', '飞盘', '青蛙', '果汁', '煎锅', '软糖', '漏斗', '蒲团', '塞口物', '垃圾', '垃圾车', '花园水管', '漱口水', '滴水嘴兽', '大蒜', '防毒面具', '瞪羚', '明胶', '宝石', '发电机', '大熊猫', '礼品包装', '姜', '长颈鹿', '束带', '玻璃', '地球仪', '手套', '山羊', '护目镜', '金鱼', '高尔夫球杆', '高尔夫球车', '小船', '鹅', '猩猩', '葫芦', '葡萄', '擦菜板', '墓碑', '盆', '青豆', '葱', '煎锅', '烤架', '粗玉米粉', '灰熊', '购物袋', '吉他', '海鸥', '枪', '梳子', '发网', '发夹', '露背背心', '火腿', '汉堡包', '锤子', '吊床', '篮子', '仓鼠', '吹风机', '手持镜子', '毛巾', '手推车', '手铐', '手帕', '把手', '手锯', '精装书', '小风琴', '帽子', '帽盒', '面纱', '发带', '床头板', '前灯', '头巾', '耳机', '马笼头', '心脏', '加热器', '直升机', '头盔', '苍鹭', '高脚椅', '铰链', '河马', '曲棍球棒', '猪', '本垒', '蜂蜜', '通风柜', '钩子', '水烟袋', '大黄蜂', '马', '软管', '热气球', '加热板', '辣酱', '沙漏', '船屋', '蜂鸟', '豆泥', '北极熊', '冰淇淋', '冰棍', '制冰机', '冰袋', '溜冰鞋', '点火器', '吸入器', 'iPod', '熨斗', '熨衣板', '夹克衫', '果酱', '罐子', '牛仔裤', '吉普车', '软心豆粒糖', '运动衫', '喷气式飞机', '宝石', '珠宝', '操纵杆', '连身裤', '皮划艇', '桶', '狗窝', '水壶', '钥匙', '钥匙卡', '短裙', '和服', '水槽', '餐桌', '风筝', '小猫', '猕猴桃', '护膝', '刀', '编织针', '旋钮', '门环', '树袋熊', '实验室大褂', '梯子', '长柄勺', '瓢虫', '小羊', '羊排', '灯', '灯柱', '灯罩', '灯笼', '挂绳', '笔记本电脑', '千层面', '门闩', '割草机', '皮革', '紧身裤', '乐高积木', '豆类', '柠檬', '柠檬水', '生菜', '车牌', '救生圈', '救生衣', '电灯泡', '避雷针', '酸橙', '豪华轿车', '狮子', '润唇膏', '酒', '蜥蜴', '原木', '棒棒糖', '音箱', '双人沙发', '机关枪', '杂志', '磁铁', '邮件插槽', '邮箱', '绿头鸭', '木槌', '猛犸象', '海牛', '橘子', '马槽', '人孔', '地图', '记号笔', '鸡尾酒', '吉祥物', '土豆泥', '捣碎器', '面具', '桅杆', '垫子', '火柴盒', '床垫', '量杯', '测量杆', '肉丸', '药', '瓜', '麦克风', '显微镜', '微波炉', '里程碑', '牛奶', '牛奶罐', '奶昔', '小型货车', '薄荷糖', '镜子', '连指手套', '搅拌器', '钱', '显示器', '猴子', '发动机', '小型摩托车', '机动车辆', '摩托车', '土堆', '鼠标', '鼠标垫', '松饼', '马克杯', '蘑菇', '琴凳', '乐器', '指甲锉', '餐巾', '围巾', '项链', '领带', '针', '巢', '报纸', '报摊', '睡衣', '饲料袋', '动物鼻带', '笔记本', '便签本', '坚果', '胡桃夹子', '桨', '章鱼食物', '章鱼', '油灯', '橄榄油', '煎蛋卷', '洋葱', '橙子', '橙汁', '鸵鸟', '长软椅', '烤箱', '工装裤', '猫头鹰', '小包', '印台', '垫子', '桨', '挂锁', '画笔', '绘画', '睡衣', '调色板', '平底锅', '锅', '薄煎饼', '连裤袜', '木瓜', '纸盘子', '纸巾', '平装书', '镇纸', '降落伞', '小鹦鹉', '滑翔伞运动', '阳伞', '羊皮纸', '派克大衣', '停车计时器', '鹦鹉', '客车', '客船', '护照', '油酥点心', '小馅饼', '豌豆', '桃子', '花生酱', '梨', '果蔬削皮工具', '木制假腿', '纤维板', '鹈鹕', '钢笔', '铅笔', '铅笔盒', '削笔器', '钟摆', '企鹅', '三角旗', '便士', '胡椒', '胡椒研磨器', '香水', '柿子', '人', '宠物', '教堂长椅', '电话簿', '留声机唱片', '钢琴', '泡菜', '皮卡车', '馅饼', '鸽子', '存钱罐', '枕头', '别针', '菠萝', '松果', '乒乓球', '风车', '烟斗', '管子', '手枪', '皮塔饼', '水壶', '干草叉', '披萨', '餐垫', '盘子', '盘', '围栏', '钳子', '犁', '羽毛', '怀表', '袖珍小刀', '拨火棍', '杆', '马球衫', '披风', '小马', '台球桌', '汽水', '邮箱', '明信片', '海报', '锅', '花盆', '土豆', '隔热垫', '陶器', '小袋', '挖掘机', '虾', '椒盐脆饼', '打印机', '射弹', '投影仪', '螺旋桨', '李子干', '布丁', '河豚', '海鹦', '哈巴狗', '南瓜', '穿孔机', '木偶', '小狗', '油炸玉米粉饼', '乳蛋饼', '被子', '兔子', '赛车', '球拍', '雷达', '散热器', '收音机', '萝卜', '木筏', '布娃娃', '雨衣', '公羊', '树莓', '老鼠', '剃须刀片', '榨汁器', '后视镜', '收据', '躺椅', '电唱机', '反射器', '遥控器', '犀牛', '排骨', '步枪', '戒指', '内河船', '路线图', '长袍', '摇椅', '啮齿动物', '旱冰鞋', '直排轮滑鞋', '擀面杖', '汽水', '路由器', '橡皮筋', '地毯', '塑料袋', '鞍座', '鞍毯', '鞍囊', '安全别针', '帆', '沙拉', '沙拉盘', '萨拉米香肠', '鲑鱼', '鲑鱼肉', '萨尔萨辣酱', '盐瓶', '凉鞋', '三明治', '书包', '锅', '茶碟', '香肠', '锯木架', '萨克斯管', '天平', '稻草人', '围巾', '校车', '剪刀', '记分牌', '刮刀', '螺丝刀', '刷子', '雕塑', '海鸟', '海马', '水上飞机', '贝壳', '缝纫机', '摇瓶', '洗发水', '鲨鱼', '削具', '马克笔', '剃须刀', '剃须膏', '披肩', '大剪刀', '羊', '牧羊犬', '果汁牛奶冻', '盾', '衬衫', '鞋', '购物袋', '购物车', '短裤', '烈酒杯', '单肩包', '铲子', '淋浴喷头', '浴帽', '浴帘', '切碎机', '招牌', '筒仓', '水槽', '滑板', '串肉扦', '滑雪板', '滑雪靴', '滑雪外套', '滑雪杖', '裙子', '无檐便帽', '雪橇', '睡袋', '悬带', '拖鞋', '冰沙饮品', '蛇', '滑雪板', '雪人', '雪地摩托', '肥皂', '足球', '短袜', '沙发', '垒球', '太阳能电池板', '宽边帽', '汤', '汤碗', '汤匙', '酸奶油', '豆浆', '航天飞机', '烟火', '抹刀', '矛', '眼镜', '调料架', '蜘蛛', '小龙虾', '海绵', '勺', '运动装', '聚光灯', '鱿鱼', '松鼠', '公共马车', '订书机', '海星', '雕塑', '肉排', '牛排刀', '方向盘', '折梯', '踏脚凳', '立体声音响系统', '炖菜', '搅拌器', '马镫', '凳子', '停车标志', '刹车灯', '炉灶', '滤网', '带子', '稻草', '草莓', '路标', '路灯', '奶酪', '触控笔', '低音炮', '糖碗', '甘蔗', '西装', '向日葵', '太阳镜', '遮阳帽', '冲浪板', '寿司', '拖把', '运动裤', '吸汗带', '毛衣', '运动衫', '红薯', '泳衣', '剑', '注射器', '辣椒酱', '乒乓球桌', '桌子', '台灯', '桌布', '转速表', '墨西哥玉米卷饼', '标签', '尾灯', '手鼓', '坦克', '坦克', '吊带背心', '磁带', '卷尺', '挂毯', '防水油布', '格子', '流苏', '茶包', '茶杯', '茶壶', '茶壶', '泰迪熊', '电话', '电话亭', '电话线杆', '长焦镜头', '电视摄像机', '电视机', '网球', '网球拍', '龙舌兰酒', '温度计', '保温瓶', '恒温器', '顶针', '线', '图钉', '冠状头饰', '老虎', '紧身衣', '计时器', '锡纸', '金属箔', '纸巾', '烤面包片', '烤面包机', '多士炉烤箱', '厕所', '卫生纸', '番茄', '钳子', '工具箱', '牙刷', '牙膏', '牙签', '盖子', '墨西哥薄饼', '拖车', '毛巾', '毛巾架', '玩具', '拖拉机', '交通信号灯', '越野摩托车', '牵引式挂车', '火车', '蹦床', '托盘', '风衣', '三角铁', '三轮车', '三脚架', '裤子', '卡车', '松露', '后备箱', '大桶', '头巾', '火鸡肉', '芜菁', '龟', '高领毛衣', '打字机', '雨伞', '内衣', '独轮车', '小便池', '瓮', '吸尘器', '花瓶', '自动售货机', '通风口', '背心', '录像带', '醋', '小提琴', '伏特加酒', '排球', '秃鹫', '华夫饼', '华夫饼烤盘', '四轮马车', '马车车轮', '手杖', '挂钟', '墙上插座', '钱包', '海象', '衣柜', '洗脸盆', '洗衣机', '手表', '水瓶', '饮水机', '水龙头', '热水器', '水壶', '水枪', '水上摩托', '滑水橇', '水塔', '喷壶', '西瓜', '风向标', '网络摄像头', '结婚蛋糕', '结婚戒指', '潜水服', '轮子', '轮椅', '生奶油', '口哨', '假发', '风铃', '风车', '花盆箱', '挡风玻璃雨刮器', '风向袋', '葡萄酒瓶', '冰酒桶', '葡萄酒杯', '眼罩', '炒锅', '狼', '木勺', '花环', '扳手', '腕带', '腕带', '游艇', '酸奶', '轭', '斑马', '西葫芦'],
        "name_english": ['aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock',
         'alcohol', 'alligator', 'almond', 'ambulance', 'amplifier', 'anklet',
         'antenna', 'apple', 'applesauce', 'apricot', 'apron', 'aquarium',
         'arctic_(type_of_shoe)', 'armband', 'armchair', 'armoire', 'armor',
         'artichoke', 'trash_can', 'ashtray', 'asparagus', 'atomizer',
         'avocado', 'award', 'awning', 'ax', 'baboon', 'baby_buggy',
         'basketball_backboard', 'backpack', 'handbag', 'suitcase', 'bagel',
         'bagpipe', 'baguet', 'bait', 'ball', 'ballet_skirt', 'balloon',
         'bamboo', 'banana', 'Band_Aid', 'bandage', 'bandanna', 'banjo',
         'banner', 'barbell', 'barge', 'barrel', 'barrette', 'barrow',
         'baseball_base', 'baseball', 'baseball_bat', 'baseball_cap',
         'baseball_glove', 'basket', 'basketball', 'bass_horn', 'bat_(animal)',
         'bath_mat', 'bath_towel', 'bathrobe', 'bathtub', 'batter_(food)',
         'battery', 'beachball', 'bead', 'bean_curd', 'beanbag', 'beanie',
         'bear', 'bed', 'bedpan', 'bedspread', 'cow', 'beef_(food)', 'beeper',
         'beer_bottle', 'beer_can', 'beetle', 'bell', 'bell_pepper', 'belt',
         'belt_buckle', 'bench', 'beret', 'bib', 'Bible', 'bicycle', 'visor',
         'billboard', 'binder', 'binoculars', 'bird', 'birdfeeder', 'birdbath',
         'birdcage', 'birdhouse', 'birthday_cake', 'birthday_card',
         'pirate_flag', 'black_sheep', 'blackberry', 'blackboard', 'blanket',
         'blazer', 'blender', 'blimp', 'blinker', 'blouse', 'blueberry',
         'gameboard', 'boat', 'bob', 'bobbin', 'bobby_pin', 'boiled_egg',
         'bolo_tie', 'deadbolt', 'bolt', 'bonnet', 'book', 'bookcase',
         'booklet', 'bookmark', 'boom_microphone', 'boot', 'bottle',
         'bottle_opener', 'bouquet', 'bow_(weapon)',
         'bow_(decorative_ribbons)', 'bow-tie', 'bowl', 'pipe_bowl',
         'bowler_hat', 'bowling_ball', 'box', 'boxing_glove', 'suspenders',
         'bracelet', 'brass_plaque', 'brassiere', 'bread-bin', 'bread',
         'breechcloth', 'bridal_gown', 'briefcase', 'broccoli', 'broach',
         'broom', 'brownie', 'brussels_sprouts', 'bubble_gum', 'bucket',
         'horse_buggy', 'bull', 'bulldog', 'bulldozer', 'bullet_train',
         'bulletin_board', 'bulletproof_vest', 'bullhorn', 'bun', 'bunk_bed',
         'buoy', 'burrito', 'bus_(vehicle)', 'business_card', 'butter',
         'butterfly', 'button', 'cab_(taxi)', 'cabana', 'cabin_car', 'cabinet',
         'locker', 'cake', 'calculator', 'calendar', 'calf', 'camcorder',
         'camel', 'camera', 'camera_lens', 'camper_(vehicle)', 'can',
         'can_opener', 'candle', 'candle_holder', 'candy_bar', 'candy_cane',
         'walking_cane', 'canister', 'canoe', 'cantaloup', 'canteen',
         'cap_(headwear)', 'bottle_cap', 'cape', 'cappuccino',
         'car_(automobile)', 'railcar_(part_of_a_train)', 'elevator_car',
         'car_battery', 'identity_card', 'card', 'cardigan', 'cargo_ship',
         'carnation', 'horse_carriage', 'carrot', 'tote_bag', 'cart', 'carton',
         'cash_register', 'casserole', 'cassette', 'cast', 'cat',
         'cauliflower', 'cayenne_(spice)', 'CD_player', 'celery',
         'cellular_telephone', 'chain_mail', 'chair', 'chaise_longue',
         'chalice', 'chandelier', 'chap', 'checkbook', 'checkerboard',
         'cherry', 'chessboard', 'chicken_(animal)', 'chickpea',
         'chili_(vegetable)', 'chime', 'chinaware', 'crisp_(potato_chip)',
         'poker_chip', 'chocolate_bar', 'chocolate_cake', 'chocolate_milk',
         'chocolate_mousse', 'choker', 'chopping_board', 'chopstick',
         'Christmas_tree', 'slide', 'cider', 'cigar_box', 'cigarette',
         'cigarette_case', 'cistern', 'clarinet', 'clasp', 'cleansing_agent',
         'cleat_(for_securing_rope)', 'clementine', 'clip', 'clipboard',
         'clippers_(for_plants)', 'cloak', 'clock', 'clock_tower',
         'clothes_hamper', 'clothespin', 'clutch_bag', 'coaster', 'coat',
         'coat_hanger', 'coatrack', 'cock', 'cockroach', 'cocoa_(beverage)',
         'coconut', 'coffee_maker', 'coffee_table', 'coffeepot', 'coil',
         'coin', 'colander', 'coleslaw', 'coloring_material',
         'combination_lock', 'pacifier', 'comic_book', 'compass',
         'computer_keyboard', 'condiment', 'cone', 'control',
         'convertible_(automobile)', 'sofa_bed', 'cooker', 'cookie',
         'cooking_utensil', 'cooler_(for_food)', 'cork_(bottle_plug)',
         'corkboard', 'corkscrew', 'edible_corn', 'cornbread', 'cornet',
         'cornice', 'cornmeal', 'corset', 'costume', 'cougar', 'coverall',
         'cowbell', 'cowboy_hat', 'crab_(animal)', 'crabmeat', 'cracker',
         'crape', 'crate', 'crayon', 'cream_pitcher', 'crescent_roll', 'crib',
         'crock_pot', 'crossbar', 'crouton', 'crow', 'crowbar', 'crown',
         'crucifix', 'cruise_ship', 'police_cruiser', 'crumb', 'crutch',
         'cub_(animal)', 'cube', 'cucumber', 'cufflink', 'cup', 'trophy_cup',
         'cupboard', 'cupcake', 'hair_curler', 'curling_iron', 'curtain',
         'cushion', 'cylinder', 'cymbal', 'dagger', 'dalmatian', 'dartboard',
         'date_(fruit)', 'deck_chair', 'deer', 'dental_floss', 'desk',
         'detergent', 'diaper', 'diary', 'die', 'dinghy', 'dining_table',
         'tux', 'dish', 'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher',
         'dishwasher_detergent', 'dispenser', 'diving_board', 'Dixie_cup',
         'dog', 'dog_collar', 'doll', 'dollar', 'dollhouse', 'dolphin',
         'domestic_ass', 'doorknob', 'doormat', 'doughnut', 'dove',
         'dragonfly', 'drawer', 'underdrawers', 'dress', 'dress_hat',
         'dress_suit', 'dresser', 'drill', 'drone', 'dropper',
         'drum_(musical_instrument)', 'drumstick', 'duck', 'duckling',
         'duct_tape', 'duffel_bag', 'dumbbell', 'dumpster', 'dustpan', 'eagle',
         'earphone', 'earplug', 'earring', 'easel', 'eclair', 'eel', 'egg',
         'egg_roll', 'egg_yolk', 'eggbeater', 'eggplant', 'electric_chair',
         'refrigerator', 'elephant', 'elk', 'envelope', 'eraser', 'escargot',
         'eyepatch', 'falcon', 'fan', 'faucet', 'fedora', 'ferret',
         'Ferris_wheel', 'ferry', 'fig_(fruit)', 'fighter_jet', 'figurine',
         'file_cabinet', 'file_(tool)', 'fire_alarm', 'fire_engine',
         'fire_extinguisher', 'fire_hose', 'fireplace', 'fireplug',
         'first-aid_kit', 'fish', 'fish_(food)', 'fishbowl', 'fishing_rod',
         'flag', 'flagpole', 'flamingo', 'flannel', 'flap', 'flash',
         'flashlight', 'fleece', 'flip-flop_(sandal)', 'flipper_(footwear)',
         'flower_arrangement', 'flute_glass', 'foal', 'folding_chair',
         'food_processor', 'football_(American)', 'football_helmet',
         'footstool', 'fork', 'forklift', 'freight_car', 'French_toast',
         'freshener', 'frisbee', 'frog', 'fruit_juice', 'frying_pan', 'fudge',
         'funnel', 'futon', 'gag', 'garbage', 'garbage_truck', 'garden_hose',
         'gargle', 'gargoyle', 'garlic', 'gasmask', 'gazelle', 'gelatin',
         'gemstone', 'generator', 'giant_panda', 'gift_wrap', 'ginger',
         'giraffe', 'cincture', 'glass_(drink_container)', 'globe', 'glove',
         'goat', 'goggles', 'goldfish', 'golf_club', 'golfcart',
         'gondola_(boat)', 'goose', 'gorilla', 'gourd', 'grape', 'grater',
         'gravestone', 'gravy_boat', 'green_bean', 'green_onion', 'griddle',
         'grill', 'grits', 'grizzly', 'grocery_bag', 'guitar', 'gull', 'gun',
         'hairbrush', 'hairnet', 'hairpin', 'halter_top', 'ham', 'hamburger',
         'hammer', 'hammock', 'hamper', 'hamster', 'hair_dryer', 'hand_glass',
         'hand_towel', 'handcart', 'handcuff', 'handkerchief', 'handle',
         'handsaw', 'hardback_book', 'harmonium', 'hat', 'hatbox', 'veil',
         'headband', 'headboard', 'headlight', 'headscarf', 'headset',
         'headstall_(for_horses)', 'heart', 'heater', 'helicopter', 'helmet',
         'heron', 'highchair', 'hinge', 'hippopotamus', 'hockey_stick', 'hog',
         'home_plate_(baseball)', 'honey', 'fume_hood', 'hook', 'hookah',
         'hornet', 'horse', 'hose', 'hot-air_balloon', 'hotplate', 'hot_sauce',
         'hourglass', 'houseboat', 'hummingbird', 'hummus', 'polar_bear',
         'icecream', 'popsicle', 'ice_maker', 'ice_pack', 'ice_skate',
         'igniter', 'inhaler', 'iPod', 'iron_(for_clothing)', 'ironing_board',
         'jacket', 'jam', 'jar', 'jean', 'jeep', 'jelly_bean', 'jersey',
         'jet_plane', 'jewel', 'jewelry', 'joystick', 'jumpsuit', 'kayak',
         'keg', 'kennel', 'kettle', 'key', 'keycard', 'kilt', 'kimono',
         'kitchen_sink', 'kitchen_table', 'kite', 'kitten', 'kiwi_fruit',
         'knee_pad', 'knife', 'knitting_needle', 'knob', 'knocker_(on_a_door)',
         'koala', 'lab_coat', 'ladder', 'ladle', 'ladybug', 'lamb_(animal)',
         'lamb-chop', 'lamp', 'lamppost', 'lampshade', 'lantern', 'lanyard',
         'laptop_computer', 'lasagna', 'latch', 'lawn_mower', 'leather',
         'legging_(clothing)', 'Lego', 'legume', 'lemon', 'lemonade',
         'lettuce', 'license_plate', 'life_buoy', 'life_jacket', 'lightbulb',
         'lightning_rod', 'lime', 'limousine', 'lion', 'lip_balm', 'liquor',
         'lizard', 'log', 'lollipop', 'speaker_(stereo_equipment)', 'loveseat',
         'machine_gun', 'magazine', 'magnet', 'mail_slot', 'mailbox_(at_home)',
         'mallard', 'mallet', 'mammoth', 'manatee', 'mandarin_orange',
         'manger', 'manhole', 'map', 'marker', 'martini', 'mascot',
         'mashed_potato', 'masher', 'mask', 'mast', 'mat_(gym_equipment)',
         'matchbox', 'mattress', 'measuring_cup', 'measuring_stick',
         'meatball', 'medicine', 'melon', 'microphone', 'microscope',
         'microwave_oven', 'milestone', 'milk', 'milk_can', 'milkshake',
         'minivan', 'mint_candy', 'mirror', 'mitten', 'mixer_(kitchen_tool)',
         'money', 'monitor_(computer_equipment) computer_monitor', 'monkey',
         'motor', 'motor_scooter', 'motor_vehicle', 'motorcycle',
         'mound_(baseball)', 'mouse_(computer_equipment)', 'mousepad',
         'muffin', 'mug', 'mushroom', 'music_stool', 'musical_instrument',
         'nailfile', 'napkin', 'neckerchief', 'necklace', 'necktie', 'needle',
         'nest', 'newspaper', 'newsstand', 'nightshirt',
         'nosebag_(for_animals)', 'noseband_(for_animals)', 'notebook',
         'notepad', 'nut', 'nutcracker', 'oar', 'octopus_(food)',
         'octopus_(animal)', 'oil_lamp', 'olive_oil', 'omelet', 'onion',
         'orange_(fruit)', 'orange_juice', 'ostrich', 'ottoman', 'oven',
         'overalls_(clothing)', 'owl', 'packet', 'inkpad', 'pad', 'paddle',
         'padlock', 'paintbrush', 'painting', 'pajamas', 'palette',
         'pan_(for_cooking)', 'pan_(metal_container)', 'pancake', 'pantyhose',
         'papaya', 'paper_plate', 'paper_towel', 'paperback_book',
         'paperweight', 'parachute', 'parakeet', 'parasail_(sports)',
         'parasol', 'parchment', 'parka', 'parking_meter', 'parrot',
         'passenger_car_(part_of_a_train)', 'passenger_ship', 'passport',
         'pastry', 'patty_(food)', 'pea_(food)', 'peach', 'peanut_butter',
         'pear', 'peeler_(tool_for_fruit_and_vegetables)', 'wooden_leg',
         'pegboard', 'pelican', 'pen', 'pencil', 'pencil_box',
         'pencil_sharpener', 'pendulum', 'penguin', 'pennant', 'penny_(coin)',
         'pepper', 'pepper_mill', 'perfume', 'persimmon', 'person', 'pet',
         'pew_(church_bench)', 'phonebook', 'phonograph_record', 'piano',
         'pickle', 'pickup_truck', 'pie', 'pigeon', 'piggy_bank', 'pillow',
         'pin_(non_jewelry)', 'pineapple', 'pinecone', 'ping-pong_ball',
         'pinwheel', 'tobacco_pipe', 'pipe', 'pistol', 'pita_(bread)',
         'pitcher_(vessel_for_liquid)', 'pitchfork', 'pizza', 'place_mat',
         'plate', 'platter', 'playpen', 'pliers', 'plow_(farm_equipment)',
         'plume', 'pocket_watch', 'pocketknife', 'poker_(fire_stirring_tool)',
         'pole', 'polo_shirt', 'poncho', 'pony', 'pool_table', 'pop_(soda)',
         'postbox_(public)', 'postcard', 'poster', 'pot', 'flowerpot',
         'potato', 'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn',
         'pretzel', 'printer', 'projectile_(weapon)', 'projector', 'propeller',
         'prune', 'pudding', 'puffer_(fish)', 'puffin', 'pug-dog', 'pumpkin',
         'puncher', 'puppet', 'puppy', 'quesadilla', 'quiche', 'quilt',
         'rabbit', 'race_car', 'racket', 'radar', 'radiator', 'radio_receiver',
         'radish', 'raft', 'rag_doll', 'raincoat', 'ram_(animal)', 'raspberry',
         'rat', 'razorblade', 'reamer_(juicer)', 'rearview_mirror', 'receipt',
         'recliner', 'record_player', 'reflector', 'remote_control',
         'rhinoceros', 'rib_(food)', 'rifle', 'ring', 'river_boat', 'road_map',
         'robe', 'rocking_chair', 'rodent', 'roller_skate', 'Rollerblade',
         'rolling_pin', 'root_beer', 'router_(computer_equipment)',
         'rubber_band', 'runner_(carpet)', 'plastic_bag',
         'saddle_(on_an_animal)', 'saddle_blanket', 'saddlebag', 'safety_pin',
         'sail', 'salad', 'salad_plate', 'salami', 'salmon_(fish)',
         'salmon_(food)', 'salsa', 'saltshaker', 'sandal_(type_of_shoe)',
         'sandwich', 'satchel', 'saucepan', 'saucer', 'sausage', 'sawhorse',
         'saxophone', 'scale_(measuring_instrument)', 'scarecrow', 'scarf',
         'school_bus', 'scissors', 'scoreboard', 'scraper', 'screwdriver',
         'scrubbing_brush', 'sculpture', 'seabird', 'seahorse', 'seaplane',
         'seashell', 'sewing_machine', 'shaker', 'shampoo', 'shark',
         'sharpener', 'Sharpie', 'shaver_(electric)', 'shaving_cream', 'shawl',
         'shears', 'sheep', 'shepherd_dog', 'sherbert', 'shield', 'shirt',
         'shoe', 'shopping_bag', 'shopping_cart', 'short_pants', 'shot_glass',
         'shoulder_bag', 'shovel', 'shower_head', 'shower_cap',
         'shower_curtain', 'shredder_(for_paper)', 'signboard', 'silo', 'sink',
         'skateboard', 'skewer', 'ski', 'ski_boot', 'ski_parka', 'ski_pole',
         'skirt', 'skullcap', 'sled', 'sleeping_bag', 'sling_(bandage)',
         'slipper_(footwear)', 'smoothie', 'snake', 'snowboard', 'snowman',
         'snowmobile', 'soap', 'soccer_ball', 'sock', 'sofa', 'softball',
         'solar_array', 'sombrero', 'soup', 'soup_bowl', 'soupspoon',
         'sour_cream', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)',
         'spatula', 'spear', 'spectacles', 'spice_rack', 'spider', 'crawfish',
         'sponge', 'spoon', 'sportswear', 'spotlight', 'squid_(food)',
         'squirrel', 'stagecoach', 'stapler_(stapling_machine)', 'starfish',
         'statue_(sculpture)', 'steak_(food)', 'steak_knife', 'steering_wheel',
         'stepladder', 'step_stool', 'stereo_(sound_system)', 'stew',
         'stirrer', 'stirrup', 'stool', 'stop_sign', 'brake_light', 'stove',
         'strainer', 'strap', 'straw_(for_drinking)', 'strawberry',
         'street_sign', 'streetlight', 'string_cheese', 'stylus', 'subwoofer',
         'sugar_bowl', 'sugarcane_(plant)', 'suit_(clothing)', 'sunflower',
         'sunglasses', 'sunhat', 'surfboard', 'sushi', 'mop', 'sweat_pants',
         'sweatband', 'sweater', 'sweatshirt', 'sweet_potato', 'swimsuit',
         'sword', 'syringe', 'Tabasco_sauce', 'table-tennis_table', 'table',
         'table_lamp', 'tablecloth', 'tachometer', 'taco', 'tag', 'taillight',
         'tambourine', 'army_tank', 'tank_(storage_vessel)',
         'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tape_measure',
         'tapestry', 'tarp', 'tartan', 'tassel', 'tea_bag', 'teacup',
         'teakettle', 'teapot', 'teddy_bear', 'telephone', 'telephone_booth',
         'telephone_pole', 'telephoto_lens', 'television_camera',
         'television_set', 'tennis_ball', 'tennis_racket', 'tequila',
         'thermometer', 'thermos_bottle', 'thermostat', 'thimble', 'thread',
         'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer',
         'tinfoil', 'tinsel', 'tissue_paper', 'toast_(food)', 'toaster',
         'toaster_oven', 'toilet', 'toilet_tissue', 'tomato', 'tongs',
         'toolbox', 'toothbrush', 'toothpaste', 'toothpick', 'cover',
         'tortilla', 'tow_truck', 'towel', 'towel_rack', 'toy',
         'tractor_(farm_equipment)', 'traffic_light', 'dirt_bike',
         'trailer_truck', 'train_(railroad_vehicle)', 'trampoline', 'tray',
         'trench_coat', 'triangle_(musical_instrument)', 'tricycle', 'tripod',
         'trousers', 'truck', 'truffle_(chocolate)', 'trunk', 'vat', 'turban',
         'turkey_(food)', 'turnip', 'turtle', 'turtleneck_(clothing)',
         'typewriter', 'umbrella', 'underwear', 'unicycle', 'urinal', 'urn',
         'vacuum_cleaner', 'vase', 'vending_machine', 'vent', 'vest',
         'videotape', 'vinegar', 'violin', 'vodka', 'volleyball', 'vulture',
         'waffle', 'waffle_iron', 'wagon', 'wagon_wheel', 'walking_stick',
         'wall_clock', 'wall_socket', 'wallet', 'walrus', 'wardrobe',
         'washbasin', 'automatic_washer', 'watch', 'water_bottle',
         'water_cooler', 'water_faucet', 'water_heater', 'water_jug',
         'water_gun', 'water_scooter', 'water_ski', 'water_tower',
         'watering_can', 'watermelon', 'weathervane', 'webcam', 'wedding_cake',
         'wedding_ring', 'wet_suit', 'wheel', 'wheelchair', 'whipped_cream',
         'whistle', 'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)',
         'windshield_wiper', 'windsock', 'wine_bottle', 'wine_bucket',
         'wineglass', 'blinder_(for_horses)', 'wok', 'wolf', 'wooden_spoon',
         'wreath', 'wrench', 'wristband', 'wristlet', 'yacht', 'yogurt',
         'yoke_(animal_equipment)', 'zebra', 'zucchini']
    },
}


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: str,
    ):
        super().__init__()
        self.dataset = dataset
        self.images = []

        if dataset == 'coco':
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
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--wedetect_checkpoint', type=str, default='')
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

    language_encoder = XLMRobertaLanguageBackbone(args.wedetect_checkpoint).cuda()
    name_chinese = ds_collections[args.dataset]['name_chinese']
    text_embeddings = []
    num_iters = len(name_chinese) // 80 + 1 if len(name_chinese) % 80 != 0 else len(name_chinese) // 80
    with torch.no_grad():
        for i in range(num_iters):
            text_embeddings.append(language_encoder(name_chinese[i*80: (i+1)*80]))
    text_embeddings = torch.cat(text_embeddings)
    text_embeddings = F.normalize(text_embeddings, dim=-1)


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
    embeddings = []
    scales = []
    biases = []
    with torch.no_grad():
        for inputs in tqdm(dataloader, disable=torch.distributed.get_rank() != 0):
            image_ids.append(inputs[0]['id'])

            images = [inputs[0]['image']]
            outputs = model(images)
            embedding = outputs[0]['embeddings'].cpu()
            scale = outputs[0]['scales'].cpu()
            bias = outputs[0]['bias'].cpu()
            
            embeddings.append(embedding.cpu())
            scales.append(scale.cpu())
            biases.append(bias.cpu())

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_ids = [None for _ in range(world_size)]
    merged_embeddings = [None for _ in range(world_size)]
    merged_scales = [None for _ in range(world_size)]
    merged_biases = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_ids, image_ids)
    torch.distributed.all_gather_object(merged_embeddings, embeddings)
    torch.distributed.all_gather_object(merged_scales, scales)
    torch.distributed.all_gather_object(merged_biases, biases)

    merged_ids = [_ for _ in itertools.chain.from_iterable(merged_ids)]
    merged_embeddings = [_ for _ in itertools.chain.from_iterable(merged_embeddings)]
    merged_scales = [_ for _ in itertools.chain.from_iterable(merged_scales)]
    merged_biases = [_ for _ in itertools.chain.from_iterable(merged_biases)]

    if torch.distributed.get_rank() == 0:
        print(f"Evaluating {args.dataset} ...")

        results = []
        for image_id, embedding, scale, bias in zip(merged_ids, merged_embeddings, merged_scales, merged_biases):
            results.append({
                'image_id': int(image_id),
                'embedding': embedding,
                'scale': scale,
                'bias': bias,
            })
        torch.save({"image_embedding": results, "text_embedding": text_embeddings}, f"{args.dataset}_{args.model}.pth")
    torch.distributed.barrier()

