# Copyright (c) Tencent Inc. All rights reserved.
"""Standalone PyTorch COCO evaluation for WeDetect.

This script re-implements the WeDetect *dual-tower* open-vocabulary detector
without any mmcv / mmdet dependency, so the detection result can be verified
end-to-end against the COCO ``instances_val2017`` benchmark:

    Vision  tower : image -> ConvNeXt backbone -> CSPRepBiFPAN neck -> head
    Language tower: text  -> XLM-RoBERTa -> linear head -> L2 normalize
    Fusion        : BNContrastiveHead (norm + einsum + logit_scale + bias)

It loads a checkpoint trained with mmdet, remaps the state-dict keys to the
plain PyTorch modules defined here, runs inference over ``val2017`` and reports
COCO mAP via ``pycocotools``.

Usage
-----
    python deploy/test_coco_pytorch.py \
        --variant base \
        --language-model xlm-roberta-base \
        --checkpoint checkpoints/wedetect_base.pth \
        --coco-ann  /path/to/instances_val2017.json \
        --coco-img  /path/to/val2017
"""
import argparse
from collections import OrderedDict
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import tqdm
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn.modules.utils import _pair

try:
    from transformers import AutoConfig, AutoTokenizer, XLMRobertaModel
except ImportError:  # transformers is only needed for the language tower
    AutoConfig = AutoTokenizer = XLMRobertaModel = None


# --------------------------------------------------------------------------- #
#  Pre-processing helpers                                                      #
# --------------------------------------------------------------------------- #
def letterbox(
    img: Image.Image,
    new_shape=(640, 640),
    color=(114, 114, 114),
    scale_up=True,
):
    """Resize a PIL image with unchanged aspect ratio using padding.

    Returns:
        img (PIL.Image): the letter-boxed image.
        ratio (float)  : the scale ratio applied to the original image.
        (dw, dh)       : half of the horizontal / vertical padding, used to
                         recover boxes back to the original image space.
    """
    shape = img.size  # (w, h)
    new_shape = (new_shape[1], new_shape[0])  # to (w, h)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:  # only shrink, never enlarge
        r = min(r, 1.0)

    new_unpad = (int(round(shape[0] * r)), int(round(shape[1] * r)))
    img_resized = img.resize(new_unpad, Image.Resampling.BILINEAR)

    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]
    left, top = dw // 2, dh // 2

    img_letterboxed = Image.new("RGB", new_shape, color)
    img_letterboxed.paste(img_resized, (left, top))

    return img_letterboxed, r, (dw / 2, dh / 2)


def filter_scores_and_topk(scores, score_thr, topk, results=None):
    """Filter results using a score threshold and keep the top-k candidates.

    Args:
        scores (Tensor): scores, shape (num_bboxes, K).
        score_thr (float): score filter threshold.
        topk (int): the number of topk candidates.
        results (dict | list | Tensor, optional): extra tensors to filter.

    Returns:
        tuple: (scores, labels, keep_idxs, filtered_results)
    """
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)

    num_topk = min(topk, valid_idxs.size(0))
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
            raise NotImplementedError(
                f'Only supports dict or list or Tensor, but got {type(results)}.')
    return scores, labels, keep_idxs, filtered_results


# --------------------------------------------------------------------------- #
#  Language tower : XLM-RoBERTa                                                #
# --------------------------------------------------------------------------- #
class XLMRobertaLanguageBackbone(nn.Module):

    def __init__(
        self,
        model_name: str,
        ckpt_path: str,
        frozen_modules: Tuple[str, ...] = (),
        dropout: float = 0.0,
        init_cfg=None,
    ) -> None:
        super().__init__()

        self.frozen_modules = frozen_modules
        cfg = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = XLMRobertaModel(cfg)
        self.language_dim = cfg.hidden_size
        if 'large' in model_name:
            self.head = nn.Linear(1024, 768, bias=True)
        else:
            self.head = nn.Linear(768, 768, bias=True)

        # Load text-model weights from the mmdet checkpoint.
        new_state_dict = OrderedDict()
        state_dict = torch.load(ckpt_path, map_location="cpu")
        for k, v in state_dict['state_dict'].items():
            if k.startswith('backbone.text_model.'):
                name = k.split("backbone.text_model.")[-1]
                new_state_dict[name] = v
        msg = self.load_state_dict(new_state_dict, strict=True)
        print(f'[text-encoder] loaded weights from {ckpt_path}: {msg}')

    def forward(self, text: List[str]) -> Tensor:
        tokens = self.tokenizer(text=text, return_tensors="pt", padding=True)
        tokens = tokens.to(device=self.model.device)
        txt_feats = self.model(**tokens)["last_hidden_state"][:, 0]
        txt_feats = self.head(txt_feats)
        return txt_feats


# --------------------------------------------------------------------------- #
#  Vision tower : ConvNeXt backbone                                           #
# --------------------------------------------------------------------------- #
class LayerNorm(nn.Module):
    r"""LayerNorm supporting ``channels_last`` (default) or ``channels_first``.

    channels_last  -> (batch, height, width, channels)
    channels_first -> (batch, channels, height, width)
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
                x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Block(nn.Module):
    r"""ConvNeXt Block.

    DwConv -> Permute to (N, H, W, C); LayerNorm -> Linear -> GELU -> Linear;
    Permute back. Implemented with channels-last linear layers as it is slightly
    faster in PyTorch.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                         requires_grad=True)
            if layer_scale_init_value > 0 else None
        )

    def forward(self, x: torch.Tensor):
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return identity + x


class ConvNeXt(nn.Module):
    r"""ConvNeXt backbone, returns the 4 stage features."""

    DEPTHS = {
        "tiny": [3, 3, 9, 3],
        "base": [3, 3, 27, 3],
        "large": [3, 3, 27, 3],
    }
    DIMS = {
        "tiny": [96, 192, 384, 768],
        "base": [128, 256, 512, 1024],
        "large": [192, 384, 768, 1536],
    }

    def __init__(self, model_name):
        super().__init__()
        if model_name not in self.DEPTHS:
            raise ValueError(f'Unknown ConvNeXt variant: {model_name}')
        depths = self.DEPTHS[model_name]
        dims = self.DIMS[model_name]

        # stem and 3 intermediate downsampling conv layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))

        # 4 feature-resolution stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, 0.0, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[
                Block(dim=dims[i], drop_path=dp_rates[cur + j],
                      layer_scale_init_value=1e-6)
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

    def forward(self, x):
        outputs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outputs.append(x)
        return tuple(outputs)


# --------------------------------------------------------------------------- #
#  Vision tower : CSPRepBiFPAN neck                                           #
# --------------------------------------------------------------------------- #
_ACTIVATIONS = {
    'relu': nn.ReLU(),
    'silu': nn.SiLU(),
    'hardswish': nn.Hardswish(),
}


class ConvModule_torch(nn.Module):
    """Conv + BN + Activation."""

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 activation_type, padding=None, groups=1, bias=False):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation_type = activation_type
        if activation_type is not None:
            self.act = _ACTIVATIONS.get(activation_type)

    def forward(self, x):
        if self.activation_type is None:
            return self.bn(self.conv(x))
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        if self.activation_type is None:
            return self.conv(x)
        return self.act(self.conv(x))


class ConvBNReLU(nn.Module):
    """Conv + BN + ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=None, groups=1, bias=False):
        super().__init__()
        self.block = ConvModule_torch(in_channels, out_channels, kernel_size,
                                      stride, 'relu', padding, groups, bias)

    def forward(self, x):
        return self.block(x)


class ConvBNSiLU(nn.Module):
    """Conv + BN + SiLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=None, groups=1, bias=False):
        super().__init__()
        self.block = ConvModule_torch(in_channels, out_channels, kernel_size,
                                      stride, 'silu', padding, groups, bias)

    def forward(self, x):
        return self.block(x)


class BottleRep(nn.Module):

    def __init__(self, in_channels, out_channels, basic_block, weight=False):
        super().__init__()
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        self.shortcut = in_channels == out_channels
        self.alpha = Parameter(torch.ones(1)) if weight else 1.0

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs


class RepBlock(nn.Module):
    """A stage block with rep-style basic blocks."""

    def __init__(self, in_channels, out_channels, block, basic_block, n=1):
        super().__init__()
        self.conv1 = BottleRep(in_channels, out_channels,
                               basic_block=basic_block, weight=True)
        n = n // 2
        self.block = nn.Sequential(*(
            BottleRep(out_channels, out_channels, basic_block=basic_block,
                      weight=True) for _ in range(n - 1))) if n > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class BepC3(nn.Module):
    """CSPStackRep Block."""

    def __init__(self, in_channels, out_channels, n=1, e=0.5):
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = ConvBNSiLU(in_channels, c_, 1, 1)
        self.cv2 = ConvBNSiLU(in_channels, c_, 1, 1)
        self.cv3 = ConvBNSiLU(2 * c_, out_channels, 1, 1)
        self.m = RepBlock(in_channels=c_, out_channels=c_, n=n,
                          block=BottleRep, basic_block=ConvBNSiLU)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class Transpose(nn.Module):
    """Normal Transpose, default for upsampling."""

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.upsample_transpose = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, bias=True)

    def forward(self, x):
        return self.upsample_transpose(x)


class BiFusion(nn.Module):
    """BiFusion Block in PAN."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cv1 = ConvBNReLU(in_channels[0], out_channels, 1, 1)
        self.cv2 = ConvBNReLU(in_channels[1], out_channels, 1, 1)
        self.cv3 = ConvBNReLU(out_channels * 3, out_channels, 1, 1)
        self.upsample = Transpose(in_channels=out_channels,
                                  out_channels=out_channels)
        self.downsample = ConvBNReLU(in_channels=out_channels,
                                     out_channels=out_channels,
                                     kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.upsample(x[0])
        x1 = self.cv1(x[1])
        x2 = self.downsample(self.cv2(x[2]))
        return self.cv3(torch.cat((x0, x1, x2), dim=1))


class CSPRepBiFPANNeck(nn.Module):
    """CSPRepBiFPANNeck module."""

    def __init__(self, model_size, scale_factor=0.75):
        super().__init__()
        channels_list = [64, 128, 256, 512, 1024, 256, 128, 128, 256, 256, 512]
        if model_size in ('small', 'base', 'large'):
            num_repeats = [1, 6, 12, 18, 6, 12, 12, 12, 12]
        elif model_size == 'tiny':
            num_repeats = [1, 6, 12, 18, 6, 6, 6, 6, 6]
        else:
            raise ValueError(f'Unknown neck variant: {model_size}')
        csp_e = 0.5
        stage_block = BepC3

        def c(i):  # scaled channel helper
            return int(channels_list[i] * scale_factor)

        self.reduce_layer0 = ConvBNReLU(c(4), c(5), kernel_size=1, stride=1)
        self.Bifusion0 = BiFusion(in_channels=[c(3), c(2)], out_channels=c(5))
        self.Rep_p4 = stage_block(c(5), c(5), n=num_repeats[5], e=csp_e)

        self.reduce_layer1 = ConvBNReLU(c(5), c(6), kernel_size=1, stride=1)
        self.Bifusion1 = BiFusion(in_channels=[c(2), c(1)], out_channels=c(6))
        self.Rep_p3 = stage_block(c(6), c(6), n=num_repeats[6], e=csp_e)

        self.downsample2 = ConvBNReLU(c(6), c(7), kernel_size=3, stride=2)
        self.Rep_n3 = stage_block(c(6) + c(7), c(8), n=num_repeats[7], e=csp_e)

        self.downsample1 = ConvBNReLU(c(8), c(9), kernel_size=3, stride=2)
        self.Rep_n4 = stage_block(c(5) + c(9), c(10), n=num_repeats[8], e=csp_e)

    def forward(self, inputs):
        (x3, x2, x1, x0) = inputs

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

        return [pan_out2, pan_out1, pan_out0]


# --------------------------------------------------------------------------- #
#  Detection head                                                             #
# --------------------------------------------------------------------------- #
class BNContrastiveHead(nn.Module):
    """Batch-Norm contrastive head for YOLO-World.

    Uses batch norm instead of L2-normalization on the image side.
    """

    def __init__(self, embed_dims: int, use_einsum: bool = True) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims, momentum=0.03, eps=0.001)
        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))  # -1.0 is stable
        self.use_einsum = use_einsum

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        if self.use_einsum:
            x = torch.einsum('bchw,bkc->bkhw', x, w)
        else:
            batch, channel, height, width = x.shape
            _, k, _ = w.shape
            x = x.permute(0, 2, 3, 1).reshape(batch, -1, channel)
            w = w.permute(0, 2, 1)
            x = torch.matmul(x, w).reshape(batch, height, width, k)
            x = x.permute(0, 3, 1, 2)
        return x * self.logit_scale.exp() + self.bias


class YOLOWorldHeadModule(nn.Module):
    """Head module for YOLO-World."""

    def __init__(self, embed_dims: int, in_channels: List[int],
                 use_bn_head: bool = False, use_einsum: bool = True,
                 freeze_all: bool = False) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.use_bn_head = use_bn_head
        self.use_einsum = use_einsum
        self.freeze_all = freeze_all
        self.reg_max = 16
        self.in_channels = in_channels
        self._init_layers()

    def _init_layers(self) -> None:
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.cls_contrasts = nn.ModuleList()
        cls_out_channels = 256
        self.featmap_strides = [8, 16, 32]
        self.num_levels = len(self.in_channels)
        reg_out_channels = max(16, self.in_channels[0] // 4, self.reg_max * 4)

        for i in range(self.num_levels):
            self.reg_preds.append(nn.Sequential(
                nn.Conv2d(self.in_channels[i], reg_out_channels, 3, 1, 1,
                          bias=False),
                nn.BatchNorm2d(reg_out_channels, momentum=0.03, eps=0.001),
                nn.SiLU(),
                nn.Conv2d(reg_out_channels, reg_out_channels, 3, 1, 1,
                          bias=False),
                nn.BatchNorm2d(reg_out_channels, momentum=0.03, eps=0.001),
                nn.SiLU(),
                nn.Conv2d(reg_out_channels, 4 * self.reg_max, 1)))
            self.cls_preds.append(nn.Sequential(
                nn.Conv2d(self.in_channels[i], cls_out_channels, 3, 1, 1,
                          bias=False),
                nn.BatchNorm2d(cls_out_channels, momentum=0.03, eps=0.001),
                nn.SiLU(),
                nn.Conv2d(cls_out_channels, cls_out_channels, 3, 1, 1,
                          bias=False),
                nn.BatchNorm2d(cls_out_channels, momentum=0.03, eps=0.001),
                nn.SiLU(),
                nn.Conv2d(cls_out_channels, self.embed_dims, 1)))
            self.cls_contrasts.append(
                BNContrastiveHead(self.embed_dims, use_einsum=self.use_einsum))

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)


# --------------------------------------------------------------------------- #
#  Prior generator & box decoding                                             #
# --------------------------------------------------------------------------- #
class MlvlPointGenerator:
    """Standard points generator for multi-level 2D points-based detectors."""

    def __init__(self, strides: Union[List[int], List[Tuple[int, int]]],
                 offset: float = 0.5) -> None:
        self.strides = [_pair(stride) for stride in strides]
        self.offset = offset

    @property
    def num_levels(self) -> int:
        return len(self.strides)

    def _meshgrid(self, x: Tensor, y: Tensor,
                  row_major: bool = True) -> Tuple[Tensor, Tensor]:
        yy, xx = torch.meshgrid(y, x)
        if row_major:
            return xx.reshape(-1), yy.reshape(-1)
        return yy.reshape(-1), xx.reshape(-1)

    def grid_priors(self, featmap_sizes: List[Tuple],
                    dtype: torch.dtype = torch.float32, device='cuda',
                    with_stride: bool = False) -> List[Tensor]:
        assert self.num_levels == len(featmap_sizes)
        return [
            self.single_level_grid_priors(
                featmap_sizes[i], level_idx=i, dtype=dtype, device=device,
                with_stride=with_stride)
            for i in range(self.num_levels)
        ]

    def single_level_grid_priors(self, featmap_size: Tuple[int], level_idx: int,
                                 dtype: torch.dtype = torch.float32,
                                 device='cuda',
                                 with_stride: bool = False) -> Tensor:
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = ((torch.arange(0, feat_w, device=device) + self.offset)
                   * stride_w).to(dtype)
        shift_y = ((torch.arange(0, feat_h, device=device) + self.offset)
                   * stride_h).to(dtype)
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        if not with_stride:
            shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        else:
            stride_w = shift_xx.new_full((shift_xx.shape[0],), stride_w).to(dtype)
            stride_h = shift_xx.new_full((shift_yy.shape[0],), stride_h).to(dtype)
            shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h], dim=-1)
        return shifts.to(device)


def distance2bbox(points: Tensor, distance: Tensor, max_shape=None) -> Tensor:
    """Decode distance prediction (l, t, r, b) to bounding box (xyxy)."""
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    bboxes = torch.stack([x1, y1, x2, y2], -1)

    if max_shape is not None:
        if bboxes.dim() == 2 and not torch.onnx.is_in_onnx_export():
            bboxes[:, 0::2].clamp_(min=0, max=max_shape[1])
            bboxes[:, 1::2].clamp_(min=0, max=max_shape[0])
            return bboxes
        if not isinstance(max_shape, torch.Tensor):
            max_shape = x1.new_tensor(max_shape)
        max_shape = max_shape[..., :2].type_as(x1)
        min_xy = x1.new_tensor(0)
        max_xy = torch.cat([max_shape, max_shape], dim=-1).flip(-1).unsqueeze(-2)
        bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
        bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)
    return bboxes


# --------------------------------------------------------------------------- #
#  Full detector (vision tower + decode + NMS)                                #
# --------------------------------------------------------------------------- #
class SimpleYOLOWorldDetector(nn.Module):
    """Plain-PyTorch implementation of the WeDetect vision tower."""

    _VARIANTS = {
        'tiny': dict(scale_factor=0.75, in_channels=[96, 192, 384],
                     prompt_dim=768, img_size=(640, 640)),
        'base': dict(scale_factor=1.0, in_channels=[128, 256, 512],
                     prompt_dim=768, img_size=(640, 640)),
        'large': dict(scale_factor=1.5, in_channels=[192, 384, 768],
                      prompt_dim=768, img_size=(1280, 1280)),
    }

    def __init__(self, backbone_size, score_thr=0.005, nms_iou=0.7,
                 pre_nms_topk=30000, post_nms_topk=300) -> None:
        super().__init__()
        if backbone_size not in self._VARIANTS:
            raise ValueError(f'Unknown variant: {backbone_size}')
        v = self._VARIANTS[backbone_size]
        self.img_size = v['img_size']
        self.score_thr = score_thr
        self.nms_iou = nms_iou
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk

        self.backbone = ConvNeXt(backbone_size)
        self.neck = CSPRepBiFPANNeck(backbone_size, v['scale_factor'])
        self.bbox_head = YOLOWorldHeadModule(
            embed_dims=v['prompt_dim'], in_channels=v['in_channels'],
            use_bn_head=True, use_einsum=True)
        self.prior_generator = MlvlPointGenerator(strides=[8, 16, 32],
                                                  offset=0.5)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, image_paths: List[str], embeddings, rescale=True):
        device = self.device
        inputs, ratios, offsets, ori_shapes = [], [], [], []
        for image_path in image_paths:
            img = Image.open(image_path).convert("RGB")
            width, height = img.size
            ori_shapes.append((height, width))
            img, ratio, offset = letterbox(img, self.img_size)
            img = torch.tensor(np.array(img)).permute(2, 0, 1) / 255.0  # CHW RGB
            inputs.append(img)
            ratios.append(ratio)
            offsets.append(offset)
        inputs = torch.stack(inputs, dim=0).to(device)

        img_feats = self.backbone(inputs)
        img_feats = self.neck(img_feats)
        results = self.head_predict(img_feats, embeddings)

        for i in range(len(results)):
            results[i]['bboxes'] -= results[i]['bboxes'].new_tensor([
                offsets[i][0], offsets[i][1], offsets[i][0], offsets[i][1]])
            if rescale:
                results[i]['bboxes'] /= ratios[i]
            results[i]['bboxes'][:, 0::2].clamp_(0, ori_shapes[i][1])
            results[i]['bboxes'][:, 1::2].clamp_(0, ori_shapes[i][0])
        return results

    def head_module_forward_single(self, img_feat, cls_pred, reg_pred,
                                    cls_contrast, embeddings):
        module = self.bbox_head
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_embed = cls_contrast.norm(cls_embed)
        cls_logits = torch.einsum('bchw,kc->bkhw', cls_embed, embeddings)
        cls_logits = cls_logits * cls_contrast.logit_scale.exp() + cls_contrast.bias

        bbox_dist_preds = reg_pred(img_feat)
        if module.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, module.reg_max, h * w]).permute(0, 3, 1, 2)
            bbox_preds = (bbox_dist_preds.softmax(3)
                          .matmul(module.proj.view([-1, 1])).squeeze(-1))
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        return cls_embed, bbox_preds, cls_logits

    def head_predict(self, img_feats, embeddings):
        bbox_embed, bbox_preds, cls_scores = [], [], []
        for i in range(len(img_feats)):
            box_embed, bbox_pred, cls_score = self.head_module_forward_single(
                img_feats[i], self.bbox_head.cls_preds[i],
                self.bbox_head.reg_preds[i], self.bbox_head.cls_contrasts[i],
                embeddings)
            bbox_embed.append(box_embed)
            bbox_preds.append(bbox_pred)
            cls_scores.append(cls_score)

        txt_channel = bbox_embed[0].shape[1]
        num_imgs = bbox_embed[0].shape[0]
        featmap_sizes = [x.shape[2:] for x in bbox_preds]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, dtype=bbox_embed[0].dtype, device=bbox_embed[0].device)
        flatten_priors = torch.cat(mlvl_priors)
        mlvl_strides = [
            flatten_priors.new_full((featmap_size.numel(),), stride)
            for featmap_size, stride in zip(featmap_sizes,
                                            self.bbox_head.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        flatten_bbox_embed = torch.cat([
            x.permute(0, 2, 3, 1).reshape(num_imgs, -1, txt_channel)
            for x in bbox_embed], dim=1)
        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  cls_scores[0].shape[1])
            for cls_score in cls_scores], dim=1).sigmoid()
        flatten_bbox_preds = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds], dim=1)
        flatten_bbox_preds = flatten_bbox_preds * flatten_stride[None, :, None]
        flatten_decoded_bbox = distance2bbox(flatten_priors[None],
                                             flatten_bbox_preds)

        results_list = []
        for bbox, embed, scores in zip(flatten_decoded_bbox, flatten_bbox_embed,
                                       flatten_cls_scores):
            scores, labels, keep_idxs, _ = filter_scores_and_topk(
                scores, self.score_thr, self.pre_nms_topk)
            bbox = bbox[keep_idxs]
            embed = embed[keep_idxs]
            idx = torchvision.ops.batched_nms(
                bbox.float(), scores.float(), labels,
                self.nms_iou)[:self.post_nms_topk]
            results_list.append({
                'bboxes': bbox[idx],
                'embeddings': embed[idx],
                'scores': scores[idx],
                'labels': labels[idx],
            })
        return results_list


# --------------------------------------------------------------------------- #
#  COCO category names (Chinese) & contiguous-id -> COCO-id mapping            #
# --------------------------------------------------------------------------- #
COCO_CLASSES_ZH = [
    "人", "自行车", "汽车", "摩托车", "飞机", "公共汽车", "火车", "卡车", "船",
    "交通灯", "消防栓", "停车标志", "停车计费表", "长凳", "鸟", "猫", "狗", "马",
    "羊", "牛", "大象", "熊", "斑马", "长颈鹿", "背包", "雨伞", "手提包", "领带",
    "手提箱", "飞盘", "滑雪板", "滑雪板", "运动球", "风筝", "棒球棒", "棒球手套",
    "滑板", "冲浪板", "网球拍", "瓶子", "酒杯", "杯子", "叉子", "刀", "勺子",
    "碗", "香蕉", "苹果", "三明治", "橙子", "西兰花", "胡萝卜", "热狗", "披萨",
    "甜甜圈", "蛋糕", "椅子", "沙发", "盆栽植物", "床", "餐桌", "厕所",
    "电视显示器", "笔记本电脑", "鼠标", "遥控器", "键盘", "手机", "微波炉",
    "烤箱", "烤面包机", "水槽", "冰箱", "书", "时钟", "花瓶", "剪刀",
    "泰迪熊小熊", "吹风机", "牙刷",
]

# contiguous label index (0..79) -> COCO category id
COCO_ID_MAP = {
    0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11,
    11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21,
    20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33,
    29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42,
    38: 43, 39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52,
    47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61,
    56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74,
    65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84,
    74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90,
}


# --------------------------------------------------------------------------- #
#  Checkpoint key remapping (mmdet -> plain PyTorch modules)                   #
# --------------------------------------------------------------------------- #
def load_vision_checkpoint(model: SimpleYOLOWorldDetector, model_path: str):
    """Remap an mmdet checkpoint and load it into ``SimpleYOLOWorldDetector``."""
    checkpoint = torch.load(model_path, map_location='cpu')['state_dict']

    # backbone (image model)
    for key in list(checkpoint.keys()):
        if 'backbone' in key:
            new_key = key.replace('backbone.image_model.model.', 'backbone.')
            checkpoint[new_key] = checkpoint.pop(key)
    # drop the language model weights (handled by XLMRobertaLanguageBackbone)
    for key in list(checkpoint.keys()):
        if key.startswith('backbone.text_model.'):
            checkpoint.pop(key)
    # detection head
    for key in list(checkpoint.keys()):
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
    print(f'[detector] loaded weights from {model_path}: {msg}')
    return model


# --------------------------------------------------------------------------- #
#  Evaluation                                                                  #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_coco(model, text_embeddings, coco_ann, coco_img):
    coco_gt = COCO(coco_ann)
    img_ids = coco_gt.getImgIds()

    results = []
    for img_id in tqdm.tqdm(img_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = f"{coco_img.rstrip('/')}/{img_info['file_name']}"
        outputs = model([img_path], text_embeddings)

        boxes = outputs[0]['bboxes']
        scores = outputs[0]['scores']
        labels = outputs[0]['labels']
        for box, score, label in zip(boxes, scores, labels):
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            results.append({
                "image_id": img_id,
                "category_id": COCO_ID_MAP[label.item()],
                "bbox": [float(xmin), float(ymin),
                         float(xmax - xmin), float(ymax - ymin)],
                "score": float(score.max().item()),
            })

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def parse_args():
    parser = argparse.ArgumentParser(
        description='Standalone PyTorch COCO evaluation for WeDetect.')
    parser.add_argument('--variant', choices=['tiny', 'base', 'large'],
                        default='base', help='model variant')
    parser.add_argument('--language-model', required=True,
                        help='path/name of the XLM-RoBERTa model directory')
    parser.add_argument('--checkpoint', required=True,
                        help='WeDetect mmdet checkpoint (.pth)')
    parser.add_argument('--coco-ann', required=True,
                        help='path to instances_val2017.json')
    parser.add_argument('--coco-img', required=True,
                        help='path to the val2017 image directory')
    parser.add_argument('--device', default='cuda',
                        help='inference device, e.g. "cuda" or "cpu"')
    parser.add_argument('--score-thr', type=float, default=0.001)
    parser.add_argument('--nms-iou', type=float, default=0.7)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Language tower -> per-class text embeddings.
    language_encoder = XLMRobertaLanguageBackbone(
        args.language_model, args.checkpoint).to(device)
    with torch.no_grad():
        text_embeddings = language_encoder(COCO_CLASSES_ZH)
    text_embeddings = F.normalize(text_embeddings, dim=-1).to(device).squeeze()

    # Vision tower.
    model = SimpleYOLOWorldDetector(args.variant, score_thr=args.score_thr,
                                    nms_iou=args.nms_iou)
    load_vision_checkpoint(model, args.checkpoint)
    model = model.to(device).eval()

    evaluate_coco(model, text_embeddings, args.coco_ann, args.coco_img)


if __name__ == '__main__':
    main()
