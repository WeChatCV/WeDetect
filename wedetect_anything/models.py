# Copyright (c) Tencent Inc. All rights reserved.
"""Standalone (mmcv / mmdet-free) model definitions for *WeDetect-Anything*.

WeDetect-Anything reuses the **WeDetect-Uni** detector as a *class-agnostic
proposal generator* to simulate open-world ("detect anything") detection:

    1. The vision tower extracts the top-k class-agnostic foreground boxes,
       each carrying a box ``embedding`` and a ``box``. "Objectness" comes from
       a set of *learnable* prompt embeddings (``self.embeddings``) instead of a
       text vocabulary -- so the proposal stage needs **no text input**.
    2. Every box ``embedding`` is then matched (cosine-similarity + per-level
       scale / bias + sigmoid) against a bank of pre-computed **text
       embeddings** (``prompt_free.npy``), encoded once from a very large
       vocabulary. ``argmax`` over the vocabulary gives the final label.

This module provides:

    * the *vision proposal tower* (``SimpleYOLOWorldProposalDetector``)
      mirroring ``generate_proposal.py``;
    * the *language tower* (``XLMRobertaLanguageBackbone``) used to encode the
      vocabulary into ``prompt_free.npy``;
    * an ONNX-friendly ``VisionProposalDeployModel`` that, given only an image,
      returns ``(scores, bboxes, embeds)``;
    * checkpoint key-remapping helpers.

The layout intentionally follows ``deploy/test_coco_pytorch.py`` so the two
pipelines stay consistent and easy to cross-check.
"""
from collections import OrderedDict
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn.modules.utils import _pair

try:
    from transformers import AutoConfig, AutoTokenizer, XLMRobertaModel
except ImportError:  # transformers is only needed for the language tower
    AutoConfig = AutoTokenizer = XLMRobertaModel = None


# --------------------------------------------------------------------------- #
#  Pre-processing                                                             #
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


# --------------------------------------------------------------------------- #
#  Language tower : XLM-RoBERTa                                               #
# --------------------------------------------------------------------------- #
class XLMRobertaLanguageBackbone(nn.Module):
    """Text encoder used to turn the vocabulary into text embeddings."""

    def __init__(self, model_name: str, ckpt_path: str = "") -> None:
        super().__init__()
        cfg = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = XLMRobertaModel(cfg)
        self.language_dim = cfg.hidden_size
        if 'large' in model_name:
            self.head = nn.Linear(1024, 768, bias=True)
        else:
            self.head = nn.Linear(768, 768, bias=True)

        if ckpt_path:
            self.load_text_weights(ckpt_path)

    def load_text_weights(self, ckpt_path: str) -> None:
        """Load only the ``backbone.text_model.*`` weights from a checkpoint."""
        new_state_dict = OrderedDict()
        state_dict = torch.load(ckpt_path, map_location="cpu")
        state_dict = state_dict.get('state_dict', state_dict)
        for k, v in state_dict.items():
            if k.startswith('backbone.text_model.'):
                new_state_dict[k.split("backbone.text_model.")[-1]] = v
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
    r"""LayerNorm supporting ``channels_last`` (default) or ``channels_first``."""

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
    r"""ConvNeXt Block (channels-last linear implementation)."""

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

    def __init__(self, model_size, scale_factor=1.0):
        super().__init__()
        channels_list = [64, 128, 256, 512, 1024, 256, 128, 128, 256, 256, 512]
        num_repeats = [1, 6, 12, 18, 6, 12, 12, 12, 12]
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
    """Batch-Norm contrastive head for YOLO-World."""

    def __init__(self, embed_dims: int, use_einsum: bool = True) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims, momentum=0.03, eps=0.001)
        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))  # -1.0 is stable
        self.use_einsum = use_einsum


class YOLOWorldHeadModule(nn.Module):
    """Head module for YOLO-World."""

    def __init__(self, embed_dims: int, in_channels: List[int],
                 use_bn_head: bool = True, use_einsum: bool = True) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.use_bn_head = use_bn_head
        self.use_einsum = use_einsum
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
                    dtype: torch.dtype = torch.float32, device='cpu',
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
                                 device='cpu',
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


def distance2bbox(points: Tensor, distance: Tensor) -> Tensor:
    """Decode distance prediction (l, t, r, b) to bounding box (xyxy)."""
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    return torch.stack([x1, y1, x2, y2], -1)


# --------------------------------------------------------------------------- #
#  WeDetect-Uni proposal detector (vision tower + learnable prompts)          #
# --------------------------------------------------------------------------- #
class SimpleYOLOWorldProposalDetector(nn.Module):
    """Class-agnostic proposal generator (mirrors ``generate_proposal.py``).

    Unlike the open-vocabulary detector that takes text embeddings as input,
    this model carries a set of *learnable* prompt embeddings
    (``self.embeddings``) that act as a generic "objectness" prompt bank, so the
    proposal stage needs no text input.
    """

    _VARIANTS = {
        'base': dict(scale_factor=1.0, in_channels=[128, 256, 512],
                     img_size=(640, 640), grid_size=[6400, 1600, 400]),
        'large': dict(scale_factor=1.5, in_channels=[192, 384, 768],
                      img_size=(1280, 1280), grid_size=[25600, 6400, 1600]),
    }

    def __init__(self, backbone_size, prompt_dim=768, num_prompts=256) -> None:
        super().__init__()
        if backbone_size not in self._VARIANTS:
            raise ValueError(f'Unknown variant: {backbone_size}')
        v = self._VARIANTS[backbone_size]
        self.img_size = v['img_size']
        self.grid_size = v['grid_size']
        self.num_prompts = num_prompts

        self.backbone = ConvNeXt(backbone_size)
        self.neck = CSPRepBiFPANNeck(backbone_size, v['scale_factor'])
        self.bbox_head = YOLOWorldHeadModule(
            embed_dims=prompt_dim, in_channels=v['in_channels'],
            use_bn_head=True, use_einsum=True)

        embeddings = F.normalize(torch.randn((num_prompts, prompt_dim)), dim=-1)
        self.embeddings = nn.Parameter(embeddings)
        self.prior_generator = MlvlPointGenerator(strides=[8, 16, 32],
                                                  offset=0.5)

    @property
    def device(self):
        return next(self.parameters()).device


# --------------------------------------------------------------------------- #
#  ONNX deploy wrapper : image -> (scores, bboxes, embeds)                    #
# --------------------------------------------------------------------------- #
class VisionProposalDeployModel(nn.Module):
    """ONNX-friendly wrapper of the proposal vision tower.

    forward(image) -> (scores, bboxes, embeds)
        image  : float32 [B, 3, H, W], RGB and divided by 255.
        scores : float32 [B, N, num_prompts]  objectness, already sigmoid.
        bboxes : float32 [B, N, 4]   decoded xyxy in the letter-boxed space.
        embeds : float32 [B, N, embed_dims]  per-box (BN-normed) embeddings,
                 to be matched against the pre-computed text embeddings.

    For ``N`` with a 640 input: ``80² + 40² + 20² = 8400``.
    """

    def __init__(self, detector: SimpleYOLOWorldProposalDetector,
                 img_size: int = 640):
        super().__init__()
        self.backbone = detector.backbone
        self.neck = detector.neck

        head = detector.bbox_head
        self.cls_preds = head.cls_preds
        self.reg_preds = head.reg_preds
        self.cls_contrasts = head.cls_contrasts
        self.reg_max = head.reg_max
        self.num_levels = head.num_levels
        self.strides_cfg = list(head.featmap_strides)
        self.register_buffer('embeddings', detector.embeddings.data.clone(),
                             persistent=False)

        self.register_buffer(
            'proj',
            torch.arange(self.reg_max, dtype=torch.float).view(-1, 1),
            persistent=False)

        # Pre-compute anchor points / strides for the fixed input size.
        featmap_sizes = [(img_size // s, img_size // s)
                         for s in self.strides_cfg]
        mlvl_priors = detector.prior_generator.grid_priors(
            featmap_sizes, dtype=torch.float32, device='cpu', with_stride=True)
        priors = torch.cat(mlvl_priors, dim=0)          # [N, 4] -> x, y, s, s
        self.register_buffer('points', priors[:, :2], persistent=False)
        self.register_buffer('strides', priors[:, 2:3], persistent=False)

    def forward(self, image):
        img_feats = self.backbone(image)
        neck_feats = self.neck(img_feats)

        cls_list, box_list, embed_list = [], [], []
        for i in range(self.num_levels):
            feat = neck_feats[i]
            b, _, h, w = feat.shape

            contrast = self.cls_contrasts[i]
            cls_embed = contrast.norm(self.cls_preds[i](feat))   # b,C,h,w
            cls_logit = torch.einsum('bchw,kc->bkhw', cls_embed,
                                     self.embeddings)            # b,K,h,w
            cls_logit = cls_logit * contrast.logit_scale.exp() + contrast.bias

            reg = self.reg_preds[i](feat)                        # b,4r,h,w
            reg = reg.reshape(-1, 4, self.reg_max, h * w).permute(0, 3, 1, 2)
            reg = reg.softmax(3).matmul(self.proj).squeeze(-1)   # b,hw,4

            cls_logit = cls_logit.permute(0, 2, 3, 1).reshape(b, h * w, -1)
            cls_embed = cls_embed.permute(0, 2, 3, 1).reshape(b, h * w, -1)
            cls_list.append(cls_logit)
            box_list.append(reg)
            embed_list.append(cls_embed)

        scores = torch.cat(cls_list, dim=1).sigmoid()            # b,N,K
        embeds = torch.cat(embed_list, dim=1)                    # b,N,C
        dist = torch.cat(box_list, dim=1)                        # b,N,4

        # DFL distance (l, t, r, b) -> xyxy
        lt = dist[..., :2] * self.strides
        rb = dist[..., 2:] * self.strides
        x1y1 = self.points - lt
        x2y2 = self.points + rb
        bboxes = torch.cat([x1y1, x2y2], dim=-1)                 # b,N,4
        return scores, bboxes, embeds


# --------------------------------------------------------------------------- #
#  Checkpoint key remapping (mmdet -> plain PyTorch modules)                   #
# --------------------------------------------------------------------------- #
def load_proposal_checkpoint(model: SimpleYOLOWorldProposalDetector,
                             model_path: str):
    """Remap an mmdet checkpoint and load it into the proposal detector.

    Mirrors the key-remapping in ``generate_proposal.py``: the backbone image
    model and the detection head are renamed; the learnable prompt embeddings
    (``embeddings``) and other matching keys are loaded with ``strict=False``.
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    checkpoint = checkpoint.get('state_dict', checkpoint)

    # backbone (image model)
    for key in list(checkpoint.keys()):
        if 'backbone' in key:
            new_key = key.replace('backbone.image_model.model.', 'backbone.')
            checkpoint[new_key] = checkpoint.pop(key)
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


def get_head_scale_bias(model: SimpleYOLOWorldProposalDetector):
    """Return the per-level ``(logit_scale.exp(), bias)`` of the contrastive
    heads. These exact constants are reused at inference time to score box
    embeddings against the text embeddings (``scale * <txt, emb> + bias``)."""
    scales, biases = [], []
    for contrast in model.bbox_head.cls_contrasts:
        scales.append(float(contrast.logit_scale.exp().item()))
        biases.append(float(contrast.bias.item()))
    return scales, biases
