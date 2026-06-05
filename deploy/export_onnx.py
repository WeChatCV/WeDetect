# Copyright (c) Tencent Inc. All rights reserved.
"""Export the standalone WeDetect detector to ONNX.

This script reuses the *mmcv/mmdet-free* model definition in
``test_coco_pytorch.py`` (``SimpleYOLOWorldDetector`` for the vision tower and
``XLMRobertaLanguageBackbone`` for the language tower) so the whole export /
validation pipeline is self-contained.

WeDetect is a *dual-tower* open-vocabulary detector:

    Vision  tower : image -> ConvNeXt backbone -> CSPRepBiFPAN neck
                            -> head (cls_preds / reg_preds)
    Language tower: text  -> XLM-RoBERTa -> linear head -> L2 normalize
    Fusion        : BNContrastiveHead  logit = scale * <norm(cls_embed), norm(txt)> + bias

The two towers only meet at the contrastive head, so two export layouts are
supported (``--mode``):

1. ``whole`` : a single ONNX graph that contains *both* towers.
               inputs : image, input_ids, attention_mask
               outputs: bboxes, scores
2. ``dual``  : two separate ONNX graphs (one per tower).
               vision   : (image, txt_feats)          -> (bboxes, scores)
               language : (input_ids, attention_mask) -> txt_feats

In both layouts the *vision* graph already performs DFL decoding and produces
decoded boxes (xyxy, in the letter-boxed input coordinate space) together with
per-class sigmoid scores. Only NMS / rescale is left to the host code
(see ``eval_onnx.py``).

Tokenisation (raw text -> input_ids / attention_mask) is **not** part of the
ONNX graph; it stays in Python with the HuggingFace tokenizer, exactly as in
``XLMRobertaLanguageBackbone``.

Usage
-----
    # dual towers (recommended)
    python deploy/export_onnx.py \
        --variant base \
        --language-model xlm-roberta-base \
        --checkpoint checkpoints/wedetect_base.pth \
        --mode dual --output-dir deploy/onnx_models

    # single fused graph
    python deploy/export_onnx.py \
        --variant base \
        --language-model xlm-roberta-base \
        --checkpoint checkpoints/wedetect_base.pth \
        --mode whole --output-dir deploy/onnx_models
"""
import argparse
import os
import os.path as osp
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make the standalone model definition importable when the script is launched
# from the project root (``python deploy/export_onnx.py``).
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from test_coco_pytorch import (  # noqa: E402
    COCO_CLASSES_ZH,
    SimpleYOLOWorldDetector,
    XLMRobertaLanguageBackbone,
    load_vision_checkpoint,
)


# --------------------------------------------------------------------------- #
#  Deployment wrappers                                                        #
# --------------------------------------------------------------------------- #
class VisionDeployModel(nn.Module):
    """Vision tower + detection head + DFL decode.

    forward(image, txt_feats) -> (bboxes, scores)
        image     : float32 [B, 3, H, W], already RGB and divided by 255.
        txt_feats : float32 [B, K, embed_dims]  (per-class text embeddings)
        bboxes    : float32 [B, N, 4]   decoded xyxy in input (letterbox) space
        scores    : float32 [B, N, K]   per-class confidence after sigmoid
    """

    def __init__(self, detector: SimpleYOLOWorldDetector, img_size: int = 640):
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

        self.register_buffer(
            'proj',
            torch.arange(self.reg_max, dtype=torch.float).view(-1, 1),
            persistent=False)

        # Pre-compute anchor points / strides for the fixed input size using
        # exactly the same prior generator as eval (test_coco_pytorch).
        featmap_sizes = [(img_size // s, img_size // s)
                         for s in self.strides_cfg]
        mlvl_priors = detector.prior_generator.grid_priors(
            featmap_sizes, dtype=torch.float32, device='cpu', with_stride=True)
        priors = torch.cat(mlvl_priors, dim=0)          # [N, 4] -> x, y, s, s
        self.register_buffer('points', priors[:, :2], persistent=False)
        self.register_buffer('strides', priors[:, 2:3], persistent=False)

    def forward(self, image, txt_feats):
        img_feats = self.backbone(image)
        neck_feats = self.neck(img_feats)

        cls_list, box_list = [], []
        for i in range(self.num_levels):
            feat = neck_feats[i]
            b, _, h, w = feat.shape

            cls_embed = self.cls_preds[i](feat)
            cls_logit = self.cls_contrasts[i](cls_embed, txt_feats)  # b,K,h,w

            reg = self.reg_preds[i](feat)                            # b,4r,h,w
            reg = reg.reshape(-1, 4, self.reg_max, h * w).permute(0, 3, 1, 2)
            reg = reg.softmax(3).matmul(self.proj).squeeze(-1)       # b,hw,4

            cls_logit = cls_logit.permute(0, 2, 3, 1).reshape(
                b, h * w, -1)                                        # b,hw,K
            cls_list.append(cls_logit)
            box_list.append(reg)

        scores = torch.cat(cls_list, dim=1).sigmoid()               # b,N,K
        dist = torch.cat(box_list, dim=1)                           # b,N,4

        # DFL distance (l, t, r, b) -> xyxy
        lt = dist[..., :2] * self.strides
        rb = dist[..., 2:] * self.strides
        x1y1 = self.points - lt
        x2y2 = self.points + rb
        bboxes = torch.cat([x1y1, x2y2], dim=-1)                    # b,N,4
        return bboxes, scores


class LanguageDeployModel(nn.Module):
    """Language tower.

    forward(input_ids, attention_mask) -> txt_feats [1, K, embed_dims]
    """

    def __init__(self, language: XLMRobertaLanguageBackbone):
        super().__init__()
        self.model = language.model      # XLMRobertaModel
        self.head = language.head        # Linear

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)
        feat = outputs['last_hidden_state'][:, 0]   # [K, hidden]
        feat = self.head(feat)                      # [K, embed_dims]
        feat = F.normalize(feat, dim=-1, p=2)
        return feat.unsqueeze(0)                    # [1, K, embed_dims]


class WholeDeployModel(nn.Module):
    """Both towers fused into a single graph.

    forward(image, input_ids, attention_mask) -> (bboxes, scores)
    """

    def __init__(self, detector: SimpleYOLOWorldDetector,
                 language: XLMRobertaLanguageBackbone, img_size: int = 640):
        super().__init__()
        self.language = LanguageDeployModel(language)
        self.vision = VisionDeployModel(detector, img_size=img_size)

    def forward(self, image, input_ids, attention_mask):
        txt_feats = self.language(input_ids, attention_mask)
        return self.vision(image, txt_feats)


# --------------------------------------------------------------------------- #
#  Builders                                                                   #
# --------------------------------------------------------------------------- #
def build_models(variant, language_model, checkpoint, device='cpu'):
    """Build the standalone vision + language towers and load weights."""
    detector = SimpleYOLOWorldDetector(variant)
    load_vision_checkpoint(detector, checkpoint)
    detector = detector.to(device).eval()

    language = XLMRobertaLanguageBackbone(language_model, checkpoint)
    language = language.to(device).eval()
    return detector, language


def make_dummy_text(language: XLMRobertaLanguageBackbone, num_classes_hint=8):
    """Build a small dummy tokenised batch for tracing the language tower."""
    sample = COCO_CLASSES_ZH[:max(2, num_classes_hint)]
    tokens = language.tokenizer(text=sample, return_tensors='pt', padding=True)
    return tokens['input_ids'], tokens['attention_mask']


# --------------------------------------------------------------------------- #
#  Export routines                                                            #
# --------------------------------------------------------------------------- #
def export_vision(detector, embed_dims, img_size, save_path, opset):
    wrapper = VisionDeployModel(detector, img_size=img_size).eval()
    dummy_img = torch.randn(1, 3, img_size, img_size)
    dummy_txt = torch.randn(1, 8, embed_dims)

    torch.onnx.export(
        wrapper, (dummy_img, dummy_txt), save_path,
        input_names=['image', 'txt_feats'],
        output_names=['bboxes', 'scores'],
        dynamic_axes={
            'txt_feats': {1: 'num_classes'},
            'scores': {2: 'num_classes'},
        },
        opset_version=opset,
        do_constant_folding=True)
    print(f'[vision]   exported -> {save_path}')


def export_language(language, save_path, opset):
    wrapper = LanguageDeployModel(language).eval()
    input_ids, attention_mask = make_dummy_text(language)

    torch.onnx.export(
        wrapper, (input_ids, attention_mask), save_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['txt_feats'],
        dynamic_axes={
            'input_ids': {0: 'num_classes', 1: 'seq_len'},
            'attention_mask': {0: 'num_classes', 1: 'seq_len'},
            'txt_feats': {1: 'num_classes'},
        },
        opset_version=opset,
        do_constant_folding=True)
    print(f'[language] exported -> {save_path}')


def export_whole(detector, language, img_size, save_path, opset):
    wrapper = WholeDeployModel(detector, language, img_size=img_size).eval()
    dummy_img = torch.randn(1, 3, img_size, img_size)
    input_ids, attention_mask = make_dummy_text(language)

    torch.onnx.export(
        wrapper, (dummy_img, input_ids, attention_mask), save_path,
        input_names=['image', 'input_ids', 'attention_mask'],
        output_names=['bboxes', 'scores'],
        dynamic_axes={
            'input_ids': {0: 'num_classes', 1: 'seq_len'},
            'attention_mask': {0: 'num_classes', 1: 'seq_len'},
            'scores': {2: 'num_classes'},
        },
        opset_version=opset,
        do_constant_folding=True)
    print(f'[whole]    exported -> {save_path}')


def maybe_simplify(path):
    try:
        import onnx
        from onnxsim import simplify
        model = onnx.load(path)
        model_sim, ok = simplify(model)
        assert ok, 'onnxsim check failed'
        onnx.save(model_sim, path)
        print(f'[simplify] {path}')
    except Exception as e:  # pragma: no cover - optional dependency
        print(f'[simplify] skipped ({e})')


# --------------------------------------------------------------------------- #
#  Entry point                                                                #
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description='Export WeDetect to ONNX')
    parser.add_argument('--variant', choices=['tiny', 'base', 'large'],
                        default='base', help='model variant')
    parser.add_argument('--language-model', required=True,
                        help='path/name of the XLM-RoBERTa model directory')
    parser.add_argument('--checkpoint', required=True,
                        help='WeDetect mmdet checkpoint (.pth)')
    parser.add_argument('--mode', choices=['whole', 'dual'], default='dual',
                        help='"whole": one ONNX with both towers; '
                             '"dual": separate vision/language ONNX files')
    parser.add_argument('--img-size', type=int, default=640,
                        help='square input size for the vision tower')
    parser.add_argument('--output-dir', default='deploy/onnx_models',
                        help='directory to save the exported ONNX model(s)')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset version')
    parser.add_argument('--device', default='cpu',
                        help='device used to load the model for export')
    parser.add_argument('--simplify', action='store_true',
                        help='run onnxsim on the exported model(s)')
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    detector, language = build_models(
        args.variant, args.language_model, args.checkpoint, args.device)
    embed_dims = detector.bbox_head.embed_dims

    tag = f'wedetect_{args.variant}'
    if args.mode == 'whole':
        path = osp.join(args.output_dir, f'{tag}_whole.onnx')
        export_whole(detector, language, args.img_size, path, args.opset)
        if args.simplify:
            maybe_simplify(path)
    else:
        v_path = osp.join(args.output_dir, f'{tag}_vision.onnx')
        l_path = osp.join(args.output_dir, f'{tag}_language.onnx')
        export_vision(detector, embed_dims, args.img_size, v_path, args.opset)
        export_language(language, l_path, args.opset)
        if args.simplify:
            maybe_simplify(v_path)
            maybe_simplify(l_path)

    print('Done.')


if __name__ == '__main__':
    main()
