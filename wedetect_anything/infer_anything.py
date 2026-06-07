# Copyright (c) Tencent Inc. All rights reserved.
"""WeDetect-Anything ONNX inference (proposal -> text matching -> visualize).

Pipeline:

    1. Proposal ONNX (exported by ``export_onnx.py``) turns an image into
       ``scores`` (objectness), ``bboxes`` and per-box ``embeds``.
    2. Class-agnostic NMS on the objectness selects the top-k proposals.
    3. Each kept box embedding is matched against the pre-computed text
       embeddings (``prompt_free.npy``):
           logit = scale_lvl * <txt_feat, box_embed> + bias_lvl
       ``sigmoid`` + ``argmax`` over the vocabulary gives the label / score.
    4. Boxes above ``--match-thr`` are rescaled to the original image and drawn.

Usage
-----
    python wedetect_anything/infer_anything.py \
        --onnx wedetect_anything/onnx_models/wedetect_anything_base.onnx \
        --meta wedetect_anything/onnx_models/wedetect_anything_base_meta.json \
        --text-feat wedetect_anything/prompt_free.npy \
        --image path/to/image.jpg \
        --output pred.jpg
"""
import argparse
import json
import os.path as osp
import sys

import cv2
import numpy as np
import onnxruntime
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import batched_nms

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from prompt_free_vocab import prompt_free_name  # noqa: E402

# Default per-level constants (used only if no meta.json is provided). These
# match the reference WeDetect-Uni base checkpoint.
_DEFAULT_META = {
    'grid_size': [6400, 1600, 400],
    'logit_scale_exp': [0.060884383, 0.19966304, 0.27937073],
    'bias': [-12.315344, -10.9335, -9.573644],
}


# --------------------------------------------------------------------------- #
#  Pre-processing                                                             #
# --------------------------------------------------------------------------- #
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), scaleup=True):
    """Letterbox a BGR/RGB numpy image to a square shape (no auto-stride)."""
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def rescale(input_shape, boxes, ori_shape):
    """Map boxes from letterbox input space back to the original image."""
    ratio = min(input_shape[0] / ori_shape[0], input_shape[1] / ori_shape[1])
    pad = ((input_shape[1] - ori_shape[1] * ratio) / 2,
           (input_shape[0] - ori_shape[0] * ratio) / 2)
    boxes[[0, 2]] -= pad[0]
    boxes[[1, 3]] -= pad[1]
    boxes[:4] /= ratio
    boxes[0] = np.clip(boxes[0], 0, ori_shape[1])
    boxes[1] = np.clip(boxes[1], 0, ori_shape[0])
    boxes[2] = np.clip(boxes[2], 0, ori_shape[1])
    boxes[3] = np.clip(boxes[3], 0, ori_shape[0])
    return boxes


def np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------- #
#  Class-agnostic NMS on objectness                                           #
# --------------------------------------------------------------------------- #
def proposal_nms(scores, bboxes, score_thr, iou_thr, keep_top_k):
    """Select the top-k proposals by max-over-prompts objectness.

    Args:
        scores : [N, num_prompts] objectness (already sigmoid).
        bboxes : [N, 4] xyxy (letterbox space).
    Returns:
        keep_idxs : LongTensor of selected box indices into [0, N).
        obj       : objectness score of each kept box.
    """
    scores = torch.from_numpy(scores).float()
    bboxes = torch.from_numpy(bboxes).float()

    obj, _ = scores.max(dim=1)             # objectness per box
    valid = obj > score_thr
    if not torch.any(valid):
        return torch.empty(0, dtype=torch.long), torch.empty(0)
    idxs = torch.nonzero(valid).squeeze(1)
    keep = batched_nms(bboxes[idxs].float(), obj[idxs],
                       torch.zeros_like(idxs), iou_thr)[:keep_top_k]
    keep_idxs = idxs[keep]
    return keep_idxs, obj[keep_idxs]


# --------------------------------------------------------------------------- #
#  Inference                                                                  #
# --------------------------------------------------------------------------- #
def load_meta(meta_path):
    if meta_path and osp.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    print('[meta] not provided, falling back to default base constants')
    return dict(_DEFAULT_META)


def build_level_constants(meta):
    """Broadcast per-level scale / bias to a per-box vector of length N."""
    grid = meta['grid_size']
    scale = meta['logit_scale_exp']
    bias = meta['bias']
    scales = np.concatenate([np.full(grid[i], scale[i], np.float32)
                             for i in range(len(grid))])
    biases = np.concatenate([np.full(grid[i], bias[i], np.float32)
                             for i in range(len(grid))])
    return scales, biases


def run(args):
    session = onnxruntime.InferenceSession(
        args.onnx, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        if args.device == 'cuda' else ['CPUExecutionProvider'])
    meta = load_meta(args.meta)
    box_scales, box_biases = build_level_constants(meta)
    text_feat = np.load(args.text_feat).astype(np.float32)
    vocab = list(prompt_free_name)
    print(f'[init] vocab={len(vocab)}, text_feat={text_feat.shape}')

    font = ImageFont.truetype(args.font, 24) if osp.exists(args.font) \
        else ImageFont.load_default()

    # ---- preprocess ------------------------------------------------------ #
    img_pil = Image.open(args.image).convert('RGB')
    origin = np.asarray(img_pil)                         # RGB
    ori_h, ori_w = origin.shape[:2]
    img_lb, _, _ = letterbox(origin.copy(), (args.img_size, args.img_size))
    inp = (img_lb / 255.0).transpose(2, 0, 1)[None].astype(np.float32)
    inp = np.ascontiguousarray(inp)

    # ---- proposal -------------------------------------------------------- #
    scores, bboxes, embeds = session.run(
        ['scores', 'bboxes', 'embeds'], {'input_image': inp})
    scores, bboxes, embeds = scores[0], bboxes[0], embeds[0]

    keep_idxs, _ = proposal_nms(
        scores, bboxes, args.score_thr, args.iou_thr, args.keep_top_k)
    if keep_idxs.numel() == 0:
        print('[result] no proposal kept')
        img_pil.save(args.output)
        return

    kept_boxes = bboxes[keep_idxs.numpy()]
    kept_embeds = embeds[keep_idxs.numpy()]
    kept_scales = box_scales[keep_idxs.numpy()]
    kept_biases = box_biases[keep_idxs.numpy()]

    # ---- text matching + draw ------------------------------------------- #
    draw = ImageDraw.Draw(img_pil)
    rng = np.random.default_rng(0)
    detected = []
    for i in range(len(keep_idxs)):
        # cosine-similarity (text_feat is L2-normalized) -> per-level scale/bias
        sim = text_feat @ kept_embeds[i]
        logit = sim * kept_scales[i] + kept_biases[i]
        prob = np_sigmoid(logit)
        cls = int(np.argmax(prob))
        conf = float(prob[cls])
        if conf < args.match_thr:
            continue

        box = rescale((args.img_size, args.img_size),
                      kept_boxes[i].copy(), (ori_h, ori_w)).round()
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        label = f'{vocab[cls]} {conf:.2f}'
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        tw = len(label) * 16
        draw.rectangle([x1, y1, x1 + tw, y1 + 26], fill=color)
        draw.text((x1 + 3, y1 + 1), label, font=font, fill='white')
        detected.append(f'{vocab[cls]}({conf:.2f})')

    img_pil.save(args.output)
    print(f'[result] {len(keep_idxs)} proposals, {len(detected)} labeled -> '
          f'{args.output}')
    print('         ' + ', '.join(detected) if detected else '         (none)')


def parse_args():
    parser = argparse.ArgumentParser(description='WeDetect-Anything inference')
    parser.add_argument('--onnx', required=True, help='proposal ONNX model')
    parser.add_argument('--meta', default='',
                        help='meta json from export_onnx (scale/bias/grid)')
    parser.add_argument('--text-feat', default='prompt_free.npy',
                        help='pre-computed text embeddings (.npy)')
    parser.add_argument('--image', required=True, help='input image path')
    parser.add_argument('--output', default='pred.jpg', help='output image path')
    parser.add_argument('--font', default='simsun.ttc',
                        help='TTF font for Chinese labels')
    parser.add_argument('--img-size', type=int, default=640,
                        help='must match the export size')
    parser.add_argument('--score-thr', type=float, default=0.1,
                        help='objectness threshold before NMS')
    parser.add_argument('--iou-thr', type=float, default=0.7,
                        help='NMS IoU threshold')
    parser.add_argument('--keep-top-k', type=int, default=100,
                        help='max proposals kept after NMS')
    parser.add_argument('--match-thr', type=float, default=0.4,
                        help='text-matching sigmoid threshold to keep a label')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
