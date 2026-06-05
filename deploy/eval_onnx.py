# Copyright (c) Tencent Inc. All rights reserved.
"""Validate the exported WeDetect ONNX model(s) on COCO ``instances_val2017``.

This script is the ONNX counterpart of ``test_coco_pytorch.py``. It runs the
network through ONNXRuntime instead of PyTorch but reuses *exactly the same*
pre-/post-processing (letterbox + RGB/255, DFL is already baked into the ONNX
graph, then score filtering + class-aware NMS + letterbox rescale) and the same
Chinese COCO vocabulary / id mapping, so the reported mAP is directly
comparable to ``test_coco_pytorch.py``.

Two layouts are supported (``--mode``), matching ``export_onnx.py``:

  * ``dual``  : two ONNX files. The language tower runs once (the vocabulary is
                fixed) and the vision tower runs per image.
                --vision-onnx ... --language-onnx ...
  * ``whole`` : a single ONNX file; the text tower is recomputed per image.
                --onnx model_whole.onnx

Usage (dual, recommended)
-------------------------
    python deploy/eval_onnx.py --mode dual \
        --language-model xlm-roberta-base \
        --vision-onnx   deploy/onnx_models/wedetect_base_vision.onnx \
        --language-onnx deploy/onnx_models/wedetect_base_language.onnx \
        --coco-ann /path/to/instances_val2017.json \
        --coco-img /path/to/val2017

Usage (whole)
-------------
    python deploy/eval_onnx.py --mode whole \
        --language-model xlm-roberta-base \
        --onnx deploy/onnx_models/wedetect_base_whole.onnx \
        --coco-ann /path/to/instances_val2017.json \
        --coco-img /path/to/val2017
"""
import argparse
import os.path as osp
import sys

import numpy as np
import torch
import torchvision
import tqdm
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import onnxruntime as ort
from transformers import AutoTokenizer

# Reuse the standalone pre-/post-processing and the COCO vocabulary so the
# ONNX evaluation matches ``test_coco_pytorch.py`` exactly.
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from test_coco_pytorch import (  # noqa: E402
    COCO_CLASSES_ZH,
    COCO_ID_MAP,
    filter_scores_and_topk,
    letterbox,
)


# --------------------------------------------------------------------------- #
#  Pre- / post-processing (mirrors test_coco_pytorch)                         #
# --------------------------------------------------------------------------- #
def preprocess_image(img_path, img_size):
    """Load an image, letterbox it and return the ONNX input + recovery info.

    Returns:
        image     : float32 [1, 3, H, W], RGB and divided by 255.
        ratio     : letterbox scale ratio.
        offset    : (dw/2, dh/2) padding to undo before rescaling.
        ori_shape : (height, width) of the original image.
    """
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    img_lb, ratio, offset = letterbox(img, (img_size, img_size))
    arr = np.array(img_lb).astype(np.float32).transpose(2, 0, 1) / 255.0
    return arr[None], ratio, offset, (height, width)


def postprocess(bboxes, scores, ratio, offset, ori_shape,
                score_thr, nms_iou, pre_nms_topk, post_nms_topk):
    """ONNX outputs (letterbox space) -> boxes/scores/labels (original image).

    bboxes : [N, 4] decoded xyxy in the letter-boxed coordinate space.
    scores : [N, K] per-class confidence (already sigmoid).
    """
    bboxes = torch.from_numpy(bboxes).float()
    scores = torch.from_numpy(scores).float()

    scores, labels, keep_idxs, _ = filter_scores_and_topk(
        scores, score_thr, pre_nms_topk)
    bboxes = bboxes[keep_idxs]

    keep = torchvision.ops.batched_nms(
        bboxes.float(), scores.float(), labels, nms_iou)[:post_nms_topk]
    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # undo letterbox padding + resize, then clamp to the original image.
    off_x, off_y = offset
    bboxes -= bboxes.new_tensor([off_x, off_y, off_x, off_y])
    bboxes /= ratio
    ori_h, ori_w = ori_shape
    bboxes[:, 0::2].clamp_(0, ori_w)
    bboxes[:, 1::2].clamp_(0, ori_h)
    return bboxes, scores, labels


# --------------------------------------------------------------------------- #
#  ONNXRuntime helpers                                                        #
# --------------------------------------------------------------------------- #
def make_session(path, device):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
        if device == 'cuda' else ['CPUExecutionProvider']
    return ort.InferenceSession(path, providers=providers)


def tokenise(language_model, class_names):
    tokenizer = AutoTokenizer.from_pretrained(language_model)
    tokens = tokenizer(text=class_names, return_tensors='np', padding=True)
    input_ids = tokens['input_ids'].astype(np.int64)
    attention_mask = tokens['attention_mask'].astype(np.int64)
    return input_ids, attention_mask


# --------------------------------------------------------------------------- #
#  Evaluation                                                                 #
# --------------------------------------------------------------------------- #
def evaluate(args):
    class_names = COCO_CLASSES_ZH
    input_ids, attention_mask = tokenise(args.language_model, class_names)
    print(f'[text] {len(class_names)} classes tokenised, '
          f'seq_len={input_ids.shape[1]}')

    # ---- ONNX sessions --------------------------------------------------- #
    txt_feats = None
    if args.mode == 'whole':
        assert args.onnx, '--onnx is required for mode=whole'
        whole_sess = make_session(args.onnx, args.device)
    else:
        assert args.vision_onnx and args.language_onnx, \
            '--vision-onnx and --language-onnx are required for mode=dual'
        vision_sess = make_session(args.vision_onnx, args.device)
        language_sess = make_session(args.language_onnx, args.device)
        txt_feats = language_sess.run(
            ['txt_feats'],
            {'input_ids': input_ids, 'attention_mask': attention_mask})[0]
        txt_feats = txt_feats.astype(np.float32)
        print(f'[text] txt_feats shape = {txt_feats.shape}')

    # ---- iterate over COCO val ------------------------------------------- #
    coco_gt = COCO(args.coco_ann)
    img_ids = coco_gt.getImgIds()
    if args.num_images > 0:
        img_ids = img_ids[:args.num_images]

    results = []
    for img_id in tqdm.tqdm(img_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = osp.join(args.coco_img, img_info['file_name'])
        image, ratio, offset, ori_shape = preprocess_image(
            img_path, args.img_size)

        if args.mode == 'whole':
            bboxes, scores = whole_sess.run(
                ['bboxes', 'scores'],
                {'image': image,
                 'input_ids': input_ids,
                 'attention_mask': attention_mask})
        else:
            bboxes, scores = vision_sess.run(
                ['bboxes', 'scores'],
                {'image': image, 'txt_feats': txt_feats})

        boxes, sc, labels = postprocess(
            bboxes[0], scores[0], ratio, offset, ori_shape,
            args.score_thr, args.nms_iou, args.pre_nms_topk,
            args.post_nms_topk)

        for box, score, label in zip(boxes, sc, labels):
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            results.append({
                'image_id': img_id,
                'category_id': COCO_ID_MAP[label.item()],
                'bbox': [float(xmin), float(ymin),
                         float(xmax - xmin), float(ymax - ymin)],
                'score': float(score.item()),
            })

    print('\nComputing metrics ...')
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    if args.num_images > 0:
        coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate WeDetect ONNX')
    parser.add_argument('--mode', choices=['whole', 'dual'], default='dual')
    parser.add_argument('--onnx', help='whole-model ONNX (mode=whole)')
    parser.add_argument('--vision-onnx', help='vision ONNX (mode=dual)')
    parser.add_argument('--language-onnx', help='language ONNX (mode=dual)')
    parser.add_argument('--language-model', required=True,
                        help='path/name of the XLM-RoBERTa model directory '
                             '(only the tokenizer is used here)')
    parser.add_argument('--coco-ann', required=True,
                        help='path to instances_val2017.json')
    parser.add_argument('--coco-img', required=True,
                        help='path to the val2017 image directory')
    parser.add_argument('--img-size', type=int, default=640,
                        help='must match the size used at export time')
    parser.add_argument('--num-images', type=int, default=-1,
                        help='limit number of evaluated images (-1 = all)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--score-thr', type=float, default=0.001)
    parser.add_argument('--nms-iou', type=float, default=0.7)
    parser.add_argument('--pre-nms-topk', type=int, default=30000)
    parser.add_argument('--post-nms-topk', type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
