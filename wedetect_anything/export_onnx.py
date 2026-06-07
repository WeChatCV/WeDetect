# Copyright (c) Tencent Inc. All rights reserved.
"""Export the WeDetect-Uni *proposal* detector to ONNX for WeDetect-Anything.

The proposal stage is purely visual: a set of learnable prompt embeddings acts
as a generic "objectness" prompt bank, so the exported graph takes **only an
image** and produces the class-agnostic proposals together with their box
embeddings:

    forward(image) -> (scores, bboxes, embeds)
        image  : float32 [1, 3, H, W]   RGB, divided by 255 (letterbox space)
        scores : float32 [1, N, num_prompts]   objectness, already sigmoid
        bboxes : float32 [1, N, 4]      decoded xyxy in the letter-boxed space
        embeds : float32 [1, N, 768]    per-box (BN-normed) embeddings

``N = 80² + 40² + 20² = 8400`` for a 640 input. DFL decoding is baked into the
graph, so the host only needs NMS + box-embedding/text matching + rescale (see
``infer_anything.py``).

Alongside the ONNX file this script writes ``<tag>_meta.json`` holding the
per-level contrastive ``logit_scale.exp()`` / ``bias`` and the per-level grid
sizes. Those exact constants are reused at inference to score box embeddings
against the text embeddings (``scale * <txt, emb> + bias``).

Usage
-----
    python wedetect_anything/export_onnx.py \
        --variant base \
        --checkpoint checkpoints/wedetect_base_uni.pth \
        --num-prompts 256 \
        --output-dir wedetect_anything/onnx_models
"""
import argparse
import json
import os
import os.path as osp
import sys

import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from models import (  # noqa: E402
    SimpleYOLOWorldProposalDetector,
    VisionProposalDeployModel,
    get_head_scale_bias,
    load_proposal_checkpoint,
)


def export_proposal(detector, img_size, save_path, opset):
    wrapper = VisionProposalDeployModel(detector, img_size=img_size).eval()
    dummy_img = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(
        wrapper, (dummy_img,), save_path,
        input_names=['input_image'],
        output_names=['scores', 'bboxes', 'embeds'],
        opset_version=opset,
        do_constant_folding=True)
    print(f'[proposal] exported -> {save_path}')


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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export WeDetect-Uni proposal model to ONNX')
    parser.add_argument('--variant', choices=['base', 'large'], default='base',
                        help='model variant')
    parser.add_argument('--checkpoint', required=True,
                        help='WeDetect-Uni mmdet checkpoint (.pth)')
    parser.add_argument('--num-prompts', type=int, default=256,
                        help='number of learnable prompt embeddings '
                             '(must match the checkpoint)')
    parser.add_argument('--prompt-dim', type=int, default=768,
                        help='embedding dimension')
    parser.add_argument('--img-size', type=int, default=640,
                        help='square input size (use 1280 for the large variant)')
    parser.add_argument('--output-dir', default='wedetect_anything/onnx_models',
                        help='directory to save the exported ONNX model + meta')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset')
    parser.add_argument('--device', default='cpu',
                        help='device used to load the model for export')
    parser.add_argument('--simplify', action='store_true',
                        help='run onnxsim on the exported model')
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    detector = SimpleYOLOWorldProposalDetector(
        args.variant, prompt_dim=args.prompt_dim, num_prompts=args.num_prompts)
    load_proposal_checkpoint(detector, args.checkpoint)
    detector = detector.to(args.device).eval()

    tag = f'wedetect_anything_{args.variant}'
    onnx_path = osp.join(args.output_dir, f'{tag}.onnx')
    export_proposal(detector, args.img_size, onnx_path, args.opset)
    if args.simplify:
        maybe_simplify(onnx_path)

    # Persist the per-level contrastive constants + grid sizes used by the host
    # to score box embeddings against the text embeddings.
    scales, biases = get_head_scale_bias(detector)
    grid_size = [(args.img_size // s) ** 2 for s in [8, 16, 32]]
    meta = {
        'variant': args.variant,
        'img_size': args.img_size,
        'num_prompts': args.num_prompts,
        'prompt_dim': args.prompt_dim,
        'grid_size': grid_size,
        'logit_scale_exp': scales,
        'bias': biases,
    }
    meta_path = osp.join(args.output_dir, f'{tag}_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f'[meta] saved -> {meta_path}\n{json.dumps(meta, ensure_ascii=False)}')
    print('Done.')


if __name__ == '__main__':
    main()
