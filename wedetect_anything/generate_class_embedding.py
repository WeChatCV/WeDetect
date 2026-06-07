# Copyright (c) Tencent Inc. All rights reserved.
"""Encode the large *prompt-free* vocabulary into text embeddings.

This is the WeDetect-Anything counterpart of the project-root
``generate_class_embedding.py``: instead of the 80 COCO classes it encodes the
very large open-world vocabulary defined in ``prompt_free_vocab.py`` (several
thousand Chinese category names) and stores the result as ``prompt_free.npy``.

At inference time every class-agnostic box embedding produced by the proposal
ONNX model is matched against this bank (cosine similarity + per-level scale /
bias + sigmoid + argmax) to "detect anything".

Output
------
``prompt_free.npy`` : float32 [vocab_size, 768], **L2-normalized** text
                      embeddings aligned row-by-row with ``prompt_free_name``.

Usage
-----
    python wedetect_anything/generate_class_embedding.py \
        --checkpoint checkpoints/wedetect_base.pth \
        --language-model xlm-roberta-base \
        --output wedetect_anything/prompt_free.npy
"""
import argparse
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from models import XLMRobertaLanguageBackbone  # noqa: E402
from prompt_free_vocab import prompt_free_name  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='Encode the prompt-free vocabulary into text embeddings.')
    parser.add_argument('--checkpoint', required=True,
                        help='WeDetect mmdet checkpoint (.pth); only the '
                             'text-encoder weights are used here')
    parser.add_argument('--language-model', default='xlm-roberta-base',
                        help='path/name of the XLM-RoBERTa model directory')
    parser.add_argument('--output', default='prompt_free.npy',
                        help='path to save the text embeddings (.npy)')
    parser.add_argument('--batch-size', type=int, default=80,
                        help='number of category names encoded per forward')
    parser.add_argument('--device', default='cuda',
                        help='inference device, e.g. "cuda" or "cpu"')
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == 'cpu' else 'cpu')

    names = list(prompt_free_name)
    print(f'[vocab] {len(names)} category names to encode')

    encoder = XLMRobertaLanguageBackbone(
        args.language_model, args.checkpoint).to(device).eval()

    embeddings = []
    bs = args.batch_size
    num_iters = (len(names) + bs - 1) // bs
    for i in range(num_iters):
        batch = names[i * bs:(i + 1) * bs]
        embeddings.append(encoder(batch))
        print(f'\r[encode] {min((i + 1) * bs, len(names))}/{len(names)}',
              end='', flush=True)
    print()

    text_embeddings = torch.cat(embeddings, dim=0)
    text_embeddings = F.normalize(text_embeddings, dim=-1).cpu().numpy()
    text_embeddings = text_embeddings.astype(np.float32)

    assert text_embeddings.shape[0] == len(names), \
        'embedding count must match the vocabulary size'
    np.save(args.output, text_embeddings)
    print(f'[done] saved {text_embeddings.shape} -> {args.output}')


if __name__ == '__main__':
    main()
