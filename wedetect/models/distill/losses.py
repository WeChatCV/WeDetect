# Copyright (c) Tencent Inc. All rights reserved.
"""Distillation losses for WeDetect (large -> tiny).

The core idea follows the most industry-recognized recipe for YOLO-family
detector knowledge distillation:

- FGD (Focal and Global Distillation, CVPR 2022): GT-guided foreground /
  background separation combined with the teacher channel attention. This is
  the de-facto standard feature-level KD used in MMYOLO.
- CWD (Channel-wise Distillation, ICCV 2023): channel-wise KL divergence,
  provided as a robust alternative that needs no GT parsing.

Because WeDetect is a *retrieval-based* open-vocabulary detector whose
classification logits are produced by contrastive matching between region
embeddings and text embeddings, the teacher (large, xlm-roberta-large) and
the student (tiny, xlm-roberta-base) live in *different* text-embedding
spaces. Direct logit / response distillation is therefore invalid without
re-aligning the text spaces. Feature-level distillation on the neck outputs
(pan_out2 / pan_out1 / pan_out0) sidesteps this issue and is the most robust
choice for v1.

The student neck output channels ``[96, 192, 384]`` differ from the teacher
neck output channels ``[192, 384, 768]``, so a 1x1 conv adapter per level is
used to project the student features into the teacher channel space before
the distance is computed.
"""
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet.registry import MODELS


def _build_fg_mask(bboxes_labels: Optional[torch.Tensor],
                   feat_size: Sequence[int],
                   stride: int,
                   num_imgs: int,
                   device: Optional[torch.device] = None) -> torch.Tensor:
    """Build a foreground mask on a feature level from GT boxes.

    Args:
        bboxes_labels (Tensor | None): (N, 6) tensor of
            ``[batch_idx, label, x1, y1, x2, y2]`` in the (padded) input
            image coordinate space. If None, an all-background mask is
            returned (FGD degrades to a pure background term).
        feat_size (tuple): (H, W) of the target feature map.
        stride (int): feature stride.
        num_imgs (int): batch size.
        device (torch.device, optional): device for the mask when
            ``bboxes_labels`` is None. Ignored otherwise.

    Returns:
        Tensor: (B, 1, H, W) float mask, 1 inside GT boxes else 0.
    """
    H, W = feat_size
    # bboxes_labels may be None (e.g. an empty batch or a caller that forgot to
    # pass GT). In that case return an all-background mask so FGD degrades
    # gracefully to a pure background distillation term instead of crashing.
    if bboxes_labels is None:
        return torch.zeros((num_imgs, 1, H, W), device=device,
                           dtype=torch.float32)
    device = bboxes_labels.device
    mask = torch.zeros((num_imgs, 1, H, W), device=device, dtype=torch.float32)
    if bboxes_labels.numel() == 0:
        return mask

    batch_idx = bboxes_labels[:, 0].long()
    x1 = (bboxes_labels[:, 2] / stride).floor().long().clamp(0, W - 1)
    y1 = (bboxes_labels[:, 3] / stride).floor().long().clamp(0, H - 1)
    x2 = (bboxes_labels[:, 4] / stride).ceil().long().clamp(1, W)
    y2 = (bboxes_labels[:, 5] / stride).ceil().long().clamp(1, H)

    for b in range(num_imgs):
        sel = batch_idx == b
        if not sel.any():
            continue
        bx1, by1, bx2, by2 = x1[sel], y1[sel], x2[sel], y2[sel]
        for j in range(bx1.shape[0]):
            mask[b, 0, by1[j]:by2[j], bx1[j]:bx2[j]] = 1.0
    return mask


@MODELS.register_module()
class NeckDistillLoss(BaseModule):
    """Feature-level distillation loss on the neck outputs (P3/P4/P5).

    The student features are first projected to the teacher channel space via
    a 1x1 conv adapter per level, then one of two distances is computed:

    - ``mode='fgd'`` (default): FGD focal distillation. Teacher channel
      attention (GAP + sigmoid) re-weights the per-position squared
      difference, and the loss is split into a foreground term (inside GT
      boxes) and a background term, exactly as in FGD.
    - ``mode='cwd'``: channel-wise KL divergence (CWD). No GT is needed.

    Args:
        student_channels (list[int]): student neck output channels, e.g.
            ``[96, 192, 384]``.
        teacher_channels (list[int]): teacher neck output channels, e.g.
            ``[192, 384, 768]``.
        mode (str): ``'fgd'`` or ``'cwd'``.
        featmap_strides (list[int]): strides of the distilled levels,
            default ``[8, 16, 32]``.
        fg_weight (float): weight of the FGD foreground term.
        bg_weight (float): weight of the FGD background term.
        temperature (float): temperature for CWD.
        loss_weight (float): overall weight of the distillation loss.
    """

    def __init__(self,
                 student_channels: Sequence[int],
                 teacher_channels: Sequence[int],
                 mode: str = 'fgd',
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 fg_weight: float = 1.0,
                 bg_weight: float = 0.5,
                 temperature: float = 4.0,
                 loss_weight: float = 1.0,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert mode in ('fgd', 'cwd'), f'unsupported mode {mode}'
        assert len(student_channels) == len(teacher_channels), \
            'student/teacher channel lists must have the same length'
        self.mode = mode
        self.featmap_strides = list(featmap_strides)
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight
        self.temperature = temperature
        self.loss_weight = loss_weight

        # 1x1 conv adapters: project student channels -> teacher channels.
        self.aligns = nn.ModuleList([
            nn.Conv2d(s_c, t_c, kernel_size=1, stride=1, padding=0)
            for s_c, t_c in zip(student_channels, teacher_channels)
        ])

    def forward(self,
                s_feats: List[torch.Tensor],
                t_feats: List[torch.Tensor],
                bboxes_labels: Optional[torch.Tensor] = None,
                img_metas: Optional[List[dict]] = None) -> dict:
        """Compute the distillation loss.

        Args:
            s_feats (list[Tensor]): student neck outputs, each (B, Cs, H, W).
            t_feats (list[Tensor]): teacher neck outputs, each (B, Ct, H, W).
            bboxes_labels (Tensor, optional): GT for FGD mode.
            img_metas (list[dict], optional): needed to read
                ``batch_input_shape`` for FGD mode.

        Returns:
            dict: ``{'loss_distill': ...}`` (and per-level terms in fgd mode).
        """
        assert len(s_feats) == len(t_feats) == len(self.aligns), \
            'feature levels must match the adapters'
        num_imgs = s_feats[0].shape[0]
        loss_distill = s_feats[0].new_zeros([])

        for lvl, (s_feat, t_feat, align, stride) in enumerate(
                zip(s_feats, t_feats, self.aligns, self.featmap_strides)):
            t_feat = t_feat.detach()
            s_aligned = align(s_feat)
            # spatial sizes must match (teacher & student run at the same res).
            assert s_aligned.shape[-2:] == t_feat.shape[-2:], \
                'student/teacher feature spatial sizes must match'

            if self.mode == 'cwd':
                loss_distill = loss_distill + self._cwd(
                    s_aligned, t_feat, self.temperature)
            else:  # fgd
                fg_mask = _build_fg_mask(
                    bboxes_labels, s_aligned.shape[-2:], stride, num_imgs,
                    device=s_aligned.device)
                bg_mask = 1.0 - fg_mask
                fg_term, bg_term = self._fgd_focal(
                    s_aligned, t_feat, fg_mask, bg_mask)
                loss_distill = loss_distill + \
                    (self.fg_weight * fg_term + self.bg_weight * bg_term)

        loss_distill = loss_distill * self.loss_weight
        return {'loss_distill': loss_distill}

    @staticmethod
    def _fgd_focal(s_feat: torch.Tensor,
                   t_feat: torch.Tensor,
                   fg_mask: torch.Tensor,
                   bg_mask: torch.Tensor):
        """FGD focal distillation.

        N_t = sigmoid(GAP(t_feat))  -> teacher channel attention (B, C, 1, 1)
        diff = (s_feat - t_feat)^2 * N_t
        loss_fg = sum(diff * G) / (sum(G) * C)
        loss_bg = sum(diff * (1-G)) / (sum(1-G) * C)
        """
        C = s_feat.shape[1]
        n_t = F.adaptive_avg_pool2d(t_feat, 1).sigmoid()  # (B, C, 1, 1)
        diff = (s_feat - t_feat).pow(2) * n_t
        fg_sum = fg_mask.sum().clamp(min=1.0)
        bg_sum = bg_mask.sum().clamp(min=1.0)
        fg_loss = (diff * fg_mask).sum() / (fg_sum * C)
        bg_loss = (diff * bg_mask).sum() / (bg_sum * C)
        return fg_loss, bg_loss

    @staticmethod
    def _cwd(s_feat: torch.Tensor,
             t_feat: torch.Tensor,
             temperature: float) -> torch.Tensor:
        """Channel-wise KL divergence (CWD).

        For each (b, c) channel, subtract the spatial mean, form a spatial
        softmax distribution (with temperature), and minimise
        KL(p_teacher || p_student). Scaled by T^2.
        """
        B, C = s_feat.shape[:2]
        s = s_feat - s_feat.mean(dim=[2, 3], keepdim=True)
        t = t_feat - t_feat.mean(dim=[2, 3], keepdim=True)
        s = s.reshape(B * C, -1)
        t = t.reshape(B * C, -1)
        log_s = F.log_softmax(s / temperature, dim=1)
        t_soft = F.softmax(t / temperature, dim=1)
        loss = F.kl_div(log_s, t_soft, reduction='batchmean') \
            * (temperature ** 2)
        return loss
