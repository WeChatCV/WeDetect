# Copyright (c) Tencent Inc. All rights reserved.
"""Knowledge-distillation detector: distill WeDetect-Large -> WeDetect-Tiny.

The detector owns a frozen *teacher* (WeDetect-Large) and a trainable
*student* (WeDetect-Tiny, this is the actual ``self`` model). During
``loss()`` the student runs the full forward (backbone + text + neck + head)
to produce the standard detection losses, while the teacher only runs the
image path (``backbone.forward_image`` + ``neck``) under ``torch.no_grad`` to
produce the neck features used by :class:`NeckDistillLoss`.

Design notes:
- The teacher's ``data_preprocessor`` is *not* used: the student's
  ``data_preprocessor`` has already normalized the inputs (identical config:
  mean=0, std=255, bgr2rgb), so the preprocessed ``batch_inputs`` are fed
  directly to ``teacher.backbone.forward_image``.
- The teacher text model is skipped on purpose: WeDetect uses ``mm_neck=
  False`` (image-only neck), so the neck features do not depend on text.
  Skipping the (large) teacher text encoder saves a lot of compute.
- The teacher is excluded from the saved ``state_dict`` so that checkpoints
  stay small. The teacher is always rebuilt in ``__init__`` and reloaded from
  its own ``teacher_checkpoint`` in ``init_weights`` (after the recursive
  init), so excluding it from checkpoints is safe for resume.
"""
from typing import List, Optional, Union

import torch
from torch import Tensor
from mmdet.structures import SampleList
from mmengine.logging import print_log
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS

from .yolo_world import YOLOWorldDetector


def _is_teacher_key(key: str) -> bool:
    return key.startswith('teacher.') or key.startswith('module.teacher.')


@MODELS.register_module()
class DistillYOLOWorldDetector(YOLOWorldDetector):
    """WeDetect distillation detector (Large teacher -> Tiny student).

    Args:
        teacher_cfg (dict): config of the teacher detector (a full
            ``YOLOWorldDetector`` config for the large model).
        teacher_checkpoint (str): path to the teacher checkpoint.
        distill_loss (dict): config of the :class:`NeckDistillLoss`.
        student_channels (list[int]): student neck output channels.
        teacher_channels (list[int]): teacher neck output channels.
        save_teacher (bool): keep teacher in ``state_dict``. Defaults to
            ``False`` (recommended).
    """

    def __init__(self,
                 teacher_cfg: dict,
                 teacher_checkpoint: str,
                 distill_loss: dict,
                 student_channels: List[int],
                 teacher_channels: List[int],
                 save_teacher: bool = False,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Build the teacher (a full YOLOWorldDetector).
        # Built AFTER super().__init__ so the parent SyncBatchNorm conversion
        # does not touch the frozen teacher. The teacher weights are loaded in
        # ``init_weights`` (after super) because mmengine's init_weights
        # recursively re-initialises submodules (e.g. head biases), which
        # would otherwise overwrite the checkpoint loaded here.
        self.teacher = MODELS.build(teacher_cfg)
        self.teacher_checkpoint = teacher_checkpoint
        self._freeze_teacher()

        self.save_teacher = save_teacher

        # Build the distillation loss (holds the channel-alignment adapters).
        distill_loss_cfg = dict(distill_loss)
        distill_loss_cfg.setdefault('student_channels', student_channels)
        distill_loss_cfg.setdefault('teacher_channels', teacher_channels)
        self.distill_loss = MODELS.build(distill_loss_cfg)

    def init_weights(self):
        """Init student weights, then (re)load the frozen teacher checkpoint.

        ``super().init_weights()`` recursively re-initialises submodules
        (including the teacher's head biases), so the teacher checkpoint is
        reloaded afterwards to restore the correct frozen weights. This runs
        before the runner applies ``load_from`` (student checkpoint), so the
        final order is: init -> teacher reload -> student load_from.
        """
        super().init_weights()
        if self.teacher_checkpoint:
            load_checkpoint(self.teacher, self.teacher_checkpoint,
                            map_location='cpu')
            print_log(
                f'Loaded teacher checkpoint from {self.teacher_checkpoint}',
                'current')
        self._freeze_teacher()

    def _freeze_teacher(self):
        """Freeze the teacher and keep it in eval mode forever."""
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        """Keep the teacher frozen/eval even when the student is training."""
        super().train(mode)
        self._freeze_teacher()
        return self

    @torch.no_grad()
    def _teacher_neck_feats(self, batch_inputs: Tensor) -> List[Tensor]:
        """Run the teacher image path only (backbone + neck), no text model."""
        backbone_feats = self.teacher.backbone.forward_image(batch_inputs)
        if self.teacher.with_neck:
            # WeDetect uses mm_neck=False (image-only neck).
            if self.teacher.mm_neck:
                # Fallback: should not happen for WeDetect, but stay safe.
                raise RuntimeError(
                    'Teacher mm_neck=True is unsupported for distillation; '
                    'the teacher text path would be required.')
            img_feats = self.teacher.neck(backbone_feats)
        else:
            img_feats = backbone_feats
        return img_feats

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Student detection loss + teacher-student feature distillation."""
        self.bbox_head.num_classes = self.num_train_classes

        # ---- student forward (full: backbone + text + neck + head) ----
        s_img_feats, s_txt_feats = self.extract_feat(
            batch_inputs, batch_data_samples)
        det_losses = self.bbox_head.loss(
            s_img_feats, s_txt_feats, batch_data_samples)

        # ---- teacher forward (image path only, no grad) ----
        t_img_feats = self._teacher_neck_feats(batch_inputs)

        # ---- distillation loss on neck outputs ----
        bboxes_labels = batch_data_samples.get('bboxes_labels', None) \
            if isinstance(batch_data_samples, dict) else None
        img_metas = batch_data_samples['img_metas'] \
            if isinstance(batch_data_samples, dict) else None
        distill_losses = self.distill_loss(
            s_img_feats, t_img_feats, bboxes_labels, img_metas)

        return {**det_losses, **distill_losses}

    # ------------------------------------------------------------------
    # Checkpoint handling: keep the teacher out of the saved state_dict so
    # that per-epoch checkpoints stay small. Resume is safe because the
    # teacher is always rebuilt + loaded from ``teacher_checkpoint``.
    # ------------------------------------------------------------------
    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        if self.save_teacher:
            return sd
        return {k: v for k, v in sd.items() if not _is_teacher_key(k)}

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        filtered = {k: v for k, v in state_dict.items()
                    if not _is_teacher_key(k)}
        # Teacher keys are intentionally absent (the teacher is rebuilt in
        # __init__ and reloaded from teacher_checkpoint in init_weights), so
        # tolerate missing keys.
        return super().load_state_dict(filtered, strict=False, **kwargs)
