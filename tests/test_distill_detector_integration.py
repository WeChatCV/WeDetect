# Copyright (c) Tencent Inc. All rights reserved.
"""Integration test for ``DistillYOLOWorldDetector.loss()`` wiring.

This test does NOT build the heavy ConvNeXt / XLM-Roberta backbones. Instead
it constructs a minimal ``DistillYOLOWorldDetector`` and monkeypatches the
feature-extraction paths so that we can verify the *wiring* in ``loss()``:

  - student ``extract_feat`` is called and its features feed the head losses,
  - teacher ``_teacher_neck_feats`` is called under ``no_grad``,
  - ``distill_loss`` receives (student_feats, teacher_feats, bboxes_labels,
    img_metas) and produces ``loss_distill``,
  - the returned dict contains both the detection losses and ``loss_distill``,
  - backward propagates into the student backbone params (which we mark as
    requiring grad) but NOT into the teacher params.

Run:
    PYTHONPATH=/workspace python tests/test_distill_detector_integration.py
"""
import sys
import types
import importlib.abc
import importlib.machinery

# ---- mmcv.ops stub (so mmdet imports under mmcv-lite) ---------------------
class _DummyOp:
    def __init__(self, *a, **k):
        pass


def _noop(*a, **k):
    return None


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    _p = 'mmcv.ops'

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self._p or fullname.startswith(self._p + '.'):
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__path__ = []
        m.__all__ = []
        m.__loader__ = self
        return m

    def exec_module(self, module):
        return None


_F = _Finder()
sys.meta_path.insert(0, _F)


def _mk(n):
    m = types.ModuleType(n)
    m.__path__ = []
    m.__all__ = []
    m.__loader__ = _F
    return m


pkg = _mk('mmcv.ops')
sys.modules['mmcv.ops'] = pkg
for _sub in ['nms', 'carafe', 'fused_bias', 'deform_conv',
             'modulated_deform_conv', 'roi_align', 'roi_pool',
             'sigmoid_focal_loss', 'point_sample', 'active_rotated_filter',
             'bbox_iou_rotated', 'riou_loss', 'masked_conv', 'corner_pool',
             'cc_attention', 'contour_expand', 'points_in_polygons',
             'points_sampler', 'assign_score_withk', 'furthest_point_sample',
             'gather_points', 'group_points', 'knn', 'ball_query',
             'roiaware_pool3d', 'voxelize', 'box_iou_quadri',
             'box_iou_rotated', 'upfirdn2d', 'sync_bn', 'pixel_group',
             'roi_align_rotated', 'rotated_feature_align', 'tetrahedra',
             'min_area_polygons']:
    sm = _mk(f'mmcv.ops.{_sub}')
    sys.modules[f'mmcv.ops.{_sub}'] = sm
    setattr(pkg, _sub, sm)
for _n in ['CornerPool', 'DeformConv', 'MaskedConv',
           'MultiScaleDeformableAttention', 'RoIPool', 'RoIAlign',
           'RoIAlignRotated', 'batched_nms', 'nms', 'nms_match', 'soft_nms',
           'deform_conv', 'point_sample', 'rel_roi_point_to_rel_img_point',
           'sigmoid_focal_loss', 'DeformConv2d', 'ModulatedDeformConv2d',
           'DeformConv2dPack', 'ModulatedDeformConv2dPack', 'SimpleRoIAlign',
           'CARAFEPack', 'CARAFE', 'BBoxOverlaps2D', 'bbox_overlaps',
           'mask_iou', 'warpperspective', 'contour_expand',
           'points_in_polygons', 'box_iou_rotated', 'riou_loss',
           'rbox_iou_loss', 'active_rotated_filter', 'rotate_points',
           'fused_bias_leakyrelu', 'Conv2dDeformConv2d', 'ReliableGPU',
           'upfirdn2d', 'SyncBatchNorm', 'MaskedConv2d',
           'CrissCrossAttention', 'Voxelization', 'PointsSampler',
           'RoIAwarePool3d', 'BallQuery', 'KNN']:
    setattr(pkg, _n,
            _DummyOp if any(t in _n for t in
                            ('Conv', 'Pool', 'Pack', 'Align', 'Attention',
                             'Sampler', 'Query', 'KNN', 'Ball', 'Voxel',
                             'RoI')) else _noop)
try:
    from mmcv.cnn.bricks import transformer as _tf
    if not hasattr(_tf, 'MultiScaleDeformableAttention'):
        _tf.MultiScaleDeformableAttention = _DummyOp
except Exception:
    pass

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from wedetect.models.detectors.distill_yolo_world import (  # noqa: E402
    DistillYOLOWorldDetector)
from wedetect.models.distill.losses import NeckDistillLoss  # noqa: E402


class _MockBackbone(nn.Module):
    """Tiny fake backbone producing 3-level features with per-level channels.

    Each level is produced by its own conv so that the channel count matches
    ``channels`` (e.g. student [96,192,384] / teacher [192,384,768]). This
    mirrors what the real ConvNeXt+CSPPAN pipeline produces and is what the
    distillation alignment adapters expect.
    """

    def __init__(self, channels, H=32):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Conv2d(3, c, 3, 2, 1) for c in channels
        ])
        self.H = H
        self.channels = channels

    def forward_image(self, x):
        return [proj(x) for proj in self.projs]


class _MockNeck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, feats):
        # identity, just ensure channel count matches
        return list(feats)


class _MockHead(nn.Module):
    """Fake head that returns a dummy detection loss (scalar)."""

    def __init__(self):
        super().__init__()
        self.num_classes = 80
        self.linear = nn.Linear(96, 1)  # so head has trainable params

    def loss(self, img_feats, txt_feats, batch_data_samples):
        # produce a scalar detection loss that depends on student feats
        s = sum(f.sum() for f in img_feats) * 0.0  # zero but in graph
        # img_feats[0]: (B, 96, H, W) -> take one spatial point -> (B, 96)
        s = s + self.linear(img_feats[0][:, :, 0, 0]).sum() * 0.001
        return {'loss_cls': s, 'loss_bbox': s.detach() * 0.0 + 0.0}

    def predict(self, *a, **k):
        return [None]


def _build_mock_detector(mode='fgd'):
    """Build a DistillYOLOWorldDetector with mocked heavy submodules.

    We bypass ``MODELS.build`` for backbone/neck/head by manually constructing
    the parent and then overriding the attributes. The teacher is a plain
    ``nn.Module`` with a ``backbone.forward_image`` and a ``neck``.
    """
    s_ch = [96, 192, 384]
    t_ch = [192, 384, 768]

    # Teacher: a bare module with backbone + neck + with_neck + mm_neck attrs.
    teacher = nn.Module()
    teacher.backbone = _MockBackbone(t_ch)
    teacher.neck = _MockNeck(t_ch)
    teacher.with_neck = True
    teacher.mm_neck = False
    teacher.data_preprocessor = None

    # Build the distill loss directly (it holds the alignment adapters).
    distill_loss = NeckDistillLoss(
        student_channels=s_ch, teacher_channels=t_ch,
        mode=mode, featmap_strides=[8, 16, 32],
        fg_weight=1.0, bg_weight=0.5, temperature=4.0, loss_weight=1.0)

    # Build the detector without calling the heavy __init__ path. We construct
    # a bare object and set the minimum attributes loss() touches.
    det = DistillYOLOWorldDetector.__new__(DistillYOLOWorldDetector)
    # nn.Module.__init__ sets up _modules/_parameters/_buffers dicts needed for
    # __setattr__ to register submodules. BaseDetector.__init__ adds more state
    # we do not need here.
    nn.Module.__init__(det)
    # YOLOWorldDetector / YOLODetector / BaseDetector attributes
    det.mm_neck = False
    det.num_train_classes = 80
    det.num_test_classes = 80
    # NOTE: with_neck is a BaseDetector property == (self.neck is not None),
    # so setting det.neck below is enough; do not assign with_neck directly.
    det.bbox_head = _MockHead()
    det.backbone = _MockBackbone(s_ch)
    det.neck = _MockNeck(s_ch)
    # distillation-specific
    det.teacher = teacher
    det.distill_loss = distill_loss
    det.save_teacher = False
    det.teacher_checkpoint = None
    # init the trainable student params (backbone + neck + head + adapters)
    det._freeze_teacher = lambda: _freeze_teacher(teacher)
    _freeze_teacher(teacher)
    return det, teacher


def _freeze_teacher(teacher):
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False


def test_loss_wiring_fgd():
    torch.manual_seed(0)
    det, teacher = _build_mock_detector(mode='fgd')

    # Replace extract_feat so we control the student feature graph.
    def extract_feat(batch_inputs, batch_data_samples):
        feats = det.backbone.forward_image(batch_inputs)
        feats = det.neck(feats)
        txt_feats = [torch.zeros(2, 80, 96)]  # unused by mock head loss
        return feats, txt_feats
    det.extract_feat = extract_feat

    B = 2
    batch_inputs = torch.randn(B, 3, 64, 64)
    bboxes_labels = torch.tensor([
        [0, 0, 16., 16., 48., 48.],
        [1, 5, 32., 32., 64., 64.],
    ])
    img_metas = [{'batch_input_shape': (64, 64)}] * B
    batch_data_samples = {
        'bboxes_labels': bboxes_labels,
        'img_metas': img_metas,
        'texts': [['cat'], ['dog']],
    }

    losses = det.loss(batch_inputs, batch_data_samples)

    # detection losses + distill loss must all be present
    assert 'loss_cls' in losses, losses.keys()
    assert 'loss_distill' in losses, losses.keys()
    total = sum(v.sum() if torch.is_tensor(v) else v for v in losses.values())
    total.backward()

    # student backbone params must have grad
    assert det.backbone.projs[0].weight.grad is not None
    assert det.backbone.projs[0].weight.grad.abs().sum() > 0, \
        'student backbone must receive non-zero grad'
    # teacher params must NOT have grad
    assert teacher.backbone.projs[0].weight.grad is None, \
        'teacher must not receive grad'
    # alignment adapters must have grad
    for i, align in enumerate(det.distill_loss.aligns):
        assert align.weight.grad is not None, f'align {i} grad is None'
        assert align.weight.grad.abs().sum() > 0, f'align {i} grad is zero'

    loss_repr = {k: (float(v) if torch.is_tensor(v) else v)
                 for k, v in losses.items()}
    print(f'[ok] fgd wiring: losses={loss_repr}')


def test_loss_wiring_cwd():
    torch.manual_seed(0)
    det, teacher = _build_mock_detector(mode='cwd')

    def extract_feat(batch_inputs, batch_data_samples):
        feats = det.backbone.forward_image(batch_inputs)
        feats = det.neck(feats)
        txt_feats = [torch.zeros(2, 80, 96)]
        return feats, txt_feats
    det.extract_feat = extract_feat

    B = 2
    batch_inputs = torch.randn(B, 3, 64, 64)
    batch_data_samples = {
        'bboxes_labels': None,  # CWD does not need GT
        'img_metas': [{'batch_input_shape': (64, 64)}] * B,
        'texts': [['cat'], ['dog']],
    }
    losses = det.loss(batch_inputs, batch_data_samples)
    assert 'loss_distill' in losses
    total = sum(v.sum() if torch.is_tensor(v) else v for v in losses.values())
    total.backward()
    assert det.backbone.projs[0].weight.grad is not None
    assert teacher.backbone.projs[0].weight.grad is None
    print('[ok] cwd wiring: backward flows to student only, teacher frozen')


def test_state_dict_excludes_teacher():
    det, teacher = _build_mock_detector(mode='cwd')
    sd = det.state_dict()
    teacher_keys = [k for k in sd if k.startswith('teacher.')]
    assert len(teacher_keys) == 0, \
        f'teacher keys leaked into state_dict: {teacher_keys[:5]}'
    # alignment adapter keys must be present (3 levels x {weight, bias} = 6)
    align_keys = [k for k in sd if k.startswith('distill_loss.aligns')]
    assert len(align_keys) == 6, \
        f'expected 6 adapter entries (3 levels x 2), got {align_keys}'
    print(f'[ok] state_dict excludes teacher ({len(sd)} keys, '
          f'{len(align_keys)} adapter keys)')


def test_save_teacher_true_keeps_teacher():
    det, teacher = _build_mock_detector(mode='cwd')
    det.save_teacher = True
    sd = det.state_dict()
    teacher_keys = [k for k in sd if k.startswith('teacher.')]
    assert len(teacher_keys) > 0, 'save_teacher=True should keep teacher keys'
    print(f'[ok] save_teacher=True keeps {len(teacher_keys)} teacher keys')


def test_train_keeps_teacher_frozen():
    det, teacher = _build_mock_detector(mode='cwd')
    det.train()
    assert not teacher.training, 'teacher must stay in eval mode'
    for p in teacher.parameters():
        assert not p.requires_grad, 'teacher params must stay frozen'
    print('[ok] det.train() keeps teacher in eval + frozen')


if __name__ == '__main__':
    test_loss_wiring_fgd()
    test_loss_wiring_cwd()
    test_state_dict_excludes_teacher()
    test_save_teacher_true_keeps_teacher()
    test_train_keeps_teacher_frozen()
    print('\nAll DistillYOLOWorldDetector integration tests passed.')
