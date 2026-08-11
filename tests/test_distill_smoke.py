# Copyright (c) Tencent Inc. All rights reserved.
"""Smoke test for the WeDetect distillation pipeline.

This test runs on a CPU-only environment WITHOUT compiled mmcv ops
(mmcv-lite). It stubs ``mmcv.ops`` so that ``import mmdet`` succeeds, then
exercises:

  1. module registration (DistillYOLOWorldDetector / NeckDistillLoss)
  2. ``_build_fg_mask`` correctness against a hand-crafted GT box
  3. ``NeckDistillLoss`` forward + backward in both ``fgd`` and ``cwd`` modes
  4. ``NeckDistillLoss`` numerical sanity (gradient flows to student features
     and to the alignment adapters; teacher features stay detached)

Run:
    python tests/test_distill_smoke.py
"""
import sys
import types

# ---------------------------------------------------------------------------
# 1. Stub mmcv.ops so mmdet imports cleanly under mmcv-lite (no C++ ops).
#    mmcv-lite ships NO compiled ops, and mmdet eagerly imports many
#    ``mmcv.ops.<sub>`` modules / names at package load time. We install a
#    meta-path finder that synthesizes a dummy module for any ``mmcv.ops.*``
#    import and exposes dummy attributes on demand. The distillation code path
#    never calls any of these, so no-ops are perfectly fine for unit tests.
# ---------------------------------------------------------------------------
import importlib.abc  # noqa: E402
import importlib.machinery  # noqa: E402


class _DummyOp:
    """A dummy class usable as a base class (e.g. for DeformConv2d)."""

    def __init__(self, *a, **k):
        pass


def _noop(*a, **k):
    return None


class _MmcvOpsStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Synthesize dummy modules for ``mmcv.ops`` and any ``mmcv.ops.*``."""

    _prefix = 'mmcv.ops'

    def find_module(self, fullname, path=None):
        if fullname == self._prefix or fullname.startswith(self._prefix + '.'):
            return self
        return None

    # py3.4+ API
    def find_spec(self, fullname, path=None, target=None):
        if fullname == self._prefix or fullname.startswith(self._prefix + '.'):
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__path__ = []          # mark as package so sub-imports work
        mod.__all__ = []
        mod.__loader__ = self
        return mod

    def exec_module(self, module):
        return None


def _make_dummy_module(name):
    mod = types.ModuleType(name)
    mod.__path__ = []
    mod.__all__ = []
    mod.__loader__ = _FINDER
    return mod


_FINDER = _MmcvOpsStubFinder()
sys.meta_path.insert(0, _FINDER)

# Pre-register the package + a few commonly-imported submodules with the names
# mmdet expects, so ``from mmcv.ops import X`` and ``from mmcv.ops.sub import Y``
# both work.
_ops_pkg = _make_dummy_module('mmcv.ops')
sys.modules['mmcv.ops'] = _ops_pkg

_dummy_submodules = {
    'nms': ['batched_nms', 'nms', 'nms_match', 'soft_nms'],
    'carafe': ['CARAFEPack', 'CARAFE', 'carafe', 'carafe_naive',
               'normal_init_carafe', 'xavier_init_carafe'],
    'fused_bias': ['fused_bias_leakyrelu', 'fused_bias_leakyrelu'],
    'deform_conv': ['deform_conv', 'DeformConv2d', 'ModulatedDeformConv2d',
                    'DeformConv2dPack', 'ModulatedDeformConv2dPack'],
    'modulated_deform_conv': ['ModulatedDeformConv2d',
                              'ModulatedDeformConv2dPack', 'modulated_deform_conv'],
    'roi_align': ['RoIAlign', 'roi_align', 'RoIAlignFunction'],
    'roi_pool': ['RoIPool', 'roi_pool'],
    'sigmoid_focal_loss': ['sigmoid_focal_loss', 'SigmoidFocalLoss'],
    'point_sample': ['point_sample', 'rel_roi_point_to_rel_img_point'],
    'active_rotated_filter': ['active_rotated_filter_forward',
                              'active_rotated_filter_backward'],
    'bbox_iou_rotated': ['bbox_iou_rotated', 'box_iou_rotated'],
    'riou_loss': ['riou_loss', 'rbox_iou_loss', 'diff_iou_rotated_2d',
                  'diff_iou_rotated_3d'],
    'masked_conv': ['MaskedConv2d', 'masked_conv'],
    'corner_pool': ['CornerPool', 'corner_pool'],
    'cc_attention': ['CrissCrossAttention'],
    'contour_expand': ['contour_expand'],
    'points_in_polygons': ['points_in_polygons'],
    'points_sampler': ['PointsSampler', 'get_uncertain_point_coords_with_randomness'],
    'assign_score_withk': ['assign_score_withk_forward',
                           'assign_score_withk_backward'],
    'furthest_point_sample': ['furthest_point_sample'],
    'gather_points': ['gather_points'],
    'group_points': ['group_points', 'GroupAll', 'QueryAndGroup'],
    'knn': ['KNN', 'knn'],
    'ball_query': ['ball_query', 'BallQuery'],
    'roiaware_pool3d': ['RoIAwarePool3d', 'RoIWarpper', 'roiaware_pool3d'],
    'voxelize': ['Voxelization', 'voxelize'],
    'box_iou_quadri': ['box_iou_quadri'],
    'box_iou_rotated': ['box_iou_rotated'],
    'upfirdn2d': ['upfirdn2d'],
    'sync_bn': ['SyncBatchNorm', 'TorchSyncBatchNorm'],
    'pixel_group': ['pixel_group'],
    'roi_align_rotated': ['RoIAlignRotated', 'roi_align_rotated'],
    'rotated_feature_align': ['rotated_feature_align'],
    'tetrahedra': ['GatherPoints', 'gather_points'],
    'min_area_polygons': ['min_area_polygons'],
}
for _sub, _names in _dummy_submodules.items():
    _mod = _make_dummy_module(f'mmcv.ops.{_sub}')
    for _n in _names:
        setattr(_mod, _n,
                _DummyOp if any(t in _n for t in
                                ('Conv', 'Pool', 'Pack', 'Align', 'Attention',
                                 'Sampler', 'Query', 'KNN', 'Ball', 'Voxel',
                                 'RoI')) else _noop)
    setattr(_ops_pkg, _sub, _mod)
    sys.modules[f'mmcv.ops.{_sub}'] = _mod

# Top-level names exposed by ``from mmcv.ops import X``.
for _n in (
        'CornerPool', 'DeformConv', 'MaskedConv', 'MultiScaleDeformableAttention',
        'RoIPool', 'RoIAlign', 'RoIAlignRotated', 'batched_nms', 'nms',
        'nms_match', 'soft_nms', 'deform_conv', 'point_sample',
        'rel_roi_point_to_rel_img_point', 'sigmoid_focal_loss',
        'DeformConv2d', 'ModulatedDeformConv2d', 'DeformConv2dPack',
        'ModulatedDeformConv2dPack', 'SimpleRoIAlign', 'CARAFEPack', 'CARAFE',
        'BBoxOverlaps2D', 'bbox_overlaps', 'mask_iou', 'warpperspective',
        'contour_expand', 'points_in_polygons', 'box_iou_rotated',
        'riou_loss', 'rbox_iou_loss', 'active_rotated_filter',
        'rotate_points', 'fused_bias_leakyrelu', 'Conv2dDeformConv2d',
        'ReliableGPU', 'upfirdn2d', 'SyncBatchNorm', 'MaskedConv2d',
        'CrissCrossAttention', 'Voxelization', 'PointsSampler',
        'RoIAwarePool3d', 'BallQuery', 'KNN'):
    setattr(_ops_pkg, _n,
            _DummyOp if any(t in _n for t in
                            ('Conv', 'Pool', 'Pack', 'Align', 'Attention',
                             'Sampler', 'Query', 'KNN', 'Ball', 'Voxel',
                             'RoI')) else _noop)

# mmcv-lite also drops a few classes from mmcv.cnn.bricks.transformer that
# depend on compiled ops (e.g. MultiScaleDeformableAttention). mmdet's
# pixel-decoder layers import them at module load time. Re-add no-op stubs.
try:
    from mmcv.cnn.bricks import transformer as _tf
    if not hasattr(_tf, 'MultiScaleDeformableAttention'):
        _tf.MultiScaleDeformableAttention = _DummyOp
except Exception:
    pass

import torch  # noqa: E402

from wedetect.models.distill.losses import NeckDistillLoss, _build_fg_mask  # noqa
from mmdet.registry import MODELS  # noqa


def _almost(a, b, tol=1e-6):
    return abs(float(a) - float(b)) < tol


def test_registration():
    cls = MODELS.get('DistillYOLOWorldDetector')
    assert cls is not None, 'DistillYOLOWorldDetector not registered'
    loss = MODELS.get('NeckDistillLoss')
    assert loss is not None, 'NeckDistillLoss not registered'
    print('[ok] registration: DistillYOLOWorldDetector & NeckDistillLoss')


def test_build_fg_mask():
    # 1 image, 1 GT box at stride 8 on a 4x4 feature map.
    # box covers x in [8, 24], y in [0, 16] -> feat cells x in [1, 3), y in [0, 2)
    #   x1 = floor(8/8) = 1 ; x2 = ceil(24/8) = 3  -> cols 1,2
    #   y1 = floor(0/8) = 0 ; y2 = ceil(16/8) = 2 -> rows 0,1
    bboxes_labels = torch.tensor([[0, 0, 8.0, 0.0, 24.0, 16.0]])
    mask = _build_fg_mask(bboxes_labels, feat_size=(4, 4), stride=8, num_imgs=1)
    assert mask.shape == (1, 1, 4, 4), mask.shape
    expected = torch.tensor([
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=torch.float32)[None, None]
    assert torch.equal(mask, expected), f'mask=\n{mask[0,0]}'
    print('[ok] _build_fg_mask: GT box projected correctly')


def _make_feats(B=2, Cs=(96, 192, 384), Ct=(192, 384, 768), Hs=(80, 40, 20)):
    s_feats = [torch.randn(B, c, h, h, requires_grad=True) for c, h in zip(Cs, Hs)]
    t_feats = [torch.randn(B, c, h, h) for c, h in zip(Ct, Hs)]
    return s_feats, t_feats


def test_cwd_mode():
    torch.manual_seed(0)
    s_feats, t_feats = _make_feats()
    loss_fn = NeckDistillLoss(
        student_channels=[96, 192, 384], teacher_channels=[192, 384, 768],
        mode='cwd', temperature=4.0, loss_weight=1.0)
    out = loss_fn(s_feats, t_feats, bboxes_labels=None, img_metas=None)
    assert 'loss_distill' in out
    out['loss_distill'].backward()
    # student features must receive grad
    for i, f in enumerate(s_feats):
        assert f.grad is not None, f'level {i} student grad is None'
        assert f.grad.abs().sum() > 0, f'level {i} student grad is zero'
    # alignment adapters must receive grad
    for i, align in enumerate(loss_fn.aligns):
        assert align.weight.grad is not None, f'align {i} weight grad is None'
        assert align.weight.grad.abs().sum() > 0, f'align {i} grad is zero'
    # teacher features must NOT require grad (they were never in the graph)
    for i, f in enumerate(t_feats):
        assert not f.requires_grad, f'teacher feat {i} should not require grad'
    print(f'[ok] cwd mode: loss={float(out["loss_distill"]):.4f}, '
          f'grads flow to student + adapters, teacher stays detached')


def test_fgd_mode():
    torch.manual_seed(0)
    s_feats, t_feats = _make_feats()
    # 2 images, 1 GT box each, in the (padded) 640x640 input space.
    bboxes_labels = torch.tensor([
        [0, 0,  64.0,  64.0, 192.0, 192.0],
        [1, 5, 128.0, 128.0, 256.0, 256.0],
    ])
    loss_fn = NeckDistillLoss(
        student_channels=[96, 192, 384], teacher_channels=[192, 384, 768],
        mode='fgd', featmap_strides=[8, 16, 32],
        fg_weight=1.0, bg_weight=0.5, loss_weight=1.0)
    out = loss_fn(s_feats, t_feats, bboxes_labels=bboxes_labels, img_metas=None)
    out['loss_distill'].backward()
    for i, f in enumerate(s_feats):
        assert f.grad is not None and f.grad.abs().sum() > 0
    for i, align in enumerate(loss_fn.aligns):
        assert align.weight.grad is not None and align.weight.grad.abs().sum() > 0
    print(f'[ok] fgd mode: loss={float(out["loss_distill"]):.4f}, '
          f'grads flow correctly with GT-guided fg/bg split')


def test_fgd_none_bboxes():
    """FGD mode with bboxes_labels=None must not crash (defensive)."""
    torch.manual_seed(0)
    s_feats, t_feats = _make_feats()
    loss_fn = NeckDistillLoss(
        student_channels=[96, 192, 384], teacher_channels=[192, 384, 768],
        mode='fgd', loss_weight=1.0)
    out = loss_fn(s_feats, t_feats, bboxes_labels=None, img_metas=None)
    out['loss_distill'].backward()
    assert out['loss_distill'].item() == out['loss_distill'].item()  # finite
    print(f'[ok] fgd mode with bboxes_labels=None: loss={float(out["loss_distill"]):.4f} (no crash)')


def test_loss_weight_scaling():
    torch.manual_seed(0)
    s_feats, t_feats = _make_feats()
    lw = 3.0
    # Build two identical losses (same adapter init) so the only difference is
    # the loss_weight multiplier. Use the same seed for both constructions.
    torch.manual_seed(42)
    loss_fn = NeckDistillLoss(
        student_channels=[96, 192, 384], teacher_channels=[192, 384, 768],
        mode='cwd', loss_weight=lw)
    torch.manual_seed(42)
    loss_fn2 = NeckDistillLoss(
        student_channels=[96, 192, 384], teacher_channels=[192, 384, 768],
        mode='cwd', loss_weight=1.0)
    # sanity: adapters are identical
    for a1, a2 in zip(loss_fn.aligns, loss_fn2.aligns):
        assert torch.equal(a1.weight, a2.weight), 'adapter init must match'
    out = loss_fn(s_feats, t_feats)
    out2 = loss_fn2(s_feats, t_feats)
    ratio = float(out['loss_distill']) / float(out2['loss_distill'])
    assert _almost(ratio, lw, tol=1e-4), f'loss_weight scaling ratio={ratio}'
    print(f'[ok] loss_weight scaling: ratio={ratio:.4f} (expected {lw})')


def test_channel_mismatch_asserts():
    """student/teacher channel length mismatch must raise."""
    try:
        NeckDistillLoss(
            student_channels=[96, 192], teacher_channels=[192, 384, 768],
            mode='cwd')
    except AssertionError:
        print('[ok] channel length mismatch correctly asserts')
    else:
        raise AssertionError('expected AssertionError for channel mismatch')


if __name__ == '__main__':
    test_registration()
    test_build_fg_mask()
    test_cwd_mode()
    test_fgd_mode()
    test_fgd_none_bboxes()
    test_loss_weight_scaling()
    test_channel_mismatch_asserts()
    print('\nAll distillation smoke tests passed.')
