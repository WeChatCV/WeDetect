# Copyright (c) Tencent Inc. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import HorizontalBoxes
from .utils import (select_candidates_in_gts, select_highest_overlaps,
                    yolov6_iou_calculator)



def bbox_overlaps(pred: torch.Tensor,
                  target: torch.Tensor,
                  iou_mode: str = 'ciou',
                  bbox_format: str = 'xywh',
                  siou_theta: float = 4.0,
                  eps: float = 1e-7) -> torch.Tensor:
    r"""Calculate overlap between two set of bboxes.
    `Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    In the CIoU implementation of YOLOv5 and MMDetection, there is a slight
    difference in the way the alpha parameter is computed.

    mmdet version:
        alpha = (ious > 0.5).float() * v / (1 - ious + v)
    YOLOv5 version:
        alpha = v / (v - ious + (1 + eps)

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2)
            or (x, y, w, h),shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        iou_mode (str): Options are ('iou', 'ciou', 'giou', 'siou').
            Defaults to "ciou".
        bbox_format (str): Options are "xywh" and "xyxy".
            Defaults to "xywh".
        siou_theta (float): siou_theta for SIoU when calculate shape cost.
            Defaults to 4.0.
        eps (float): Eps to avoid log(0).

    Returns:
        Tensor: shape (n, ).
    """
    assert iou_mode in ('iou', 'ciou', 'giou', 'siou')
    assert bbox_format in ('xyxy', 'xywh')
    if bbox_format == 'xywh':
        pred = HorizontalBoxes.cxcywh_to_xyxy(pred)
        target = HorizontalBoxes.cxcywh_to_xyxy(target)

    bbox1_x1, bbox1_y1 = pred[..., 0], pred[..., 1]
    bbox1_x2, bbox1_y2 = pred[..., 2], pred[..., 3]
    bbox2_x1, bbox2_y1 = target[..., 0], target[..., 1]
    bbox2_x2, bbox2_y2 = target[..., 2], target[..., 3]

    # Overlap
    overlap = (torch.min(bbox1_x2, bbox2_x2) -
               torch.max(bbox1_x1, bbox2_x1)).clamp(0) * \
              (torch.min(bbox1_y2, bbox2_y2) -
               torch.max(bbox1_y1, bbox2_y1)).clamp(0)

    # Union
    w1, h1 = bbox1_x2 - bbox1_x1, bbox1_y2 - bbox1_y1
    w2, h2 = bbox2_x2 - bbox2_x1, bbox2_y2 - bbox2_y1
    union = (w1 * h1) + (w2 * h2) - overlap + eps

    h1 = bbox1_y2 - bbox1_y1 + eps
    h2 = bbox2_y2 - bbox2_y1 + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[..., :2], target[..., :2])
    enclose_x2y2 = torch.max(pred[..., 2:], target[..., 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    enclose_w = enclose_wh[..., 0]  # cw
    enclose_h = enclose_wh[..., 1]  # ch

    if iou_mode == 'ciou':
        # CIoU = IoU - ( (ρ^2(b_pred,b_gt) / c^2) + (alpha x v) )

        # calculate enclose area (c^2)
        enclose_area = enclose_w**2 + enclose_h**2 + eps

        # calculate ρ^2(b_pred,b_gt):
        # euclidean distance between b_pred(bbox2) and b_gt(bbox1)
        # center point, because bbox format is xyxy -> left-top xy and
        # right-bottom xy, so need to / 4 to get center point.
        rho2_left_item = ((bbox2_x1 + bbox2_x2) - (bbox1_x1 + bbox1_x2))**2 / 4
        rho2_right_item = ((bbox2_y1 + bbox2_y2) -
                           (bbox1_y1 + bbox1_y2))**2 / 4
        rho2 = rho2_left_item + rho2_right_item  # rho^2 (ρ^2)

        # Width and height ratio (v)
        wh_ratio = (4 / (math.pi**2)) * torch.pow(
            torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

        with torch.no_grad():
            alpha = wh_ratio / (wh_ratio - ious + (1 + eps))

        # CIoU
        ious = ious - ((rho2 / enclose_area) + (alpha * wh_ratio))

    elif iou_mode == 'giou':
        # GIoU = IoU - ( (A_c - union) / A_c )
        convex_area = enclose_w * enclose_h + eps  # convex area (A_c)
        ious = ious - (convex_area - union) / convex_area

    elif iou_mode == 'siou':
        # SIoU: https://arxiv.org/pdf/2205.12740.pdf
        # SIoU = IoU - ( (Distance Cost + Shape Cost) / 2 )

        # calculate sigma (σ):
        # euclidean distance between bbox2(pred) and bbox1(gt) center point,
        # sigma_cw = b_cx_gt - b_cx
        sigma_cw = (bbox2_x1 + bbox2_x2) / 2 - (bbox1_x1 + bbox1_x2) / 2 + eps
        # sigma_ch = b_cy_gt - b_cy
        sigma_ch = (bbox2_y1 + bbox2_y2) / 2 - (bbox1_y1 + bbox1_y2) / 2 + eps
        # sigma = √( (sigma_cw ** 2) - (sigma_ch ** 2) )
        sigma = torch.pow(sigma_cw**2 + sigma_ch**2, 0.5)

        # choose minimize alpha, sin(alpha)
        sin_alpha = torch.abs(sigma_ch) / sigma
        sin_beta = torch.abs(sigma_cw) / sigma
        sin_alpha = torch.where(sin_alpha <= math.sin(math.pi / 4), sin_alpha,
                                sin_beta)

        # Angle cost = 1 - 2 * ( sin^2 ( arcsin(x) - (pi / 4) ) )
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)

        # Distance cost = Σ_(t=x,y) (1 - e ^ (- γ ρ_t))
        rho_x = (sigma_cw / enclose_w)**2  # ρ_x
        rho_y = (sigma_ch / enclose_h)**2  # ρ_y
        gamma = 2 - angle_cost  # γ
        distance_cost = (1 - torch.exp(-1 * gamma * rho_x)) + (
            1 - torch.exp(-1 * gamma * rho_y))

        # Shape cost = Ω = Σ_(t=w,h) ( ( 1 - ( e ^ (-ω_t) ) ) ^ θ )
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)  # ω_w
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)  # ω_h
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w),
                               siou_theta) + torch.pow(
                                   1 - torch.exp(-1 * omiga_h), siou_theta)

        ious = ious - ((distance_cost + shape_cost) * 0.5)

    return ious.clamp(min=-1.0, max=1.0)




@TASK_UTILS.register_module()
class BatchTaskAlignedAssigner(nn.Module):
    """This code referenced to
    https://github.com/meituan/YOLOv6/blob/main/yolov6/
    assigners/tal_assigner.py.
    Batch Task aligned assigner base on the paper:
    `TOOD: Task-aligned One-stage Object Detection.
    <https://arxiv.org/abs/2108.07755>`_.
    Assign a corresponding gt bboxes or background to a batch of
    predicted bboxes. Each bbox will be assigned with `0` or a
    positive integer indicating the ground truth index.
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        num_classes (int): number of class
        topk (int): number of bbox selected in each level
        alpha (float): Hyper-parameters related to alignment_metrics.
            Defaults to 1.0
        beta (float): Hyper-parameters related to alignment_metrics.
            Defaults to 6.
        eps (float): Eps to avoid log(0). Default set to 1e-9
        use_ciou (bool): Whether to use ciou while calculating iou.
            Defaults to False.
    """

    def __init__(self,
                 num_classes: int,
                 topk: int = 13,
                 alpha: float = 1.0,
                 beta: float = 6.0,
                 eps: float = 1e-7,
                 use_ciou: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.use_ciou = use_ciou

    @torch.no_grad()
    def forward(
        self,
        pred_bboxes: Tensor,
        pred_scores: Tensor,
        priors: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        pad_bbox_flag: Tensor,
    ) -> dict:
        """Assign gt to bboxes.

        The assignment is done in following steps
        1. compute alignment metric between all bbox (bbox of all pyramid
           levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free
           detector only can predict positive distance)
        Args:
            pred_bboxes (Tensor): Predict bboxes,
                shape(batch_size, num_priors, 4)
            pred_scores (Tensor): Scores of predict bboxes,
                shape(batch_size, num_priors, num_classes)
            priors (Tensor): Model priors,  shape (num_priors, 4)
            gt_labels (Tensor): Ground true labels,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground true bboxes,
                shape(batch_size, num_gt, 4)
            pad_bbox_flag (Tensor): Ground truth bbox mask,
                1 means bbox, 0 means no bbox,
                shape(batch_size, num_gt, 1)
        Returns:
            assigned_result (dict) Assigned result:
                assigned_labels (Tensor): Assigned labels,
                    shape(batch_size, num_priors)
                assigned_bboxes (Tensor): Assigned boxes,
                    shape(batch_size, num_priors, 4)
                assigned_scores (Tensor): Assigned scores,
                    shape(batch_size, num_priors, num_classes)
                fg_mask_pre_prior (Tensor): Force ground truth matching mask,
                    shape(batch_size, num_priors)
        """
        # (num_priors, 4) -> (num_priors, 2)
        priors = priors[:, :2]

        batch_size = pred_scores.size(0)
        num_gt = gt_bboxes.size(1)

        assigned_result = {
            'assigned_labels':
            gt_bboxes.new_full(pred_scores[..., 0].shape, self.num_classes),
            'assigned_bboxes':
            gt_bboxes.new_full(pred_bboxes.shape, 0),
            'assigned_scores':
            gt_bboxes.new_full(pred_scores.shape, 0),
            'fg_mask_pre_prior':
            gt_bboxes.new_full(pred_scores[..., 0].shape, 0)
        }

        if num_gt == 0:
            return assigned_result

        pos_mask, alignment_metrics, overlaps = self.get_pos_mask(
            pred_bboxes, pred_scores, priors, gt_labels, gt_bboxes,
            pad_bbox_flag, batch_size, num_gt)

        (assigned_gt_idxs, fg_mask_pre_prior,
         pos_mask) = select_highest_overlaps(pos_mask, overlaps, num_gt)

        # assigned target
        assigned_labels, assigned_bboxes, assigned_scores = self.get_targets(
            gt_labels, gt_bboxes, assigned_gt_idxs, fg_mask_pre_prior,
            batch_size, num_gt)

        # normalize
        alignment_metrics *= pos_mask
        pos_align_metrics = alignment_metrics.max(axis=-1, keepdim=True)[0]
        pos_overlaps = (overlaps * pos_mask).max(axis=-1, keepdim=True)[0]
        norm_align_metric = (
            alignment_metrics * pos_overlaps /
            (pos_align_metrics + self.eps)).max(-2)[0].unsqueeze(-1)
        assigned_scores = assigned_scores * norm_align_metric

        assigned_result['assigned_labels'] = assigned_labels
        assigned_result['assigned_bboxes'] = assigned_bboxes
        assigned_result['assigned_scores'] = assigned_scores
        assigned_result['fg_mask_pre_prior'] = fg_mask_pre_prior.bool()
        return assigned_result

    def get_pos_mask(self, pred_bboxes: Tensor, pred_scores: Tensor,
                     priors: Tensor, gt_labels: Tensor, gt_bboxes: Tensor,
                     pad_bbox_flag: Tensor, batch_size: int,
                     num_gt: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Get possible mask.

        Args:
            pred_bboxes (Tensor): Predict bboxes,
                shape(batch_size, num_priors, 4)
            pred_scores (Tensor): Scores of predict bbox,
                shape(batch_size, num_priors, num_classes)
            priors (Tensor): Model priors, shape (num_priors, 2)
            gt_labels (Tensor): Ground true labels,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground true bboxes,
                shape(batch_size, num_gt, 4)
            pad_bbox_flag (Tensor): Ground truth bbox mask,
                1 means bbox, 0 means no bbox,
                shape(batch_size, num_gt, 1)
            batch_size (int): Batch size.
            num_gt (int): Number of ground truth.
        Returns:
            pos_mask (Tensor): Possible mask,
                shape(batch_size, num_gt, num_priors)
            alignment_metrics (Tensor): Alignment metrics,
                shape(batch_size, num_gt, num_priors)
            overlaps (Tensor): Overlaps of gt_bboxes and pred_bboxes,
                shape(batch_size, num_gt, num_priors)
        """

        # Compute alignment metric between all bbox and gt
        alignment_metrics, overlaps = \
            self.get_box_metrics(pred_bboxes, pred_scores, gt_labels,
                                 gt_bboxes, batch_size, num_gt)

        # get is_in_gts mask
        is_in_gts = select_candidates_in_gts(priors, gt_bboxes)

        # get topk_metric mask
        topk_metric = self.select_topk_candidates(
            alignment_metrics * is_in_gts,
            topk_mask=pad_bbox_flag.repeat([1, 1, self.topk]).bool())

        # merge all mask to a final mask
        pos_mask = topk_metric * is_in_gts * pad_bbox_flag

        return pos_mask, alignment_metrics, overlaps

    def get_box_metrics(self, pred_bboxes: Tensor, pred_scores: Tensor,
                        gt_labels: Tensor, gt_bboxes: Tensor, batch_size: int,
                        num_gt: int) -> Tuple[Tensor, Tensor]:
        """Compute alignment metric between all bbox and gt.

        Args:
            pred_bboxes (Tensor): Predict bboxes,
                shape(batch_size, num_priors, 4)
            pred_scores (Tensor): Scores of predict bbox,
                shape(batch_size, num_priors, num_classes)
            gt_labels (Tensor): Ground true labels,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground true bboxes,
                shape(batch_size, num_gt, 4)
            batch_size (int): Batch size.
            num_gt (int): Number of ground truth.
        Returns:
            alignment_metrics (Tensor): Align metric,
                shape(batch_size, num_gt, num_priors)
            overlaps (Tensor): Overlaps, shape(batch_size, num_gt, num_priors)
        """
        pred_scores = pred_scores.permute(0, 2, 1)
        gt_labels = gt_labels.to(torch.long)
        idx = torch.zeros([2, batch_size, num_gt], dtype=torch.long)
        idx[0] = torch.arange(end=batch_size).view(-1, 1).repeat(1, num_gt)
        idx[1] = gt_labels.squeeze(-1)
        bbox_scores = pred_scores[idx[0], idx[1]]
        # TODO: need to replace the yolov6_iou_calculator function
        if self.use_ciou:
            overlaps = bbox_overlaps(
                pred_bboxes.unsqueeze(1),
                gt_bboxes.unsqueeze(2),
                iou_mode='ciou',
                bbox_format='xyxy').clamp(0)
        else:
            overlaps = yolov6_iou_calculator(gt_bboxes, pred_bboxes)

        alignment_metrics = bbox_scores.pow(self.alpha) * overlaps.pow(
            self.beta)

        return alignment_metrics, overlaps

    def select_topk_candidates(self,
                               alignment_gt_metrics: Tensor,
                               using_largest_topk: bool = True,
                               topk_mask: Optional[Tensor] = None) -> Tensor:
        """Compute alignment metric between all bbox and gt.

        Args:
            alignment_gt_metrics (Tensor): Alignment metric of gt candidates,
                shape(batch_size, num_gt, num_priors)
            using_largest_topk (bool): Controls whether to using largest or
                smallest elements.
            topk_mask (Tensor): Topk mask,
                shape(batch_size, num_gt, self.topk)
        Returns:
            Tensor: Topk candidates mask,
                shape(batch_size, num_gt, num_priors)
        """
        num_priors = alignment_gt_metrics.shape[-1]
        topk_metrics, topk_idxs = torch.topk(
            alignment_gt_metrics,
            self.topk,
            axis=-1,
            largest=using_largest_topk)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(axis=-1, keepdim=True) >
                         self.eps).tile([1, 1, self.topk])
        topk_idxs = torch.where(topk_mask, topk_idxs,
                                torch.zeros_like(topk_idxs))
        is_in_topk = F.one_hot(topk_idxs, num_priors).sum(axis=-2)
        is_in_topk = torch.where(is_in_topk > 1, torch.zeros_like(is_in_topk),
                                 is_in_topk)
        return is_in_topk.to(alignment_gt_metrics.dtype)

    def get_targets(self, gt_labels: Tensor, gt_bboxes: Tensor,
                    assigned_gt_idxs: Tensor, fg_mask_pre_prior: Tensor,
                    batch_size: int,
                    num_gt: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Get assigner info.

        Args:
            gt_labels (Tensor): Ground true labels,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground true bboxes,
                shape(batch_size, num_gt, 4)
            assigned_gt_idxs (Tensor): Assigned ground truth indexes,
                shape(batch_size, num_priors)
            fg_mask_pre_prior (Tensor): Force ground truth matching mask,
                shape(batch_size, num_priors)
            batch_size (int): Batch size.
            num_gt (int): Number of ground truth.
        Returns:
            assigned_labels (Tensor): Assigned labels,
                shape(batch_size, num_priors)
            assigned_bboxes (Tensor): Assigned bboxes,
                shape(batch_size, num_priors)
            assigned_scores (Tensor): Assigned scores,
                shape(batch_size, num_priors)
        """
        # assigned target labels
        batch_ind = torch.arange(
            end=batch_size, dtype=torch.int64, device=gt_labels.device)[...,
                                                                        None]
        assigned_gt_idxs = assigned_gt_idxs + batch_ind * num_gt
        assigned_labels = gt_labels.long().flatten()[assigned_gt_idxs]

        # assigned target boxes
        assigned_bboxes = gt_bboxes.reshape([-1, 4])[assigned_gt_idxs]

        # assigned target scores
        assigned_labels[assigned_labels < 0] = 0
        assigned_scores = F.one_hot(assigned_labels, self.num_classes)
        force_gt_scores_mask = fg_mask_pre_prior[:, :, None].repeat(
            1, 1, self.num_classes)
        assigned_scores = torch.where(force_gt_scores_mask > 0,
                                      assigned_scores,
                                      torch.full_like(assigned_scores, 0))

        return assigned_labels, assigned_bboxes, assigned_scores
