
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from pycocotools.coco import COCO
from recall import bbox_overlaps


# refer from https://github.com/henghuiding/gRefCOCO/blob/main/mdetr/datasets/refexp.py # noqa

class gRefCOCOMetric():
    default_prefix: Optional[str] = 'grefcoco'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: str = 'bbox',
                 iou_thrs: float = 0.5,
                 thresh_score: float = 0.7,
                 thresh_f1: float = 1.0,
                 ) -> None:
        super().__init__()
        self.metric = metric
        self.iou_thrs = iou_thrs
        self.thresh_score = thresh_score
        self.thresh_f1 = thresh_f1

        self.coco = COCO(ann_file)

    def compute_metrics(self, results: list) -> Dict[str, float]:

        correct_image = 0
        num_image = 0
        nt = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

        for result in results:
            img_id = result['img_id']
            TP = 0

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids[0])

            converted_bbox_all = []
            no_target_flag = False
            for one_target in target:
                if one_target['category_id'] == -1:
                    no_target_flag = True
                target_bbox = one_target['bbox']
                converted_bbox = [
                    target_bbox[0],
                    target_bbox[1],
                    target_bbox[2] + target_bbox[0],
                    target_bbox[3] + target_bbox[1],
                ]
                converted_bbox_all.append(
                    np.array(converted_bbox).reshape(-1, 4))
            gt_bbox_all = np.concatenate(converted_bbox_all, axis=0)

            # idx = result['scores'] >= self.thresh_score
            # filtered_boxes = result['bboxes'][idx]
            filtered_boxes = torch.tensor(result['bboxes'])

            iou = bbox_overlaps(filtered_boxes.numpy(), gt_bbox_all)
            iou = torch.from_numpy(iou)

            num_prediction = filtered_boxes.shape[0]
            num_gt = gt_bbox_all.shape[0]
            if no_target_flag:
                if num_prediction >= 1:
                    nt['FN'] += 1
                else:
                    nt['TP'] += 1
                if num_prediction >= 1:
                    f_1 = 0.
                else:
                    f_1 = 1.0
            else:
                if num_prediction >= 1:
                    nt['TN'] += 1
                else:
                    nt['FP'] += 1
                for i in range(min(num_prediction, num_gt)):
                    top_value, top_index = torch.topk(iou.flatten(0, 1), 1)
                    if top_value < self.iou_thrs:
                        break
                    else:
                        top_index_x = top_index // num_gt
                        top_index_y = top_index % num_gt
                        TP += 1
                        iou[top_index_x[0], :] = 0.0
                        iou[:, top_index_y[0]] = 0.0
                FP = num_prediction - TP
                FN = num_gt - TP
                f_1 = 2 * TP / (2 * TP + FP + FN)

            if f_1 >= self.thresh_f1:
                correct_image += 1
            num_image += 1

        score = correct_image / max(num_image, 1)
        results = {
            'F1_score': score,
            'T_acc': nt['TN'] / (nt['TN'] + nt['FP']),
            'N_acc': nt['TP'] / (nt['TP'] + nt['FN'])
        }
        # logger.info(results)
        return results
