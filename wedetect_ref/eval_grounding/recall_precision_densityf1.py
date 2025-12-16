import argparse
import json
import os
from collections import defaultdict

import numpy as np
from pycocotools import mask as mask_utils
from tabulate import tabulate
from tqdm import tqdm


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (box1_area + box2_area - intersection)


def calculate_metrics(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Calculate recall and precision for given boxes"""
    # Special handling for rejection domain
    if len(gt_boxes) == 0:
        # For rejection cases, pred_boxes should also be empty
        return 1.0, 1.0 if len(pred_boxes) == 0 else 0.0, 0.0

    # Original logic for other cases
    if len(pred_boxes) == 0:
        return 0.0, 0.0
    if len(pred_boxes) == 1 and len(pred_boxes[0]) == 0:
        return 0.0, 0.0

    matches = 0
    used_preds = set()

    for gt_box in gt_boxes:
        best_iou = 0
        best_pred_idx = -1

        for i, pred_box in enumerate(pred_boxes):
            if i in used_preds:
                continue
            iou = calculate_iou(gt_box, pred_box)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_pred_idx = i

        if best_pred_idx != -1:
            matches += 1
            used_preds.add(best_pred_idx)

    recall = matches / len(gt_boxes)
    precision = matches / len(pred_boxes)

    return recall, precision


def is_point_prediction(pred):
    """Check if prediction is a point (2 values) or box (4 values)"""
    return len(pred) == 2


def point_in_mask(point, mask_dict):
    """Check if point falls inside the mask using COCO RLE format"""
    binary_mask = mask_utils.decode(mask_dict)
    h, w = binary_mask.shape
    x, y = int(round(point[0])), int(round(point[1]))
    if 0 <= x < w and 0 <= y < h:
        return binary_mask[y, x] == 1
    return False


def calculate_point_metrics(gt_boxes, gt_masks, pred_points):
    """Calculate metrics for point predictions"""
    if len(gt_boxes) == 0 and len(pred_points) == 0:
        return 1.0, 1.0
    if len(gt_boxes) == 0:
        return 0.0, 0.0
    if len(pred_points) == 0:
        return 0.0, 1.0

    matches = 0
    used_preds = set()

    for gt_idx, gt_mask in enumerate(gt_masks):
        for i, point in enumerate(pred_points):
            if i in used_preds:
                continue
            if point_in_mask(point, gt_mask):
                matches += 1
                used_preds.add(i)
                break

    recall = matches / len(gt_boxes)
    precision = matches / len(pred_points)

    return recall, precision


def calculate_density_f1(precision, recall, gt_count, pred_count, total_persons):
    """Calculate density-aware F1 score using density ratio

    Args:
        precision (float): Precision score
        recall (float): Recall score
        gt_count (int): Number of ground truth boxes
        pred_count (int): Number of predicted boxes
        total_persons (int): Total number of persons in the image (from candidate_boxes)
    """
    # Calculate standard F1 first
    if precision + recall == 0:
        return 0.0
    standard_f1 = 2 * (precision * recall) / (precision + recall)

    # Calculate density ratios
    gt_density_ratio = gt_count / total_persons if total_persons > 0 else 0.0
    pred_density_ratio = pred_count / total_persons if total_persons > 0 else 0.0

    # Calculate density penalty
    if pred_density_ratio > 0:
        penalty = min(1.0, gt_density_ratio / pred_density_ratio)
    else:
        penalty = 0.0

    return standard_f1 * penalty


def evaluate_dataset(gt_data, pred_data):
    """Evaluate the entire dataset and return metrics"""
    domain_metrics = defaultdict(list)
    subdomain_metrics = defaultdict(list)
    box_count_metrics = defaultdict(list)
    rejection_scores = {"total": 0, "correct": 0}  # Track rejection performance

    is_point = False

    pred_lookup = {item["id"]: item["extracted_predictions"] for item in pred_data}

    for gt_item in tqdm(gt_data, desc="Evaluating predictions"):
        gt_boxes = gt_item["answer_boxes"]
        total_persons = len(gt_item["candidate_boxes"])
        pred = pred_lookup.get(gt_item["id"], [])

        # Special handling for rejection domain
        if gt_item["domain"] == "rejection":
            rejection_scores["total"] += 1
            try:
                if len(pred) == 0 or len(pred[0]) == 0:  # Correct rejection
                    rejection_scores["correct"] += 1
                continue  # Skip regular metrics for rejection domain
            except:
                raise ValueError(f"{pred}")
        box_count = len(gt_boxes)
        box_count_range = get_box_count_range(box_count)

        if is_point:
            recall, precision = calculate_point_metrics(
                gt_boxes, gt_item["answer_segmentations"], pred
            )
            density_f1 = calculate_density_f1(
                precision, recall, len(gt_boxes), len(pred), total_persons
            )
            metrics_tuple = (
                recall,
                precision,
                total_persons,
                len(pred),
                len(gt_boxes),
                density_f1,
            )
        else:
            recalls = []
            precisions = []
            density_f1s = []
            for iou_thresh in np.arange(0.5, 1.0, 0.05):
                recall, precision = calculate_metrics(gt_boxes, pred, iou_thresh)
                recalls.append(recall)
                precisions.append(precision)
                density_f1 = calculate_density_f1(
                    precision, recall, len(gt_boxes), len(pred), total_persons
                )
                density_f1s.append(density_f1)

            recall_50, precision_50 = calculate_metrics(gt_boxes, pred, 0.5)
            recall_mean = np.mean(recalls)
            precision_mean = np.mean(precisions)
            density_f1_50 = calculate_density_f1(
                precision_50, recall_50, len(gt_boxes), len(pred), total_persons
            )
            density_f1_mean = np.mean(density_f1s)
            metrics_tuple = (
                recall_50,
                precision_50,
                recall_mean,
                precision_mean,
                total_persons,
                len(pred),
                len(gt_boxes),
                density_f1_50,
                density_f1_mean,
            )

        domain_metrics[gt_item["domain"]].append(metrics_tuple)
        subdomain_metrics[gt_item["sub_domain"]].append(metrics_tuple)
        box_count_metrics[box_count_range].append(metrics_tuple)

    return {
        "domain": domain_metrics,
        "subdomain": subdomain_metrics,
        "box_count": box_count_metrics,
        "is_point": is_point,
        "rejection_score": rejection_scores[
            "correct"
        ],  # Return absolute number of correct rejections
    }


def get_box_count_range(count):
    """Get the range category for a given box count"""
    if count == 1:
        return "1"
    elif 2 <= count <= 5:
        return "2-5"
    elif 6 <= count <= 10:
        return "6-10"
    else:
        return ">10"


def format_row_data(metrics, model_name, is_point):
    """Helper function to format row data based on prediction type"""
    if is_point:
        # For point predictions: (recall, precision, total_persons, pred_count, gt_count, density_f1)
        recalls, precisions, _, _, _, density_f1s = zip(*metrics)
        recall = float(np.mean(recalls))
        precision = float(np.mean(precisions))
        density_f1 = float(np.mean(density_f1s))

        return {
            "recall": recall,
            "precision": precision,
            "density_f1": density_f1,
            "count": len(metrics),
            "model_name": model_name,
            "is_point": True,
        }
    else:
        # For box predictions: (recall_50, precision_50, recall_mean, precision_mean,
        #                      total_persons, pred_count, gt_count, density_f1_50, density_f1_mean)
        (
            recalls_50,
            precisions_50,
            recalls_mean,
            precisions_mean,
            _,
            _,
            _,
            density_f1_50s,
            density_f1_means,
        ) = zip(*metrics)

        recall_50 = float(np.mean(recalls_50))
        precision_50 = float(np.mean(precisions_50))
        recall_mean = float(np.mean(recalls_mean))
        precision_mean = float(np.mean(precisions_mean))
        density_f1_50 = float(np.mean(density_f1_50s))
        density_f1_mean = float(np.mean(density_f1_means))

        return {
            "recall_50": recall_50,
            "recall_mean": recall_mean,
            "precision_50": precision_50,
            "precision_mean": precision_mean,
            "density_f1_50": density_f1_50,
            "density_f1_mean": density_f1_mean,
            "count": len(metrics),
            "model_name": model_name,
            "is_point": False,
        }


def format_row(data, show_name=True, category_name="", subdomain_name=""):
    """Helper function to format a table row"""
    row = [
        category_name if show_name else "",  # Domain
        subdomain_name if show_name else "",  # Subdomain
        data["model_name"],  # Model name
    ]

    if data["is_point"]:
        # Point metrics
        row.extend(
            [
                f"{data['recall']:.3f}",
                f"{data['precision']:.3f}",
                f"{data['density_f1']:.3f}",  # F1 for point predictions
                "",  # Empty cell for Recall@0.5
                "",  # Empty cell for Recall@0.5:0.95
                "",  # Empty cell for Precision@0.5
                "",  # Empty cell for Precision@0.5:0.95
                "",  # Empty cell for F1@0.5
                "",  # Empty cell for F1@0.5:0.95
                "",  # Empty cell for Rejection Score
            ]
        )
    else:
        # Box metrics
        row.extend(
            [
                "",  # Empty cell for Recall@Point
                "",  # Empty cell for Precision@Point
                "",  # Empty cell for F1@Point
                f"{data['recall_50']:.3f}",
                f"{data['recall_mean']:.3f}",
                f"{data['precision_50']:.3f}",
                f"{data['precision_mean']:.3f}",
                f"{data['density_f1_50']:.3f}",
                f"{data['density_f1_mean']:.3f}",
                "",  # Empty cell for Rejection Score
            ]
        )

    # Add count as the last column
    row.append(data["count"])

    return row


def convert_table_to_json(rows, headers):
    """Convert table rows to structured JSON format while preserving all information"""
    json_data = []
    current_domain = None
    current_subdomain = None
    current_box_count = None

    # Create a mapping for header keys that need special handling
    header_key_map = {
        "Box Count": "box count",  # Keep space, don't convert to underscore
        "Box\nCount": "box count",  # Handle newline case
        "Rejection\nScore": "rejection_score",  # Handle rejection score
    }

    for row in rows:
        if all(cell == "-" * 10 for cell in row):  # Skip separator rows
            continue

        entry = {}
        for i, header in enumerate(headers):
            if i >= len(row):  # Skip if row is shorter than headers
                continue
            # Use special mapping for certain headers, otherwise use default transformation
            header_key = header_key_map.get(header, header.replace("\n", "_").lower())
            value = row[i]

            # Update tracking variables
            if header_key == "domain" and value:
                current_domain = value
            elif header_key == "subdomain" and value:
                current_subdomain = value
            elif header_key == "box count" and value:
                current_box_count = value

            # Use tracked values when current row value is empty
            if value == "":
                if header_key == "domain":
                    value = current_domain
                elif header_key == "subdomain":
                    value = current_subdomain
                elif header_key == "box count":
                    value = current_box_count

            # Keep box count as string, convert other numeric strings to float
            if isinstance(value, str):
                if header_key != "box count":
                    try:
                        if "." in value:
                            value = float(value)
                        elif value.isdigit():
                            value = int(value)
                    except ValueError:
                        pass

            entry[header_key] = value

        json_data.append(entry)
    return json_data


def dump_tables(domain_rows, box_rows, domain_headers, box_headers, dump_dir):
    """Dump tables to markdown and JSON files"""
    # Create directory if it doesn't exist
    os.makedirs(dump_dir, exist_ok=True)

    # Prepare markdown content
    md_content = "# Evaluation Results\n\n"
    md_content += "## Comparative Domain and Subdomain Metrics\n\n"
    md_content += tabulate(domain_rows, headers=domain_headers, tablefmt="pipe")
    md_content += "\n\n"
    md_content += "## Comparative Box Count Metrics\n\n"
    md_content += tabulate(box_rows, headers=box_headers, tablefmt="pipe")

    # Prepare JSON content
    domain_metrics = convert_table_to_json(domain_rows, domain_headers)
    box_metrics = convert_table_to_json(box_rows, box_headers)

    # Fix box count format in box_metrics
    for entry in box_metrics:
        if "box_count" in entry:
            # Convert numeric box count to range string if needed
            if isinstance(entry["box_count"], (int, float)):
                entry["box count"] = get_box_count_range(int(entry["box_count"]))
            elif entry["box_count"] == "":
                # Use the previous valid box count
                continue
            # Move from box_count to "box count"
            entry["box count"] = entry.pop("box_count")

    json_content = {
        "domain_subdomain_metrics": domain_metrics,
        "box_count_metrics": box_metrics,
    }

    # Write markdown file
    md_path = os.path.join(dump_dir, "comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Write JSON file
    json_path = os.path.join(dump_dir, "comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_content, f, indent=2, ensure_ascii=False)


def get_all_box_ranges(all_metrics):
    """Get all unique box count ranges from all metrics"""
    ranges = set()
    for metrics in all_metrics.values():
        ranges.update(metrics["box_count"].keys())
    return ranges


def print_comparative_metrics(all_metrics, gt_data, dump_path=None):
    """Print comparative metrics for all models in same tables"""
    model_types = {
        model_name: metrics["is_point"] for model_name, metrics in all_metrics.items()
    }

    # Get all unique domains
    all_domains = set()
    for metrics in all_metrics.values():
        all_domains.update(metrics["domain"].keys())

    # Prepare headers
    headers = [
        "Domain",
        "Subdomain",
        "Model",
        "Recall\n@Point",
        "Precision\n@Point",
        "DensityF1\n@Point",
        "Recall\n@0.5",
        "Recall\n@0.5:0.95",
        "Precision\n@0.5",
        "Precision\n@0.5:0.95",
        "DensityF1\n@0.5",
        "DensityF1\n@0.5:0.95",
        "Rejection\nScore",
        "Samples",
    ]

    # Process domain and subdomain metrics
    domain_rows = []
    for domain in sorted(all_domains):
        # Process domain metrics
        domain_metrics = []
        for model_name, metrics in all_metrics.items():
            if domain in metrics["domain"]:
                values = metrics["domain"][domain]
                domain_metrics.append(
                    format_row_data(values, model_name, model_types[model_name])
                )

        # Sort domain metrics by DensityF1@0.5:0.95 score
        domain_metrics.sort(
            reverse=True,
            key=lambda x: x["density_f1"] if x["is_point"] else x["density_f1_mean"],
        )

        # Add domain rows
        for i, metrics in enumerate(domain_metrics):
            domain_rows.append(format_row(metrics, i == 0, domain, "Overall"))

        # Process subdomains
        all_subdomains = set()
        for metrics in all_metrics.values():
            for subdomain in metrics["subdomain"].keys():
                first_sample = next(
                    gt for gt in gt_data if gt["sub_domain"] == subdomain
                )
                if first_sample["domain"] == domain:
                    all_subdomains.add(subdomain)

        for subdomain in sorted(all_subdomains):
            subdomain_metrics = []
            for model_name, metrics in all_metrics.items():
                if subdomain in metrics["subdomain"]:
                    values = metrics["subdomain"][subdomain]
                    subdomain_metrics.append(
                        format_row_data(values, model_name, model_types[model_name])
                    )

            # Sort subdomain metrics by DensityF1@0.5:0.95 score
            subdomain_metrics.sort(
                reverse=True,
                key=lambda x: (
                    x["density_f1"] if x["is_point"] else x["density_f1_mean"]
                ),
            )

            # Add subdomain rows
            for i, metrics in enumerate(subdomain_metrics):
                row = format_row(metrics, i == 0, "", subdomain)
                domain_rows.append(row)

    # Add separator before averages
    domain_rows.append(["-" * 10] * len(headers))

    # Add overall averages
    average_metrics = []
    for model_name, metrics in all_metrics.items():
        all_values = []
        for values in metrics["domain"].values():
            all_values.extend(values)

        avg_metrics = format_row_data(all_values, model_name, model_types[model_name])
        average_metrics.append(avg_metrics)

    # Sort average metrics by DensityF1@0.5:0.95 score
    average_metrics.sort(
        reverse=True,
        key=lambda x: x["density_f1"] if x["is_point"] else x["density_f1_mean"],
    )

    # Add sorted average rows
    for metrics in average_metrics:
        row = format_row(metrics, True, "Average", "All Data")
        domain_rows.append(row)

    # Add rejection score to the metrics
    domain_rows.append(["-" * 10] * len(headers))
    for model_name, metrics in all_metrics.items():
        if "rejection_score" in metrics:
            row = [
                "Rejection",  # Domain
                "Overall",  # Subdomain
                model_name,  # Model name
                "",  # Recall@Point
                "",  # Precision@Point
                "",  # DensityF1@Point
                "",  # Recall@0.5
                "",  # Recall@0.5:0.95
                "",  # Precision@0.5
                "",  # Precision@0.5:0.95
                "",  # DensityF1@0.5
                "",  # DensityF1@0.5:0.95
                str(metrics["rejection_score"]),  # Rejection Score
                "",  # Samples
            ]
            domain_rows.append(row)

    # Generate tables
    domain_table = tabulate(domain_rows, headers=headers, tablefmt="grid")
    print("\nComparative Domain and Subdomain Metrics:")
    print(domain_table)

    # Process box count metrics
    box_headers = [
        "Box Count",
        "Model",
        "Recall\n@Point",
        "Precision\n@Point",
        "DensityF1\n@Point",
        "Recall\n@0.5",
        "Recall\n@0.5:0.95",
        "Precision\n@0.5",
        "Precision\n@0.5:0.95",
        "DensityF1\n@0.5",
        "DensityF1\n@0.5:0.95",
        "Rejection\nScore",
        "Samples",
    ]

    box_rows = []
    # Sort range keys in a logical order
    range_order = {"1": 0, "2-5": 1, "6-10": 2, ">10": 3}
    for range_key in sorted(
        get_all_box_ranges(all_metrics), key=lambda x: range_order.get(x, 999)
    ):
        range_metrics = []
        for model_name, metrics in all_metrics.items():
            if range_key in metrics["box_count"]:
                values = metrics["box_count"][range_key]
                range_metrics.append(
                    format_row_data(values, model_name, model_types[model_name])
                )

        # Sort by F1 score
        range_metrics.sort(
            reverse=True,
            key=lambda x: x["density_f1"] if x["is_point"] else x["density_f1_mean"],
        )

        # Add rows with range key only for first model
        for i, metrics in enumerate(range_metrics):
            row = format_row(metrics, i == 0, range_key, "")
            row.pop(1)  # Remove subdomain column for box count metrics
            box_rows.append(row)

    # Add separator before averages
    box_rows.append(["-" * 10] * len(box_headers))

    # Add overall averages
    average_metrics = []
    for model_name, metrics in all_metrics.items():
        all_values = []
        for values in metrics["box_count"].values():
            all_values.extend(values)

        avg_metrics = format_row_data(all_values, model_name, model_types[model_name])
        average_metrics.append(avg_metrics)

    # Sort average metrics by DensityF1@0.5:0.95 score
    average_metrics.sort(
        reverse=True,
        key=lambda x: x["density_f1"] if x["is_point"] else x["density_f1_mean"],
    )

    # Add sorted average rows
    for metrics in average_metrics:
        row = format_row(metrics, True, "Average", "")
        row.pop(1)  # Remove subdomain column for box count metrics
        box_rows.append(row)

    box_table = tabulate(box_rows, box_headers, tablefmt="grid")
    print("\nComparative Box Count Metrics:")
    print(box_table)

    # Dump tables if path is provided
    if dump_path:
        dump_tables(domain_rows, box_rows, headers, box_headers, dump_path)


def recall_precision_densityf1(gt_path, pred_path, dump_path=None):
    # Load ground truth data
    gt_data = [json.loads(line) for line in open(gt_path, "r")]

    # Process prediction files
    all_metrics = {}
    pred_names = ["Model_1"]

    # Ensure we have matching names for all prediction files
    if len(pred_names) < len(pred_path):
        pred_names.extend(
            [f"Model_{i+1}" for i in range(len(pred_names), len(pred_path))]
        )

    # Calculate metrics for each prediction file
    for pred_path, pred_name in zip(pred_path, pred_names):
        pred_data = [json.loads(line) for line in open(pred_path, "r")]
        all_metrics[pred_name] = evaluate_dataset(gt_data, pred_data)

    if dump_path is not None:
        os.path.makedirs(os.path.dirname(dump_path), exist_ok=True)
        # Print results with all models in same tables and optionally dump to file
        print_comparative_metrics(all_metrics, gt_data, dump_path)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--gt_path",
#         type=str,
#         default="IDEA-Research/HumanRef/annotations.jsonl",
#         help="Path to ground truth JSONL file",
#     )
#     parser.add_argument(
#         "--pred_path",
#         type=str,
#         nargs="+",
#         help="Path(s) to prediction JSONL file(s)",
#         default=[
#             "IDEA-Research/HumanRef/evaluation_results/eval_deepseekvl2/deepseekvl2_small_results.jsonl",
#             "IDEA-Research/HumanRef/evaluation_results/eval_ferret/ferret7b_results.jsonl",
#             "IDEA-Research/HumanRef/evaluation_results/eval_groma/groma7b_results.jsonl",
#             "IDEA-Research/HumanRef/evaluation_results/eval_internvl2/internvl2.5_8b_results.jsonl",
#             "IDEA-Research/HumanRef/evaluation_results/eval_shikra/shikra7b_results.jsonl",
#             "IDEA-Research/HumanRef/evaluation_results/eval_molmo/molmo-7b-d-0924_results.jsonl",
#             "IDEA-Research/HumanRef/evaluation_results/eval_qwen2vl/qwen2.5-7B.jsonl",
#             "IDEA-Research/HumanRef/evaluation_results/eval_chatrex/ChatRex-Vicuna7B.jsonl",
#             "IDEA-Research/HumanRef/evaluation_results/eval_dinox/dinox_results.jsonl",
#             "IDEA-Research/HumanRef/evaluation_results/eval_rexseek/rexseek_7b.jsonl",
#             "IDEA-Research/HumanRef/evaluation_results/eval_full_gt_person/results.jsonl",
#         ],
#     )
#     parser.add_argument(
#         "--pred_names",
#         type=str,
#         nargs="+",
#         default=[
#             "DeepSeek-VL2-small",
#             "Ferret-7B",
#             "Groma-7B",
#             "InternVl-2.5-8B",
#             "Shikra-7B",
#             "Molmo-7B-D-0924",
#             "Qwen2.5-VL-7B",
#             "ChatRex-7B",
#             "DINOX",
#             "RexSeek-7B",
#             "Baseline",
#         ],
#         help="Name(s) for prediction files (optional)",
#     )
#     parser.add_argument(
#         "--dump_path",
#         type=str,
#         default="IDEA-Research/HumanRef/evaluation_results/compare_different_models",
#         help="Directory to dump markdown and JSON results (optional)",
#     )
#     args = parser.parse_args()

#     # Load ground truth data
#     gt_data = [json.loads(line) for line in open(args.gt_path, "r")]

#     # Process prediction files
#     all_metrics = {}
#     pred_names = (
#         args.pred_names
#         if args.pred_names
#         else [f"Model_{i+1}" for i in range(len(args.pred_path))]
#     )

#     # Ensure we have matching names for all prediction files
#     if len(pred_names) < len(args.pred_path):
#         pred_names.extend(
#             [f"Model_{i+1}" for i in range(len(pred_names), len(args.pred_path))]
#         )

#     # Calculate metrics for each prediction file
#     for pred_path, pred_name in zip(args.pred_path, pred_names):
#         pred_data = [json.loads(line) for line in open(pred_path, "r")]
#         all_metrics[pred_name] = evaluate_dataset(gt_data, pred_data)

#     # Print results with all models in same tables and optionally dump to file
#     print_comparative_metrics(all_metrics, gt_data, args.dump_path)


# if __name__ == "__main__":
#     main()
