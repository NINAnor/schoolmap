import glob
import os

import hydra
import numpy as np
from PIL import Image

from utils.extras import LABELS


def pixel_accuracy(pred, label, num_classes):
    correct_per_class = np.zeros(num_classes)
    total_per_class = np.zeros(num_classes)

    for cls in range(num_classes):
        cls_mask = label == cls
        correct_per_class[cls] = np.sum((pred == cls) & cls_mask)
        total_per_class[cls] = np.sum(cls_mask)

    class_wise_accuracy = {
        cls: correct_per_class[cls] / total_per_class[cls]
        if total_per_class[cls] > 0
        else np.nan
        for cls in range(num_classes)
    }

    overall_accuracy = correct_per_class.sum() / total_per_class.sum()

    return overall_accuracy, class_wise_accuracy


def intersection_over_union(pred, label, num_classes):
    iou_per_class = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        label_inds = label == cls
        intersection = (pred_inds & label_inds).sum().item()
        union = (pred_inds | label_inds).sum().item()

        if union == 0:
            iou_per_class.append(np.nan)
        else:
            iou_per_class.append(intersection / union)

    return np.nanmean(iou_per_class)


def dice_coefficient(pred_mask, gt_mask, num_classes):
    dice_scores = []
    for c in range(num_classes):
        pred_inds = pred_mask == c
        label_inds = gt_mask == c

        intersection = (pred_inds & label_inds).sum().item()
        denominator = pred_inds.sum().item() + label_inds.sum().item()

        if denominator == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2 * intersection) / denominator

        dice_scores.append(dice)

    return np.mean(dice_scores)


def precision(pred, label, num_classes):
    precision_per_class = {}
    for cls in range(num_classes):
        pred_inds = pred == cls
        label_inds = label == cls
        true_positive = (pred_inds & label_inds).sum().item()
        false_positive = (pred_inds & ~label_inds).sum().item()

        if true_positive + false_positive == 0:
            precision_per_class[cls] = np.nan
        else:
            precision_per_class[cls] = true_positive / (true_positive + false_positive)

    return precision_per_class


def recall(pred, label, num_classes):
    recall_per_class = {}
    for cls in range(num_classes):
        pred_inds = pred == cls
        label_inds = label == cls
        true_positive = (pred_inds & label_inds).sum().item()
        false_negative = (~pred_inds & label_inds).sum().item()

        if true_positive + false_negative == 0:
            recall_per_class[cls] = np.nan
        else:
            recall_per_class[cls] = true_positive / (true_positive + false_negative)

    return recall_per_class


def apply_ignore_index(pred, label, ignore_value=-1):
    pred = np.where(label == ignore_value, ignore_value, pred)
    return pred, label


def evaluate_segmentation_metrics(pred_mask, gt_mask, num_classes):
    pred_mask = np.array(pred_mask)
    gt_mask = np.array(gt_mask)

    overall_pa, class_wise_pa = pixel_accuracy(pred_mask, gt_mask, num_classes)
    iou = intersection_over_union(pred_mask, gt_mask, num_classes)
    dice = dice_coefficient(pred_mask, gt_mask, num_classes)
    prec_per_class = precision(pred_mask, gt_mask, num_classes)
    rec_per_class = recall(pred_mask, gt_mask, num_classes)

    avg_precision = np.nanmean(list(prec_per_class.values()))
    avg_recall = np.nanmean(list(rec_per_class.values()))

    return {
        "Pixel Accuracy (Overall)": overall_pa,
        "Class-wise Pixel Accuracy": class_wise_pa,
        "Mean IoU": iou,
        "Mean Dice Coefficient": dice,
        "Precision": avg_precision,
        "Recall": avg_recall,
        "Class-wise Precision": prec_per_class,
        "Class-wise Recall": rec_per_class,
    }


def process_folders(pred_folder, gt_folder, num_classes=8, ignore_value=-1):
    metrics_list = []

    pred_files = sorted(glob.glob(pred_folder + "/*.tif"))
    gt_files = sorted(glob.glob(gt_folder + "/*.tif"))

    for pred_file, gt_file in zip(pred_files, gt_files):
        pred_mask = np.array(Image.open(os.path.join(pred_folder, pred_file)))
        gt_mask = np.array(Image.open(os.path.join(gt_folder, gt_file)))

        pred_mask, gt_mask = apply_ignore_index(pred_mask, gt_mask, ignore_value)

        metrics = evaluate_segmentation_metrics(pred_mask, gt_mask, num_classes)
        metrics_list.append(metrics)

    return metrics_list


def aggregate_metrics(metrics_list):
    avg_metrics = {}
    num_metrics = len(metrics_list)

    # Aggregate non-class-wise metrics
    for key in [
        "Pixel Accuracy (Overall)",
        "Mean IoU",
        "Mean Dice Coefficient",
        "Precision",
        "Recall",
    ]:
        avg_metrics[key] = sum(d[key] for d in metrics_list) / num_metrics

    # Aggregate class-wise metrics
    class_precision = {
        cls: [] for cls in metrics_list[0]["Class-wise Precision"].keys()
    }
    class_recall = {cls: [] for cls in metrics_list[0]["Class-wise Recall"].keys()}
    class_pixel_accuracy = {
        cls: [] for cls in metrics_list[0]["Class-wise Pixel Accuracy"].keys()
    }

    for metrics in metrics_list:
        for cls, value in metrics["Class-wise Precision"].items():
            class_precision[cls].append(value)
        for cls, value in metrics["Class-wise Recall"].items():
            class_recall[cls].append(value)
        for cls, value in metrics["Class-wise Pixel Accuracy"].items():
            class_pixel_accuracy[cls].append(value)

    avg_metrics["Class-wise Precision"] = {
        cls: np.nanmean(values) for cls, values in class_precision.items()
    }
    avg_metrics["Class-wise Recall"] = {
        cls: np.nanmean(values) for cls, values in class_recall.items()
    }
    avg_metrics["Class-wise Pixel Accuracy"] = {
        cls: np.nanmean(values) for cls, values in class_pixel_accuracy.items()
    }

    return avg_metrics


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    pred_folder = cfg.paths.PRED_TEST_MASKS
    gt_folder = cfg.paths.GT_TEST_MASKS
    num_classes = cfg.train.NUM_CLASSES
    ignore_value = -1

    metrics_list = process_folders(pred_folder, gt_folder, num_classes, ignore_value)
    avg_metrics = aggregate_metrics(metrics_list)

    print("Average Metrics for all masks:")
    for metric, value in avg_metrics.items():
        if metric not in [
            "Class-wise Precision",
            "Class-wise Recall",
            "Class-wise Pixel Accuracy",
        ]:
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}:")
            for cls, cls_value in value.items():
                cls_name = LABELS[cls]["name"]
                print(f"  Class {cls_name}: {cls_value:.4f}")


if __name__ == "__main__":
    main()
