import glob
import os

import hydra
import numpy as np
from PIL import Image

# TODO: Get an estimate of % or error on each class


def pixel_accuracy(pred, label):
    correct = (pred == label).sum().item()
    total = label.size
    return correct / total


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
    precision_per_class = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        label_inds = label == cls
        true_positive = (pred_inds & label_inds).sum().item()
        false_positive = (pred_inds & ~label_inds).sum().item()

        if true_positive + false_positive == 0:
            precision_per_class.append(np.nan)
        else:
            precision_per_class.append(true_positive / (true_positive + false_positive))

    return np.nanmean(precision_per_class)


def recall(pred, label, num_classes):
    recall_per_class = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        label_inds = label == cls
        true_positive = (pred_inds & label_inds).sum().item()
        false_negative = (~pred_inds & label_inds).sum().item()

        if true_positive + false_negative == 0:
            recall_per_class.append(np.nan)
        else:
            recall_per_class.append(true_positive / (true_positive + false_negative))

    return np.nanmean(recall_per_class)


def apply_ignore_index(pred, label, ignore_value=-1):
    pred = np.where(label == ignore_value, ignore_value, pred)
    return pred, label


def evaluate_segmentation_metrics(pred_mask, gt_mask, num_classes=8):
    pred_mask = np.array(pred_mask)
    gt_mask = np.array(gt_mask)

    pa = pixel_accuracy(pred_mask, gt_mask)
    iou = intersection_over_union(pred_mask, gt_mask, num_classes)
    dice = dice_coefficient(pred_mask, gt_mask, num_classes)
    prec = precision(pred_mask, gt_mask, num_classes)
    rec = recall(pred_mask, gt_mask, num_classes)

    return {
        "Pixel Accuracy": pa,
        "Mean IoU": iou,
        "Mean Dice Coefficient": dice,
        "Precision": prec,
        "Recall": rec,
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

    for key in metrics_list[0].keys():
        avg_metrics[key] = sum(d[key] for d in metrics_list) / num_metrics

    return avg_metrics


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    pred_folder = cfg.paths.PRED_TEST_MASKS
    gt_folder = cfg.paths.GT_TEST_MASKS
    num_classes = 8
    ignore_value = -1

    metrics_list = process_folders(pred_folder, gt_folder, num_classes, ignore_value)
    avg_metrics = aggregate_metrics(metrics_list)

    print("Average Metrics for all masks:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
