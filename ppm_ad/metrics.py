from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _safe(metric, y_true, y_score) -> float:
    try:
        return float(metric(y_true, y_score))
    except ValueError:
        return float("nan")


def best_pixel_metrics(masks: np.ndarray, scores: np.ndarray):
    truth = masks.astype(bool).ravel()
    values = scores.ravel()
    precision, recall, thresholds = precision_recall_curve(truth, values)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    index = int(np.nanargmax(f1))
    threshold = thresholds[min(index, len(thresholds) - 1)] if len(thresholds) else 0.0
    prediction = values >= threshold
    intersection = np.logical_and(prediction, truth).sum()
    union = np.logical_or(prediction, truth).sum()
    return float(f1[index]), float(intersection / max(union, 1)), float(threshold)


def aupro(masks: np.ndarray, scores: np.ndarray, max_fpr: float = 0.3, steps: int = 100) -> float:
    masks = masks.astype(bool)
    thresholds = np.linspace(float(scores.max()), float(scores.min()), steps)
    normal_pixels = (~masks).sum()
    regions: list[tuple[int, np.ndarray]] = []
    for image_index, mask in enumerate(masks):
        components, count = ndimage.label(mask)
        flat_components = components.ravel()
        for component_id in range(1, count + 1):
            regions.append((image_index, np.flatnonzero(flat_components == component_id)))
    points = []
    for threshold in thresholds:
        prediction = scores >= threshold
        fpr = np.logical_and(prediction, ~masks).sum() / max(normal_pixels, 1)
        flat_prediction = prediction.reshape(len(prediction), -1)
        overlaps = [flat_prediction[i, region].mean() for i, region in regions]
        points.append((fpr, float(np.mean(overlaps)) if overlaps else 0.0))
    points = np.asarray(sorted(points))
    keep = points[:, 0] <= max_fpr
    if keep.sum() < 2:
        return 0.0
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(points[keep, 1], points[keep, 0]) / max_fpr)


def fragmentation(masks: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    ratios = []
    for score in scores:
        binary = score >= threshold
        _, count = ndimage.label(binary)
        ratios.append(count / max(binary.sum(), 1))
    return float(np.mean(ratios))


def evaluate(labels, image_scores, masks, pixel_scores) -> dict[str, float]:
    labels = np.asarray(labels)
    image_scores = np.asarray(image_scores)
    masks = np.asarray(masks).squeeze(1)
    pixel_scores = np.asarray(pixel_scores)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        max_f1, iou, threshold = best_pixel_metrics(masks, pixel_scores)
        return {
            "image_auroc": _safe(roc_auc_score, labels, image_scores),
            "image_ap": _safe(average_precision_score, labels, image_scores),
            "pixel_auroc": _safe(roc_auc_score, masks.ravel(), pixel_scores.ravel()),
            "pixel_ap": _safe(average_precision_score, masks.ravel(), pixel_scores.ravel()),
            "aupro_0.3": aupro(masks, pixel_scores),
            "max_f1": max_f1,
            "iou_at_best_f1": iou,
            "fragmentation": fragmentation(masks, pixel_scores, threshold),
            "pixel_threshold": threshold,
        }

