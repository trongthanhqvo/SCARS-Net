from __future__ import annotations

import numpy as np


def recording_binary_scores(
    window_labels: np.ndarray,
    recording_ids: np.ndarray,
    window_probabilities: np.ndarray,
    classes: np.ndarray,
    background_label: str = "background",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes = np.asarray(classes)
    background = np.flatnonzero(classes.astype(str) == background_label)
    if len(background) != 1:
        raise ValueError("Detection requires exactly one canonical background class")
    output_y, output_scores, output_recordings = [], [], []
    for recording_id in sorted(set(recording_ids.tolist()), key=str):
        mask = recording_ids == recording_id
        labels = np.unique(window_labels[mask])
        if len(labels) != 1:
            raise ValueError("A detection recording crosses labels")
        probability = np.mean(window_probabilities[mask], axis=0)
        output_y.append(0 if str(labels[0]) == background_label else 1)
        output_scores.append(1.0 - float(probability[background[0]]))
        output_recordings.append(str(recording_id))
    return (
        np.asarray(output_y, dtype=int),
        np.asarray(output_scores, dtype=float),
        np.asarray(output_recordings, dtype=object),
    )


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    positive = np.asarray(scores)[np.asarray(y_true) == 1]
    negative = np.asarray(scores)[np.asarray(y_true) == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    comparison = positive[:, None] - negative[None, :]
    return float((np.sum(comparison > 0) + 0.5 * np.sum(comparison == 0)) / comparison.size)


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-np.asarray(scores), kind="stable")
    y = np.asarray(y_true, dtype=int)[order]
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float(np.sum(precision * y) / positives)


def source_threshold(y_source: np.ndarray, scores: np.ndarray, far: float = 0.05) -> float:
    negative = np.asarray(scores)[np.asarray(y_source) == 0]
    if not len(negative):
        raise ValueError("Source threshold requires source background examples")
    return float(np.quantile(negative, 1.0 - far, method="higher"))


def source_threshold_by_domain(
    y_source: np.ndarray,
    scores: np.ndarray,
    source_domains: np.ndarray,
    far: float = 0.05,
) -> dict[str, object]:
    """Maximum source-domain background quantile, fitted without target data."""
    y_source = np.asarray(y_source, dtype=int)
    scores = np.asarray(scores, dtype=float)
    source_domains = np.asarray(source_domains, dtype=object)
    if not (len(y_source) == len(scores) == len(source_domains)):
        raise ValueError("Detection threshold inputs must align")
    per_domain = {}
    for domain in sorted(set(source_domains.tolist()), key=str):
        mask = source_domains == domain
        per_domain[str(domain)] = source_threshold(y_source[mask], scores[mask], far=far)
    return {"threshold": max(per_domain.values()), "per_source_domain": per_domain, "far": far}


def detection_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, object]:
    y = np.asarray(y_true, dtype=int)
    predicted = (np.asarray(scores) >= threshold).astype(int)
    tn = int(np.sum((y == 0) & (predicted == 0)))
    fp = int(np.sum((y == 0) & (predicted == 1)))
    fn = int(np.sum((y == 1) & (predicted == 0)))
    tp = int(np.sum((y == 1) & (predicted == 1)))
    return {
        "auroc": binary_auc(y, scores),
        "auprc": average_precision(y, scores),
        "far": fp / max(fp + tn, 1),
        "miss_rate": fn / max(fn + tp, 1),
        "threshold": threshold,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "per_class_recall": {
            "background": tn / max(tn + fp, 1),
            "uav": tp / max(tp + fn, 1),
        },
    }
