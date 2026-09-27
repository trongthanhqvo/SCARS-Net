from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    lookup = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for truth, predicted in zip(y_true, y_pred):
        matrix[lookup[truth], lookup[predicted]] += 1
    return labels, matrix


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    scores = []
    for label in labels:
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(2.0 * precision * recall / max(precision + recall, 1.0e-12))
    return float(np.mean(scores))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    labels, matrix = confusion_matrix(y_true, y_pred)
    per_class = {}
    for index, label in enumerate(labels):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index].sum() - tp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        per_class[str(label)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(2.0 * precision * recall / max(precision + recall, 1.0e-12)),
            "support": int(matrix[index].sum()),
        }
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "balanced_accuracy": float(np.mean([row["recall"] for row in per_class.values()])),
        "confusion_matrix": matrix.tolist(),
        "normalized_confusion_matrix": normalized.tolist(),
        "label_order": [str(label) for label in labels],
        "per_class": per_class,
        "per_class_recall": {key: value["recall"] for key, value in per_class.items()},
    }


def recording_level_metrics(
    window_labels: np.ndarray,
    recording_ids: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, object]:
    """Aggregate predictions to one replicate per recording before scoring."""
    true_labels = []
    predicted_labels = []
    recording_probabilities = []
    ordered_recordings = []
    for recording_id in sorted(np.unique(recording_ids), key=str):
        mask = recording_ids == recording_id
        labels = np.unique(window_labels[mask])
        if len(labels) != 1:
            raise ValueError(f"Recording {recording_id} has inconsistent window labels")
        true_labels.append(labels[0])
        probability = np.mean(probabilities[mask], axis=0)
        recording_probabilities.append(probability)
        predicted_labels.append(classes[int(np.argmax(probability))])
        ordered_recordings.append(str(recording_id))
    metrics = classification_metrics(np.asarray(true_labels), np.asarray(predicted_labels))
    metrics.update(
        {
            "replication_unit": "recording",
            "recording_count": len(ordered_recordings),
            "recording_order": ordered_recordings,
            "recording_truth": [str(label) for label in true_labels],
            "recording_prediction": [str(label) for label in predicted_labels],
            "recording_probabilities": np.asarray(recording_probabilities).tolist(),
            "recording_confidence": np.max(recording_probabilities, axis=1).tolist(),
        }
    )
    return metrics


def calibration_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    bins: int = 10,
) -> dict[str, object]:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    classes = np.asarray(classes)
    if probabilities.ndim != 2 or len(probabilities) != len(y_true):
        raise ValueError("Calibration requires aligned [N,K] probabilities and labels")
    prediction_index = np.argmax(probabilities, axis=1)
    confidence = probabilities[np.arange(len(probabilities)), prediction_index]
    correct = classes[prediction_index] == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    for index in range(bins):
        mask = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
        )
        if not np.any(mask):
            continue
        mean_confidence = float(np.mean(confidence[mask]))
        empirical_accuracy = float(np.mean(correct[mask]))
        ece += float(np.mean(mask)) * abs(mean_confidence - empirical_accuracy)
        rows.append(
            {
                "bin_low": float(edges[index]),
                "bin_high": float(edges[index + 1]),
                "confidence": mean_confidence,
                "empirical_accuracy": empirical_accuracy,
                "support": int(np.sum(mask)),
            }
        )
    class_index = {label: index for index, label in enumerate(classes)}
    one_hot = np.zeros_like(probabilities)
    for row, label in enumerate(y_true):
        one_hot[row, class_index[label]] = 1.0
    true_probability = probabilities[
        np.arange(len(probabilities)),
        np.asarray([class_index[label] for label in y_true], dtype=int),
    ]
    return {
        "ece": float(ece),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "nll": float(-np.mean(np.log(np.clip(true_probability, 1.0e-12, 1.0)))),
        "bins": rows,
    }
