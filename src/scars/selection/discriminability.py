from __future__ import annotations

import numpy as np


def regularized_fisher_ratio(
    features: np.ndarray, labels: np.ndarray, ridge: float = 1.0e-3
) -> float:
    x = np.asarray(features, dtype=np.float64).reshape(len(features), -1)
    y = np.asarray(labels)
    grand = x.mean(axis=0)
    between = 0.0
    within = 0.0
    for label in np.unique(y):
        group = x[y == label]
        center = group.mean(axis=0)
        between += len(group) * float(np.sum((center - grand) ** 2))
        within += float(np.sum((group - center) ** 2))
    return float(between / (within + ridge * x.shape[1]))
