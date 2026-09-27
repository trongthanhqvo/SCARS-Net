from __future__ import annotations

import numpy as np


COSINE_THRESHOLD = 0.995
NMAE_THRESHOLD = 0.010


def representation_similarity(a: np.ndarray, b: np.ndarray) -> dict[str, object]:
    x = np.asarray(a, dtype=float).reshape(len(a), -1)
    y = np.asarray(b, dtype=float).reshape(len(b), -1)
    if x.shape[1] != y.shape[1]:
        query = np.linspace(0.0, 1.0, max(x.shape[1], y.shape[1]))
        x = np.stack([np.interp(query, np.linspace(0.0, 1.0, x.shape[1]), row) for row in x])
        y = np.stack([np.interp(query, np.linspace(0.0, 1.0, y.shape[1]), row) for row in y])
    cosine = np.sum(x * y, axis=1) / np.maximum(
        np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1.0e-12
    )
    dynamic = np.maximum(np.ptp(np.concatenate([x, y], axis=1), axis=1), 1.0e-12)
    nmae = np.mean(np.abs(x - y), axis=1) / dynamic
    mean_cosine, mean_nmae = float(np.mean(cosine)), float(np.mean(nmae))
    return {
        "cosine": mean_cosine,
        "normalized_mae": mean_nmae,
        "degenerate": mean_cosine >= COSINE_THRESHOLD and mean_nmae <= NMAE_THRESHOLD,
        "rule": {"cosine_min": COSINE_THRESHOLD, "nmae_max": NMAE_THRESHOLD, "join": "and"},
    }
