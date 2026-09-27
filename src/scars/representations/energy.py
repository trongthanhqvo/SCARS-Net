from __future__ import annotations

import numpy as np

from .resize import resize_2d
from .stft_anchor import frames


def raw_energy_map(
    x: np.ndarray,
    output_bins: int,
    frame: int = 128,
    hop: int = 32,
) -> np.ndarray:
    """Registered local mean-power trace, repeated then resized to the common grid."""
    local = np.mean(np.abs(frames(x, frame, hop)) ** 2, axis=1)
    native = np.repeat(local[:, None], 2, axis=1)
    return resize_2d(native, output_bins, output_bins)


def source_energy_reference(maps: np.ndarray, recording_ids: np.ndarray) -> float:
    """Median window energy within recording, then median across recordings."""
    maps = np.asarray(maps)
    recording_ids = np.asarray(recording_ids)
    if len(maps) != len(recording_ids) or len(maps) == 0:
        raise ValueError("Energy reference requires one recording ID per source window")
    window_energy = np.median(maps.reshape(len(maps), -1), axis=1)
    recording_medians = [
        np.median(window_energy[recording_ids == recording_id])
        for recording_id in sorted(set(recording_ids.tolist()))
    ]
    return max(float(np.median(recording_medians)), 1.0e-8)


def normalize_energy(values: np.ndarray, reference: float) -> np.ndarray:
    return np.log1p(np.maximum(values, 0.0) / reference)
