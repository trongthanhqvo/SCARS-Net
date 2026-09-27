from __future__ import annotations

import numpy as np

from .resize import resize_2d
from .stft_anchor import frames


EPS = 1.0e-8


def discover_cyclic_bins(
    source_windows: np.ndarray,
    count: int,
    minimum_separation_bins: int = 2,
    effective_frame: int | None = None,
    recording_ids: np.ndarray | None = None,
) -> np.ndarray:
    source_windows = np.asarray(source_windows)
    if source_windows.ndim != 2 or len(source_windows) == 0:
        raise ValueError("Cyclic-frequency discovery requires a non-empty equal-length source batch")
    if recording_ids is None:
        recording_ids = np.asarray([f"window-{index}" for index in range(len(source_windows))])
    recording_ids = np.asarray(recording_ids)
    if len(recording_ids) != len(source_windows):
        raise ValueError("recording_ids must align with source_windows")
    spectra = []
    for x in source_windows:
        envelope = np.abs(x) - float(np.mean(np.abs(x)))
        spectra.append(np.abs(np.fft.rfft(envelope)) ** 2)
    spectra_array = np.asarray(spectra)
    recording_envelopes = np.stack(
        [
            np.median(spectra_array[recording_ids == recording_id], axis=0)
            for recording_id in sorted(set(recording_ids.tolist()), key=str)
        ]
    )
    score = np.median(recording_envelopes, axis=0)
    score[:2] = 0.0
    selected: list[int] = []
    selected_effective_shifts: set[int] = set()
    # Primary key: descending source score. Frozen tie rule: smallest
    # non-negative cyclic bin. np.lexsort makes that rule explicit.
    indices = np.arange(len(score))
    local_maximum = np.zeros_like(score, dtype=bool)
    if len(score) >= 3:
        local_maximum[1:-1] = (score[1:-1] >= score[:-2]) & (score[1:-1] >= score[2:])
    candidate_indices = indices[local_maximum]
    for index in candidate_indices[np.lexsort((candidate_indices, -score[candidate_indices]))]:
        index = int(index)
        alpha = index / source_windows.shape[1]
        effective_shift = None
        if effective_frame is not None:
            effective_shift = max(
                1,
                min(
                    effective_frame - 1,
                    int(np.floor(float(alpha) * effective_frame + 0.5)),
                ),
            )
            if effective_shift in selected_effective_shifts:
                continue
        if all(abs(index - old) >= minimum_separation_bins for old in selected):
            selected.append(index)
            if effective_shift is not None:
                selected_effective_shifts.add(effective_shift)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(
            "Insufficient source cyclic-frequency candidates with unique effective SCF shifts; "
            "increase frame_samples or reduce cyclic_count"
        )
    # Store alpha in cycles/sample, not discovery-window indices. Transforming
    # a different frame length then uses round(alpha * frame), avoiding the
    # previous modulo-frame frequency inconsistency.
    return np.asarray(sorted(selected), dtype=np.float64) / source_windows.shape[1]


def permute_frozen_bins(bins: np.ndarray, seed: int) -> np.ndarray:
    bins = np.asarray(bins, dtype=np.float64)
    if bins.size < 2:
        raise ValueError("Permutation control needs at least two bins")
    order = np.random.default_rng(seed).permutation(bins.size)
    if np.array_equal(order, np.arange(bins.size)):
        order = np.roll(order, 1)
    permuted = bins[order]
    if np.array_equal(permuted, bins):
        raise AssertionError("Cyclic permutation control did not change index order")
    return permuted


def native_spectral_correlation_from_frames(
    chunks: np.ndarray,
    cyclic_bins: np.ndarray,
    denominator_floor: float = EPS,
) -> np.ndarray:
    """Native-grid estimator before FFT shift/resizing of the map axes."""
    chunks = np.asarray(chunks)
    if chunks.ndim != 2 or chunks.shape[0] == 0:
        raise ValueError("SCF estimator requires a non-empty frame matrix")
    frame = chunks.shape[1]
    spectrum = np.fft.fft(chunks * np.hanning(frame), axis=1)
    rows = []
    for cyclic_bin in cyclic_bins:
        shift = max(1, min(frame - 1, int(np.floor(float(cyclic_bin) * frame + 0.5))))
        plus = np.roll(spectrum, shift // 2, axis=1)
        minus = np.roll(spectrum, -(shift - shift // 2), axis=1)
        cross = np.mean(plus * np.conj(minus), axis=0)
        p_plus = np.mean(np.abs(plus) ** 2, axis=0)
        p_minus = np.mean(np.abs(minus) ** 2, axis=0)
        rows.append(np.abs(cross) / np.sqrt(p_plus * p_minus + denominator_floor))
    return np.stack(rows)


def spectral_correlation_map(
    x: np.ndarray,
    cyclic_bins: np.ndarray,
    output_bins: int,
    frame: int = 128,
    hop: int = 32,
    denominator_floor: float = EPS,
) -> np.ndarray:
    native = native_spectral_correlation_from_frames(
        frames(x, frame, hop), cyclic_bins, denominator_floor
    )
    return resize_2d(np.fft.fftshift(native, axes=1), output_bins, output_bins)
