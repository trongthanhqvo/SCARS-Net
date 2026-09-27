from __future__ import annotations

import numpy as np

from .resize import resize_2d


def frames(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if x.size < frame:
        x = np.pad(x, (0, frame - x.size))
    starts = np.arange(0, x.size - frame + 1, hop)
    return np.stack([x[start : start + frame] for start in starts])


def stft_power(x: np.ndarray, frame: int = 128, hop: int = 32) -> np.ndarray:
    chunks = frames(x, frame, hop)
    spectrum = np.fft.fftshift(
        np.fft.fft(chunks * np.hanning(frame), n=frame, axis=1), axes=1
    )
    return np.abs(spectrum).T**2


def stft_anchor(x: np.ndarray, output_bins: int, frame: int = 128, hop: int = 32) -> np.ndarray:
    return resize_2d(np.log1p(stft_power(x, frame, hop)), output_bins, output_bins)
