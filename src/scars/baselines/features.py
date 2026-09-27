from __future__ import annotations

import numpy as np

from scars.representations.resize import resize_2d
from scars.representations.stft_anchor import frames


class SourceGlobalStandardizer:
    def __init__(self, epsilon: float = 1.0e-6):
        self.epsilon = epsilon
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, values: np.ndarray, *, split_role: str) -> "SourceGlobalStandardizer":
        if split_role != "source_fit":
            raise PermissionError("Baseline standardization may fit only on source_fit")
        values = np.asarray(values, dtype=np.float64)
        axes = tuple(index for index in range(values.ndim) if index != 1)
        self.mean = values.mean(axis=axes, keepdims=True)
        self.scale = np.maximum(values.std(axis=axes, keepdims=True), self.epsilon)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("SourceGlobalStandardizer is not fitted")
        return ((np.asarray(values) - self.mean) / self.scale).astype(np.float32)


def real_imag(iq: np.ndarray) -> np.ndarray:
    iq = np.asarray(iq)
    return np.stack([np.real(iq), np.imag(iq)], axis=1).astype(np.float32)


def magnitude_phase(iq: np.ndarray) -> np.ndarray:
    iq = np.asarray(iq)
    magnitude = np.log1p(np.abs(iq))
    phase = np.angle(iq) / np.pi
    return np.stack([magnitude, phase], axis=1).astype(np.float32)


def log_stft_images(
    iq: np.ndarray, output_bins: int, frame: int = 128, hop: int = 32
) -> np.ndarray:
    images = []
    for window in np.asarray(iq):
        chunks = frames(window, frame, hop) * np.hanning(frame)
        power = np.log1p(np.abs(np.fft.fftshift(np.fft.fft(chunks), axes=1)) ** 2)
        images.append(resize_2d(power.T, output_bins, output_bins))
    return np.asarray(images, dtype=np.float32)[:, None]


def cwt_images(iq: np.ndarray, output_bins: int, scales: int = 16) -> np.ndarray:
    images = []
    for window in np.asarray(iq):
        rows = []
        maximum_scale = min(256.0, max(2.0, len(window) / 2.0))
        for scale in np.geomspace(2.0, maximum_scale, scales):
            radius = max(4, int(np.ceil(4.5 * 1.5 * scale)))
            time = np.arange(-radius, radius + 1, dtype=np.float64)
            wavelet = np.exp(2j * np.pi * time / scale) * np.exp(
                -0.5 * (time / (1.5 * scale)) ** 2
            )
            wavelet /= max(float(np.linalg.norm(wavelet)), 1.0e-12)
            convolution = np.convolve(window, np.conj(wavelet[::-1]), mode="full")
            start = (len(convolution) - len(window)) // 2
            rows.append(np.log1p(np.abs(convolution[start : start + len(window)])))
        images.append(resize_2d(np.asarray(rows), output_bins, output_bins))
    return np.asarray(images, dtype=np.float32)[:, None]


def _zigzag_indices(rows: int, columns: int) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    for diagonal in range(rows + columns - 1):
        cells = [
            (row, diagonal - row)
            for row in range(rows)
            if 0 <= diagonal - row < columns
        ]
        output.extend(cells if diagonal % 2 else cells[::-1])
    return output


def stft_dct_features(
    iq: np.ndarray, output_bins: int = 16, retained_coefficients: int = 64
) -> np.ndarray:
    from scipy.fft import dctn

    images = log_stft_images(iq, output_bins)[:, 0]
    indices = _zigzag_indices(output_bins, output_bins)[:retained_coefficients]
    return np.asarray(
        [[dctn(image, type=2, norm="ortho")[row, column] for row, column in indices] for image in images],
        dtype=np.float32,
    )


def _db4_packet_level3(signal: np.ndarray) -> list[np.ndarray]:
    # PyWavelets' ``db4`` analysis coefficients, embedded to keep the baseline portable.
    low = np.asarray(
        [
            -0.010597401785069032,
            0.0328830116668852,
            0.030841381835560764,
            -0.18703481171888114,
            -0.027983769416859854,
            0.6308807679298587,
            0.7148465705529157,
            0.2303778133088965,
        ]
    )
    high = ((-1.0) ** np.arange(len(low))) * low[::-1]
    nodes = [np.asarray(signal)]
    for _ in range(3):
        children = []
        for node in nodes:
            children.extend(
                [np.convolve(node, low, mode="same")[::2], np.convolve(node, high, mode="same")[::2]]
            )
        nodes = children
    return nodes


def fixed_wavelet_subband_features(iq: np.ndarray, bands: int = 8) -> np.ndarray:
    if bands != 8:
        raise ValueError("The registered level-3 db4 packet has exactly eight nodes")
    output = []
    for window in np.asarray(iq):
        nodes = _db4_packet_level3(window)
        output.append([np.log1p(np.mean(np.abs(node) ** 2)) for node in nodes])
    return np.asarray(output, dtype=np.float32)


def log_psd_sobel_images(iq: np.ndarray, output_bins: int = 16) -> np.ndarray:
    from scipy.ndimage import sobel

    output = []
    for window in np.asarray(iq):
        psd = np.log1p(np.abs(np.fft.fftshift(np.fft.fft(window))) ** 2)
        base = resize_2d(np.repeat(psd[None, :], output_bins, axis=0), output_bins, output_bins)
        horizontal = sobel(base, axis=1, mode="reflect")
        vertical = sobel(base, axis=0, mode="reflect")
        output.append(np.stack([base, horizontal, vertical]))
    return np.asarray(output, dtype=np.float32)
