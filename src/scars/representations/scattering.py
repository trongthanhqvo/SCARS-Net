from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .resize import resize_2d, resize_2d_operator_norm


EPS = 1.0e-8


@dataclass(frozen=True)
class ScatteringMetadata:
    backend: str
    exact_registered_discrete_graph: bool
    continuous_wst_equivalence_claim: bool
    paths: tuple[str, ...]
    path_l1_products: tuple[float, ...]
    pooling_operator_norm: float
    resize_operator_norm: float
    log1p_lipschitz_bound: float
    boundary: str


def _filter(n: int, center: float, bandwidth: float) -> tuple[np.ndarray, float]:
    """Periodic, analytic, zero-mean Morlet wavelet in the DFT domain.

    The zero-frequency Gaussian correction makes the discrete wavelet exactly
    zero mean on the registered FFT grid.  Negative DFT frequencies are then
    removed, yielding the analytic filters used by the SCARS WST contract.
    """
    frequency = np.fft.fftfreq(n)
    bandwidth = max(float(bandwidth), EPS)
    carrier = np.exp(-0.5 * ((frequency - center) / bandwidth) ** 2)
    dc = np.exp(-0.5 * (frequency / bandwidth) ** 2)
    correction = np.exp(-0.5 * (center / bandwidth) ** 2)
    spectrum = carrier - correction * dc
    spectrum[frequency <= 0] = 0.0
    spectrum[0] = 0.0
    norm = float(np.sum(np.abs(np.fft.ifft(spectrum))))
    applied = spectrum / max(norm, EPS)
    applied_l1_norm = float(np.sum(np.abs(np.fft.ifft(applied))))
    return applied, applied_l1_norm


def scattering_path_count(j: int, q: int) -> tuple[int, int]:
    """Return the registered order-1/order-2 path counts for cost audits."""
    first = max(int(j) * int(q), 1)
    second = sum(
        max(int(j) - (index // max(int(q), 1)) - 1, 0)
        for index in range(first)
    )
    return first, second


def scattering_map(x: np.ndarray, j: int, q: int, output_bins: int) -> tuple[np.ndarray, ScatteringMetadata]:
    """Registered order-1/2 periodic analytic-Morlet scattering transform."""
    n = x.size
    q = max(int(q), 1)
    j = max(int(j), 1)
    centers = 0.45 * 2.0 ** (-np.arange(j * q, dtype=np.float64) / q)
    second_centers = 0.45 * 2.0 ** (-np.arange(j, dtype=np.float64))
    x_fft = np.fft.fft(x)
    first: list[np.ndarray] = []
    paths: list[np.ndarray] = []
    path_names: list[str] = []
    path_products: list[float] = []
    first_norms: list[float] = []
    for index, center in enumerate(centers):
        wavelet, norm = _filter(n, float(center), max(float(center) / q, 1.0 / n))
        value = np.abs(np.fft.ifft(x_fft * wavelet))
        first.append(value)
        first_norms.append(norm)
        paths.append(value)
        path_names.append(f"order1:{index}")
        path_products.append(norm)
    for first_index in range(len(first)):
        first_fft = np.fft.fft(first[first_index])
        # Standard scattering scale ordering: the second wavelet must be
        # lower-frequency than the first wavelet's carrier.
        second_start = first_index // q + 1
        for second_index in range(second_start, j):
            center = second_centers[second_index]
            wavelet, norm = _filter(n, float(center), max(float(center), 1.0 / n))
            paths.append(np.abs(np.fft.ifft(first_fft * wavelet)))
            path_names.append(f"order2:{first_index}>{second_index}")
            path_products.append(first_norms[first_index] * norm)
    path_map = np.stack(paths)
    cut = (n // output_bins) * output_bins
    pooled = (
        path_map.mean(axis=1, keepdims=True)
        if cut == 0
        else path_map[:, :cut].reshape(path_map.shape[0], output_bins, -1).mean(axis=2)
    )
    pooling_operator_norm = (
        1.0 / np.sqrt(n) if cut == 0 else 1.0 / np.sqrt(cut // output_bins)
    )
    resize_operator_norm = resize_2d_operator_norm(
        pooled.shape[0], pooled.shape[1], output_bins, output_bins
    )
    return (
        resize_2d(np.log1p(pooled), output_bins, output_bins),
        ScatteringMetadata(
            backend="registered_periodic_analytic_morlet_order12",
            exact_registered_discrete_graph=True,
            continuous_wst_equivalence_claim=False,
            paths=tuple(path_names),
            path_l1_products=tuple(path_products),
            pooling_operator_norm=float(pooling_operator_norm),
            resize_operator_norm=float(resize_operator_norm),
            log1p_lipschitz_bound=1.0,
            boundary="periodic_fft_then_deterministic_resize",
        ),
    )
