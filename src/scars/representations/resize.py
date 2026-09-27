from __future__ import annotations

import numpy as np


def resize_axis_matrix(old_size: int, new_size: int) -> np.ndarray:
    """Exact linear-interpolation matrix used by :func:`_resize_axis`."""
    if old_size <= 0 or new_size <= 0:
        raise ValueError("Resize dimensions must be positive")
    if old_size == new_size:
        return np.eye(old_size, dtype=float)
    old = np.linspace(0.0, 1.0, old_size)
    new = np.linspace(0.0, 1.0, new_size)
    return np.stack([np.interp(new, old, np.eye(old_size)[index]) for index in range(old_size)], axis=1)


def resize_2d_operator_norm(
    old_height: int, old_width: int, new_height: int, new_width: int
) -> float:
    """Spectral norm of the separable 2-D resize (Kronecker product)."""
    height_norm = float(np.linalg.norm(resize_axis_matrix(old_height, new_height), ord=2))
    width_norm = float(np.linalg.norm(resize_axis_matrix(old_width, new_width), ord=2))
    return height_norm * width_norm


def _resize_axis(values: np.ndarray, size: int, axis: int) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[axis] == size:
        return values.copy()
    old = np.linspace(0.0, 1.0, values.shape[axis])
    new = np.linspace(0.0, 1.0, size)
    moved = np.moveaxis(values, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    resized = np.stack([np.interp(new, old, row) for row in flat])
    return np.moveaxis(resized.reshape(moved.shape[:-1] + (size,)), -1, axis)


def resize_2d(values: np.ndarray, height: int, width: int) -> np.ndarray:
    values = np.atleast_2d(values)
    return _resize_axis(_resize_axis(values, height, 0), width, 1)
