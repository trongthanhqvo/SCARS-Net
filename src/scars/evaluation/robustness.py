from __future__ import annotations

import numpy as np


def normalized_curve_area(grid: list[float], values: list[float]) -> float:
    x = np.asarray(grid, dtype=float)
    y = np.asarray(values, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    span = float(x[-1] - x[0])
    integration = getattr(np, "trapezoid", np.trapz)
    return float(integration(y, x) / span) if span > 0 else float("nan")
