from __future__ import annotations

from typing import Callable

import numpy as np


def midranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def _residual_projector(control: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(len(control)), control])
    return np.eye(len(control)) - x @ np.linalg.pinv(x)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _checked_correlation(a: np.ndarray, b: np.ndarray, eta: float) -> float | None:
    """Return a finite correlation, or ``None`` for a degenerate draw.

    The check is deliberately per draw.  An observed nonzero residual does not
    imply that every Freedman--Lane permutation remains outside the nuisance
    design span.
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if not np.isfinite(norm_a) or not np.isfinite(norm_b) or norm_a <= eta or norm_b <= eta:
        return None
    value = float(np.dot(a, b) / (norm_a * norm_b))
    return value if np.isfinite(value) else None


def coherent_freedman_lane(
    instability: np.ndarray,
    degradation: np.ndarray,
    source_f1: np.ndarray,
    seed: int = 24021,
    permutations: int = 99_999,
    eta: float = 1.0e-10,
    minimum_joint_indices: int = 8,
) -> dict[str, object]:
    """One-sided domain-balanced partial-rank sensitivity diagnostic.

    Arrays have shape ``(domains, ordered_configurations)``. Midranks are
    computed once; a single configuration permutation is applied to every
    domain per draw, after which residualization and correlation are
    recomputed. With a nonconstant source-F1 nuisance design, arbitrary
    configuration permutations generally do not preserve the residual
    projector. The returned plus-one tail fraction is therefore a sensitivity
    diagnostic, not an exact conditional p-value or a valid Holm input.
    """
    l = np.asarray(instability, dtype=float)
    d = np.asarray(degradation, dtype=float)
    f = np.asarray(source_f1, dtype=float)
    if l.shape != d.shape or l.shape != f.shape or l.ndim != 2:
        raise ValueError("H2 arrays must share shape (domains, ordered configurations)")
    domains, configurations = l.shape
    if domains < 3:
        return {"status": "blocked", "reason": "fewer_than_three_domains"}
    ranked: list[dict[str, np.ndarray]] = []
    domain_statistics: list[float] = []
    for domain in range(domains):
        y = midranks(d[domain])
        z = midranks(l[domain])
        control = midranks(f[domain])
        m = _residual_projector(control)
        residual_y, residual_z = m @ y, m @ z
        joint = int(np.sum((np.abs(residual_y) > eta) & (np.abs(residual_z) > eta)))
        if np.linalg.norm(residual_y) <= eta or np.linalg.norm(residual_z) <= eta:
            return {"status": "blocked", "reason": f"zero_residual_norm_domain_{domain}"}
        if joint < minimum_joint_indices:
            return {
                "status": "blocked",
                "reason": f"insufficient_joint_residual_indices_domain_{domain}",
                "observed": joint,
            }
        observed_correlation = _checked_correlation(residual_y, residual_z, eta)
        if observed_correlation is None:
            return {"status": "blocked", "reason": f"nonfinite_observed_domain_{domain}"}
        domain_statistics.append(observed_correlation)
        ranked.append(
            {
                "y": y,
                "z": z,
                "control": control,
                "m": m,
                "residual_y": residual_y,
                "residual_z": residual_z,
            }
        )
    observed = float(np.mean(domain_statistics))
    rng = np.random.default_rng(seed)
    exceedances = 0
    for draw_index in range(permutations):
        permutation = rng.permutation(configurations)
        permuted_statistics = []
        for domain_index, item in enumerate(ranked):
            m = item["m"]
            y_star = (np.eye(configurations) - m) @ item["y"] + item["residual_y"][permutation]
            correlation = _checked_correlation(m @ y_star, item["residual_z"], eta)
            if correlation is None:
                return {
                    "status": "blocked",
                    "reason": "degenerate_permutation_residual",
                    "offending_draw": draw_index,
                    "offending_domain": domain_index,
                    "permutation": permutation.tolist(),
                    "seed": seed,
                    "eta": eta,
                    "configuration_count": configurations,
                }
            permuted_statistics.append(correlation)
        exceedances += float(np.mean(permuted_statistics)) >= observed - 1.0e-15
    tail_fraction = float((1 + exceedances) / (permutations + 1))
    return {
        "status": "ok",
        "alternative": "greater_diagnostic_tail",
        "estimate": observed,
        "per_domain": domain_statistics,
        "tail_fraction_diagnostic": tail_fraction,
        "inferential_use": "forbidden_diagnostic_only",
        "permutations": permutations,
        "seed": seed,
        "eta": eta,
        "minimum_joint_indices": minimum_joint_indices,
        "configuration_count": configurations,
        "ranks": [
            {
                "degradation": item["y"].tolist(),
                "instability": item["z"].tolist(),
                "source_f1": item["control"].tolist(),
                "degradation_residual": item["residual_y"].tolist(),
                "instability_residual": item["residual_z"].tolist(),
            }
            for item in ranked
        ],
    }


def block_bootstrap_ci(
    block_ids: np.ndarray,
    values: np.ndarray,
    seed: int,
    resamples: int = 10_000,
    confidence: float = 0.95,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> tuple[float, float]:
    blocks = np.unique(block_ids)
    if len(blocks) < 2:
        return float("nan"), float("nan")
    groups = [np.asarray(values)[block_ids == block] for block in blocks]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(resamples):
        draw = rng.integers(0, len(groups), size=len(groups))
        estimates.append(float(statistic(np.concatenate([groups[index] for index in draw]))))
    alpha = 1.0 - confidence
    low, high = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def holm_correction(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(
        ((key, float(value)) for key, value in p_values.items() if np.isfinite(value)),
        key=lambda item: item[1],
    )
    output = {key: float("nan") for key in p_values}
    running = 0.0
    for index, (key, value) in enumerate(ordered):
        running = max(running, (len(ordered) - index) * value)
        output[key] = min(1.0, running)
    return output


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    difference = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(difference) < 2:
        return float("nan")
    scale = float(np.std(difference, ddof=1))
    if scale <= 1.0e-12:
        return 0.0 if abs(float(np.mean(difference))) <= 1.0e-12 else float("inf")
    return float(np.mean(difference) / scale)
