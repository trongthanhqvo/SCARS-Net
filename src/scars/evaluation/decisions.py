from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .classification import macro_f1
from .statistics import holm_correction, midranks


DECISION_ENGINE_VERSION = "scars-confirmatory-1.1.0"


def paired_bootstrap_p(
    paired_difference: np.ndarray,
    *,
    alternative: str,
    margin: float = 0.0,
    seed: int,
    resamples: int = 10_000,
) -> dict[str, float]:
    """Recording-block one-sided paired bootstrap with plus-one tail count."""
    values = np.asarray(paired_difference, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("paired_difference needs at least two finite recording effects")
    observed = float(np.mean(values))
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [np.mean(values[rng.integers(0, len(values), len(values))]) for _ in range(resamples)]
    )
    centered = draws - observed
    if alternative == "superiority":
        # H0: effect <= margin; large positive values support the alternative.
        p_value = (1 + np.sum(centered >= observed - margin)) / (resamples + 1)
    elif alternative == "noninferiority":
        # H0: effect <= -margin.
        p_value = (1 + np.sum(centered >= observed + margin)) / (resamples + 1)
    else:
        raise ValueError("alternative must be superiority or noninferiority")
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": observed,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "raw_p": float(p_value),
        "resamples": int(resamples),
        "seed": int(seed),
        "margin": float(margin),
    }


def h1_raw_p(candidate_records: Sequence[dict[str, object]]) -> dict[str, object]:
    """Registered candidate-level IUT followed by candidate Bonferroni."""
    if not candidate_records:
        return {"status": "untestable", "reason": "no_pre_specified_wc_candidates"}
    if any(not bool(candidate.get("measurement_valid", False)) for candidate in candidate_records):
        return {
            "status": "untestable",
            "reason": "invalid_registered_representation_latency_measurement",
        }
    rows = []
    for candidate in candidate_records:
        p_instability = max(
            float(candidate["p_instability_vs_w"]),
            float(candidate["p_instability_vs_stft"]),
        )
        p_f1 = max(float(candidate["p_f1_vs_w"]), float(candidate["p_f1_vs_stft"]))
        feasible = bool(candidate["feasible"])
        p_iut = max(p_instability, p_f1) if feasible else 1.0
        rows.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "p_instability": p_instability,
                "p_f1": p_f1,
                "feasible": feasible,
                "measurement_valid": True,
                "p_iut": p_iut,
            }
        )
    raw = min(1.0, len(rows) * min(float(row["p_iut"]) for row in rows))
    return {"status": "ok", "raw_p": raw, "candidate_count": len(rows), "candidates": rows}


def _rank_residual(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    ranked = midranks(values)
    design = np.column_stack([np.ones(len(ranked)), controls])
    return ranked - design @ (np.linalg.pinv(design) @ ranked)


def h2_statistic(
    instability: np.ndarray,
    source_f1: np.ndarray,
    held_f1: np.ndarray,
    log_macs: np.ndarray,
    resolution: np.ndarray,
    *,
    eta: float = 1.0e-10,
) -> tuple[float, list[dict[str, object]]]:
    """Equal-domain normalized residual cross-product for the frozen bank."""
    arrays = [np.asarray(value, dtype=float) for value in (instability, source_f1, held_f1)]
    if any(value.ndim != 2 for value in arrays) or not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
        raise ValueError("H2 domain arrays must share shape [domains, configurations]")
    domains, configurations = arrays[0].shape
    if domains < 3 or configurations < 8:
        raise ValueError("H2 requires at least three domains and eight configurations")
    log_macs = np.asarray(log_macs, dtype=float)
    resolution = np.asarray(resolution, dtype=float)
    if log_macs.shape != (configurations,) or resolution.shape != (configurations,):
        raise ValueError("H2 cost/resolution covariates must align with configurations")
    per_domain = []
    statistics = []
    for domain in range(domains):
        degradation = arrays[1][domain] - arrays[2][domain]
        controls = np.column_stack(
            [midranks(arrays[1][domain]), midranks(log_macs), midranks(resolution)]
        )
        residual_l = _rank_residual(arrays[0][domain], controls)
        residual_d = _rank_residual(degradation, controls)
        norm_l = float(np.linalg.norm(residual_l))
        norm_d = float(np.linalg.norm(residual_d))
        joint = int(np.sum((np.abs(residual_l) > eta) & (np.abs(residual_d) > eta)))
        if norm_l <= eta or norm_d <= eta or joint < 8:
            raise ValueError(f"degenerate_h2_domain_{domain}")
        statistic = float(np.dot(residual_l, residual_d) / (norm_l * norm_d))
        statistics.append(statistic)
        per_domain.append(
            {
                "domain_index": domain,
                "statistic": statistic,
                "residual_instability": residual_l.tolist(),
                "residual_degradation": residual_d.tolist(),
                "joint_nonzero": joint,
            }
        )
    return float(np.mean(statistics)), per_domain


@dataclass(frozen=True)
class H2DomainRecords:
    source_truth: np.ndarray
    source_prediction: np.ndarray
    source_instability: np.ndarray
    source_domain: np.ndarray
    held_truth: np.ndarray
    held_prediction: np.ndarray


def _stratified_indices(labels: np.ndarray, domains: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = []
    strata = sorted(set(zip(map(str, domains.tolist()), map(str, labels.tolist()))))
    for domain, label in strata:
        eligible = np.flatnonzero((domains.astype(str) == domain) & (labels.astype(str) == label))
        output.extend(eligible[rng.integers(0, len(eligible), len(eligible))].tolist())
    return np.asarray(output, dtype=int)


def _metric_vector(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.asarray([macro_f1(truth, prediction[:, index]) for index in range(prediction.shape[1])])


def h2_hierarchical_bootstrap(
    domains: Sequence[H2DomainRecords],
    log_macs: np.ndarray,
    resolution: np.ndarray,
    *,
    seed: int = 24022,
    resamples: int = 10_000,
) -> dict[str, object]:
    if len(domains) < 3:
        return {"status": "untestable", "reason": "fewer_than_three_domains"}
    source_f1 = []
    held_f1 = []
    instability = []
    for domain in domains:
        source_f1.append(_metric_vector(domain.source_truth, domain.source_prediction))
        held_f1.append(_metric_vector(domain.held_truth, domain.held_prediction))
        instability.append(np.mean(domain.source_instability, axis=0))
    try:
        observed, detail = h2_statistic(
            np.asarray(instability), np.asarray(source_f1), np.asarray(held_f1), log_macs, resolution
        )
    except ValueError as error:
        return {"status": "untestable", "reason": str(error)}
    rng = np.random.default_rng(seed)
    bootstrap = []
    invalid = 0
    for _ in range(resamples):
        draw_source_f1 = []
        draw_held_f1 = []
        draw_instability = []
        for domain in domains:
            source_index = _stratified_indices(
                domain.source_truth, domain.source_domain, rng
            )
            held_index = _stratified_indices(
                domain.held_truth,
                np.repeat("held", len(domain.held_truth)).astype(object),
                rng,
            )
            draw_source_f1.append(
                _metric_vector(domain.source_truth[source_index], domain.source_prediction[source_index])
            )
            draw_held_f1.append(
                _metric_vector(domain.held_truth[held_index], domain.held_prediction[held_index])
            )
            draw_instability.append(np.mean(domain.source_instability[source_index], axis=0))
        try:
            value, _ = h2_statistic(
                np.asarray(draw_instability),
                np.asarray(draw_source_f1),
                np.asarray(draw_held_f1),
                log_macs,
                resolution,
            )
            bootstrap.append(value)
        except ValueError:
            invalid += 1
    valid_fraction = len(bootstrap) / resamples
    if valid_fraction < 0.95:
        return {
            "status": "untestable",
            "reason": "fewer_than_95_percent_valid_bootstrap_draws",
            "valid_fraction": valid_fraction,
        }
    values = np.asarray(bootstrap)
    raw_p = float((1 + np.sum(values - observed >= observed)) / (len(values) + 1))
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "status": "ok",
        "procedure": "registered_one_sided_centered_bootstrap",
        "null_boundary": 0.0,
        "dependence_preservation": "each_resampled_recording_carries_the_complete_configuration_vector",
        "domain_aggregation": "fixed_domains_equal_weighted",
        "estimate": observed,
        "raw_p": raw_p,
        "bootstrap_ci95": [float(low), float(high)],
        "valid_fraction": valid_fraction,
        "invalid_draws": invalid,
        "resamples": resamples,
        "seed": seed,
        "per_domain": detail,
    }


def h3_raw_p(components: Sequence[dict[str, object]]) -> dict[str, object]:
    if not components:
        return {"status": "untestable", "reason": "no_active_family_deletions"}
    failed_degeneracy = []
    p_values = []
    for component in components:
        degenerate = float(component["cosine"]) >= 0.995 and float(component["nmae"]) <= 0.010
        if degenerate:
            failed_degeneracy.append(str(component["family"]))
        p_values.append(float(component["raw_p"]))
    raw = 1.0 if failed_degeneracy else max(p_values)
    return {
        "status": "ok",
        "raw_p": raw,
        "degenerate_families": failed_degeneracy,
        "components": list(components),
    }


def h4_raw_p(
    *,
    p_performance_noninferiority: float,
    p_latency_superiority: float | None,
    bytes_strictly_reduced: bool,
    macs_strictly_reduced: bool,
    measurement_valid: bool,
) -> dict[str, object]:
    if not measurement_valid or p_latency_superiority is None:
        return {"status": "untestable", "reason": "invalid_latency_measurement"}
    if not bytes_strictly_reduced or not macs_strictly_reduced:
        return {"status": "ok", "raw_p": 1.0, "reason": "deterministic_cost_gate_failed"}
    return {
        "status": "ok",
        "raw_p": max(float(p_performance_noninferiority), float(p_latency_superiority)),
    }


def finalize_hypotheses(
    raw_records: dict[str, dict[str, object]], alpha: float = 0.05
) -> dict[str, object]:
    required = ("H1", "H2", "H3", "H4")
    if any(raw_records.get(key, {}).get("status") != "ok" for key in required):
        return {
            "status": "untestable",
            "holm": None,
            "hypotheses": {
                key: {"status": "untestable", "reason": raw_records.get(key, {}).get("reason")}
                for key in required
            },
        }
    raw = {key: float(raw_records[key]["raw_p"]) for key in required}
    adjusted = holm_correction(raw)
    return {
        "status": "decided",
        "holm": adjusted,
        "hypotheses": {
            key: {
                "status": "supported" if adjusted[key] <= alpha else "negative",
                "raw_p": raw[key],
                "holm_p": adjusted[key],
            }
            for key in required
        },
    }


def m1_state_machine(
    *,
    eligible: bool,
    integrity_pass: bool,
    calibration_pass: bool,
    relation_coverage_by_nuisance: dict[str, float],
    relation_recording_count: int,
    delta_gate: float,
    delta_gate_ci_low: float,
    delta_shuffle: float,
    delta_shuffle_ci_low: float,
    worst_domain_ci_low: float,
    compute_measurement_valid: bool,
    parameter_ratio: float,
    mac_ratio: float,
    latency_ratio: float,
) -> dict[str, object]:
    coverage_pass = (
        relation_recording_count >= 20
        and bool(relation_coverage_by_nuisance)
        and min(relation_coverage_by_nuisance.values()) >= 0.50
    )
    if not (eligible and integrity_pass and calibration_pass and coverage_pass and compute_measurement_valid):
        return {"status": "untestable", "reason": "eligibility_integrity_calibration_coverage_or_compute_invalid"}
    if delta_gate <= 0 or delta_shuffle <= 0:
        return {"status": "negative", "reason": "nonpositive_direct_contrast"}
    compute_pass = parameter_ratio <= 1.10 and mac_ratio <= 1.10 and latency_ratio <= 1.25
    supported = (
        delta_gate_ci_low > 0
        and delta_shuffle_ci_low > 0
        and worst_domain_ci_low > -0.01
        and compute_pass
    )
    if supported:
        return {"status": "supported", "reason": "all_registered_components_pass"}
    return {
        "status": "partial",
        "reason": "positive_estimates_but_interval_worst_domain_or_compute_gate_incomplete",
        "compute_pass": compute_pass,
    }
