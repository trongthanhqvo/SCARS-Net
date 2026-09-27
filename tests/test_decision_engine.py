from __future__ import annotations

import numpy as np

from scars.evaluation.classification import calibration_metrics, classification_metrics
from scars.evaluation.decisions import (
    finalize_hypotheses,
    h1_raw_p,
    h2_statistic,
    h3_raw_p,
    h4_raw_p,
    m1_state_machine,
)
from scars.evaluation.detection import source_threshold_by_domain
from scars.evaluation.robustness import normalized_curve_area


def test_h1_uses_both_comparators_and_candidate_bonferroni():
    result = h1_raw_p(
        [
            {
                "candidate_id": "a",
                "p_instability_vs_w": 0.01,
                "p_instability_vs_stft": 0.03,
                "p_f1_vs_w": 0.02,
                "p_f1_vs_stft": 0.04,
                "feasible": True,
                "measurement_valid": True,
            },
            {
                "candidate_id": "b",
                "p_instability_vs_w": 0.2,
                "p_instability_vs_stft": 0.1,
                "p_f1_vs_w": 0.2,
                "p_f1_vs_stft": 0.1,
                "feasible": True,
                "measurement_valid": True,
            },
        ]
    )
    assert result["raw_p"] == 0.08


def test_h2_statistic_conditions_on_f1_macs_and_resolution():
    rng = np.random.default_rng(7)
    instability = rng.normal(size=(3, 12))
    source_f1 = rng.uniform(0.3, 0.9, size=(3, 12))
    held_f1 = source_f1 - 0.15 * instability + rng.normal(scale=0.03, size=(3, 12))
    statistic, detail = h2_statistic(
        instability,
        source_f1,
        held_f1,
        np.log(np.linspace(1.0e5, 1.0e7, 12)),
        np.asarray([8, 16, 32] * 4),
    )
    assert np.isfinite(statistic)
    assert len(detail) == 3


def test_h3_degeneracy_and_h4_deterministic_cost_fail_closed():
    h3 = h3_raw_p([{"family": "E", "cosine": 0.996, "nmae": 0.009, "raw_p": 0.01}])
    assert h3["raw_p"] == 1.0
    h4 = h4_raw_p(
        p_performance_noninferiority=0.01,
        p_latency_superiority=0.02,
        bytes_strictly_reduced=False,
        macs_strictly_reduced=True,
        measurement_valid=True,
    )
    assert h4["raw_p"] == 1.0


def test_holm_is_all_or_nothing_and_m1_precedence_is_exact():
    decided = finalize_hypotheses(
        {key: {"status": "ok", "raw_p": 0.001} for key in ("H1", "H2", "H3", "H4")}
    )
    assert all(row["status"] == "supported" for row in decided["hypotheses"].values())
    blocked = finalize_hypotheses(
        {"H1": {"status": "ok", "raw_p": 0.01}, "H2": {"status": "untestable"}}
    )
    assert blocked["status"] == "untestable"
    m1 = m1_state_machine(
        eligible=True,
        integrity_pass=True,
        calibration_pass=True,
        relation_coverage_by_nuisance={"awgn": 0.8},
        relation_recording_count=30,
        delta_gate=0.02,
        delta_gate_ci_low=0.001,
        delta_shuffle=0.03,
        delta_shuffle_ci_low=0.002,
        worst_domain_ci_low=-0.005,
        compute_measurement_valid=True,
        parameter_ratio=1.05,
        mac_ratio=1.05,
        latency_ratio=1.10,
    )
    assert m1["status"] == "supported"


def test_registered_metrics_are_emitted_and_detection_threshold_is_domain_safe():
    metrics = classification_metrics(np.asarray([0, 0, 1, 1]), np.asarray([0, 1, 1, 1]))
    assert "balanced_accuracy" in metrics
    assert "normalized_confusion_matrix" in metrics
    assert set(metrics["per_class"]["0"]) == {"precision", "recall", "f1", "support"}
    calibration = calibration_metrics(
        np.asarray([0, 1]), np.asarray([[0.8, 0.2], [0.4, 0.6]]), np.asarray([0, 1])
    )
    assert 0 <= calibration["ece"] <= 1
    threshold = source_threshold_by_domain(
        np.asarray([0, 0, 0, 0]),
        np.asarray([0.1, 0.2, 0.3, 0.9]),
        np.asarray(["a", "a", "b", "b"]),
    )
    assert threshold["threshold"] == 0.9
    assert normalized_curve_area([-10, 0, 10], [0.2, 0.6, 1.0]) > 0
