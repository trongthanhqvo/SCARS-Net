from __future__ import annotations

import numpy as np
import pytest

from scars.evaluation.classification import recording_level_metrics
from scars.evaluation.statistics import (
    _checked_correlation,
    _residual_projector,
    coherent_freedman_lane,
    holm_correction,
    midranks,
)
from scars.selection.pareto import ObjectiveRecord, pareto_front, select_unique
from scars.selection.sensitivity import (
    append_invariance_check,
    family_linf_displacement,
    recording_family_domain_reduce,
)


def _record(stable_id, sensitivity, f1, latency):
    return ObjectiveRecord(stable_id, sensitivity, f1, latency, 10.0, 100.0, True)


def test_pareto_uses_exact_three_axes_and_unique_source_rule():
    records = [
        _record("A", 0.2, 0.80, 2.0),
        _record("B", 0.1, 0.79, 1.0),
        _record("C", 0.3, 0.70, 3.0),
    ]
    assert pareto_front(records) == ["A", "B"]
    selected, front = select_unique(records, source_f1_absolute_tolerance=0.01)
    assert front == ["A", "B"]
    assert selected == "B"


def test_pareto_rejects_duplicate_stable_ids():
    records = [_record("same", 0.1, 0.8, 1.0), _record("same", 0.2, 0.7, 2.0)]
    with pytest.raises(ValueError, match="unique"):
        pareto_front(records)


def test_pareto_objective_has_no_target_metric_field():
    assert set(ObjectiveRecord.__dataclass_fields__) == {
        "stable_id",
        "nuisance_sensitivity",
        "source_selection_macro_f1",
        "batch1_latency_ms",
        "bytes_per_sample",
        "estimated_macs",
        "feasible",
    }


def test_linf_append_invariance_and_family_max():
    clean = np.asarray([[0.0, 1.0], [0.5, 0.5]])
    perturbed = np.asarray([[0.4, 1.0], [0.5, 0.8]])
    np.testing.assert_allclose(family_linf_displacement(clean, perturbed), [0.4, 0.3])
    assert append_invariance_check(clean, perturbed, 31)


def test_family_median_precedes_family_max_noncommuting_counterexample():
    # max-per-window then median would be 1; the registered order is 0.
    families = {
        "W": np.asarray([1.0, 0.0, 0.0]),
        "C": np.asarray([0.0, 1.0, 0.0]),
    }
    recording_ids = np.asarray(["r", "r", "r"])
    domains = np.asarray(["d", "d", "d"])
    assert recording_family_domain_reduce(families, recording_ids, domains) == 0.0


def test_appended_zero_displacement_family_does_not_change_recording_statistic():
    base = {"W": np.asarray([0.1, 0.4, 0.2])}
    augmented = {**base, "C": np.zeros(3)}
    ids = np.asarray(["r", "r", "r"])
    domains = np.asarray(["d", "d", "d"])
    assert recording_family_domain_reduce(base, ids, domains) == recording_family_domain_reduce(
        augmented, ids, domains
    )


def test_freedman_lane_is_deterministic_and_domain_balanced():
    rng = np.random.default_rng(3)
    source = rng.uniform(0.55, 0.9, size=(3, 12))
    instability = rng.uniform(0.05, 0.8, size=(3, 12))
    degradation = 0.2 * source + 0.5 * instability + rng.normal(0, 0.03, size=(3, 12))
    first = coherent_freedman_lane(instability, degradation, source, seed=7, permutations=199)
    second = coherent_freedman_lane(instability, degradation, source, seed=7, permutations=199)
    assert first["status"] == "ok"
    assert first["tail_fraction_diagnostic"] == second["tail_fraction_diagnostic"]
    assert first["estimate"] == second["estimate"]
    assert len(first["per_domain"]) == 3
    assert first["inferential_use"] == "forbidden_diagnostic_only"


def test_nonconstant_nuisance_projector_is_not_permutation_invariant():
    projector = _residual_projector(np.arange(1, 13, dtype=float))
    permutation = np.eye(12)
    permutation[[0, 1]] = permutation[[1, 0]]
    assert np.linalg.norm(permutation @ projector @ permutation.T - projector) > 0.1


def test_freedman_lane_constructed_permutation_degeneracy_is_detected():
    source_ranks = np.arange(1, 13, dtype=float)
    degradation_order = np.asarray([5, 10, 6, 2, 8, 9, 7, 4, 12, 3, 1, 11], dtype=float)
    projector = _residual_projector(source_ranks)
    residual = projector @ midranks(degradation_order)
    offending = np.argsort(degradation_order)
    permuted_residual = projector @ residual[offending]
    assert np.linalg.norm(residual) > 1.0
    assert _checked_correlation(permuted_residual, residual, 1.0e-10) is None


def test_holm_is_monotone_in_sorted_order():
    adjusted = holm_correction({"H1": 0.01, "H2": 0.03, "H3": 0.2, "H4": 0.04})
    assert adjusted["H1"] <= adjusted["H2"] <= adjusted["H4"] <= adjusted["H3"]


def test_recording_metrics_aggregate_windows_before_scoring():
    labels = np.asarray([0, 0, 1, 1])
    recording_ids = np.asarray(["a", "a", "b", "b"])
    probabilities = np.asarray([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8], [0.4, 0.6]])
    metrics = recording_level_metrics(labels, recording_ids, probabilities, np.asarray([0, 1]))
    assert metrics["macro_f1"] == 1.0
    assert metrics["recording_count"] == 2
    assert metrics["replication_unit"] == "recording"
