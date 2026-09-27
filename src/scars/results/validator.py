from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from scars.baselines import H2_BANK_ORDER, frozen_external_sota_registry
from scars.evaluation.classification import classification_metrics, macro_f1
from scars.evaluation.statistics import holm_correction
from scars.evaluation.decisions import (
    DECISION_ENGINE_VERSION,
    finalize_hypotheses,
    h3_raw_p,
    h4_raw_p,
    m1_state_machine,
)
from scars.results.registry import (
    CONFIRMATORY_H2_CONFIGURATION_ORDER,
    RECOGNITION_CONDITIONS,
    manuscript_registry_envelope,
)
from .schema import REQUIRED_TOP_LEVEL


class ResultValidationError(ValueError):
    pass


# A decided real artifact is deliberately impossible until the registered
# H1/H3/H4 inferential procedures and their raw-record recomputation are
# implemented.  This constant must become a versioned non-null identifier only
# in a protocol revision made before target access and accompanied by tests.
PUBLICATION_DECISION_ENGINE_VERSION: str | None = DECISION_ENGINE_VERSION


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _close(left: Any, right: Any, tolerance: float = 1.0e-12) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=tolerance))
    except (TypeError, ValueError):
        return False


def _assert_finite(value: Any, path: str) -> None:
    if isinstance(value, float) and not np.isfinite(value):
        raise ResultValidationError(f"Non-finite value at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")


def validate_results(
    payload: dict[str, Any],
    artifact_root: Path | None = None,
    *,
    independent_recompute: bool = True,
) -> None:
    missing = [key for key in REQUIRED_TOP_LEVEL if key not in payload]
    if missing:
        raise ResultValidationError(f"Missing top-level keys: {missing}")
    if payload["evidence_type"] not in {"synthetic_dev", "real_confirmatory", "pilot_two_dataset", "pilot_three_dataset"}:
        raise ResultValidationError("Invalid evidence_type")
    if payload["evidence_type"] in {"pilot_two_dataset", "pilot_three_dataset"}:
        _validate_pilot(payload, artifact_root)
        _assert_finite(payload, "results")
        return
    if (
        payload.get("schema_version") == "scars-canonical-results-2.0"
        and payload["evidence_type"] == "real_confirmatory"
        and payload.get("statistics", {}).get("decision_engine_version")
        == DECISION_ENGINE_VERSION
    ):
        _validate_canonical_v2(payload, artifact_root, independent_recompute)
        _assert_finite(payload, "results")
        return
    if payload["evidence_type"] == "synthetic_dev":
        invalid = [
            key
            for key, item in payload["hypotheses"].items()
            if item.get("status") not in {"open", "blocked"}
        ]
        if invalid:
            raise ResultValidationError(f"Synthetic evidence cannot decide hypotheses: {invalid}")
    else:
        _validate_real_confirmatory(payload, artifact_root)
    for key in ("H1", "H2", "H3", "H4"):
        if key not in payload["hypotheses"]:
            raise ResultValidationError(f"Missing hypothesis {key}")
    for fold in payload["held_domain"].get("folds", []):
        if "fold_id" not in fold or "conditions" not in fold:
            raise ResultValidationError("Held-domain fold lacks fold_id/conditions")
    _assert_finite(payload, "results")


def _validate_pilot(payload, artifact_root):
    from scars.experiment.pilot_policy import policy_for
    POLICY = policy_for({"campaign_mode": payload["evidence_type"]})
    if payload.get("pilot_policy") != POLICY or payload["run"].get("status") != "finalized":
        raise ResultValidationError("Missing finalized amended pilot contract")
    if artifact_root is None:
        raise ResultValidationError("Pilot validation requires the campaign artifacts")
    root = Path(artifact_root).resolve()
    artifacts = payload.get("provenance", {}).get("artifacts", [])
    if not artifacts:
        raise ResultValidationError("Pilot artifact hashes are missing")
    for entry in artifacts:
        path = (root / entry["path"]).resolve()
        if root not in path.parents or not path.is_file():
            raise ResultValidationError("Invalid pilot artifact path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ResultValidationError("Pilot artifact hash mismatch")
    if set(payload["datasets"]["domains"]) != set(POLICY["datasets"]):
        raise ResultValidationError("Pilot dataset set differs from its registered policy")
    if any(payload["hypotheses"].get(key, {}).get("status") != "open" for key in ("H1", "H2", "H3", "H4")):
        raise ResultValidationError("Pilot cannot confirm hypotheses")
    folds = payload["held_domain"]["folds"]
    if len(folds) != len(POLICY["datasets"]) or len({fold["fold_id"] for fold in folds}) != len(POLICY["datasets"]):
        raise ResultValidationError("Pilot requires one distinct held fold per dataset")
    primary = []
    for fold in folds:
        if fold.get("status") != "complete" or fold.get("target_reads") != 1:
            raise ResultValidationError("Incomplete pilot target fold")
        if not set(RECOGNITION_CONDITIONS).issubset(fold["conditions"]):
            raise ResultValidationError("Pilot recognition registry incomplete")
        for condition, record in fold["conditions"].items():
            rows = record["seeds"]
            if len(rows) != (1 if condition == "fixed_wavelet_subbands" else 5):
                raise ResultValidationError("Pilot seed count mismatch")
            reference = rows[0]["metrics"]
            order = reference["recording_order"]
            if len(set(order)) != len(order):
                raise ResultValidationError("Duplicate inferential recording")
            for row in rows:
                metric = row["metrics"]
                for key in ("recording_order", "recording_truth", "probability_class_order"):
                    if metric[key] != reference[key]:
                        raise ResultValidationError("Pilot ensemble alignment mismatch")
                probabilities = np.asarray(metric["recording_probabilities"], dtype=float)
                if probabilities.shape != (len(order), len(metric["probability_class_order"])) or not np.allclose(probabilities.sum(axis=1), 1):
                    raise ResultValidationError("Invalid pilot probabilities")
                if np.any(probabilities < 0) or metric.get("replication_unit") != "recording":
                    raise ResultValidationError("Invalid pilot inferential unit/probabilities")
        from scars.cli.finalize_results import _ensemble_seed_rows
        truth, prediction, _, _ = _ensemble_seed_rows(fold["conditions"]["pcrd"]["seeds"])
        primary.append(macro_f1(truth, prediction))
    if not np.isclose(payload["primary_metric"]["value"], np.mean(primary)):
        raise ResultValidationError("Pilot primary metric does not recompute")


def _validate_canonical_v2(
    payload: dict[str, Any],
    artifact_root: Path | None,
    independent_recompute: bool,
) -> None:
    if payload.get("run", {}).get("status") != "finalized":
        raise ResultValidationError("Canonical real run must be finalized")
    if artifact_root is None:
        raise ResultValidationError("Canonical real results require an artifact root")
    if payload.get("statistics", {}).get("decision_engine_version") != DECISION_ENGINE_VERSION:
        raise ResultValidationError("Decision engine version is missing or stale")
    static_metadata = payload.get("static_metadata")
    if (
        not isinstance(static_metadata, dict)
        or static_metadata.get("status") != "computed_without_real_iq_or_training"
        or static_metadata.get("empirical_evidence") is not False
    ):
        raise ResultValidationError("Canonical evidence lacks validated non-empirical static metadata")
    required_static_nulls = {
        "dataset_eligibility",
        "shared_ontology_and_class_count_K",
        "recording_group_and_class_counts",
        "source_fitted_normalizers",
        "source_frozen_cyclic_frequency_values",
        "source_selected_active_family_count_A",
        "capacity_matched_early_fusion_width",
        "trained_checkpoint_parameters_if_selection_changes_topology",
        "representation_and_model_latency",
        "peak_memory_and_energy",
        "recognition_detection_robustness_and_calibration_metrics",
        "confidence_intervals_effect_sizes_and_p_values",
        "H1_H4_and_M1_decisions",
    }
    real_run_only = static_metadata.get("real_run_only")
    if not isinstance(real_run_only, dict) or set(real_run_only) != required_static_nulls:
        raise ResultValidationError("Static metadata real-run-only registry is missing or stale")
    if any(value is not None for value in real_run_only.values()):
        raise ResultValidationError("Static metadata populated a real-run-only value")
    if payload.get("registry") != manuscript_registry_envelope()["registry"]:
        raise ResultValidationError("Result registry differs from the frozen runtime/manuscript registry")
    artifact_root = Path(artifact_root).resolve()
    artifact_records = payload.get("provenance", {}).get("artifact_manifest", [])
    required_artifact_roles = {
        "recordings_manifest",
        "split_manifest",
        "source_freeze",
        "model_freeze",
        "target_metrics",
    }
    if not isinstance(artifact_records, list) or not required_artifact_roles.issubset(
        {item.get("role") for item in artifact_records if isinstance(item, dict)}
    ):
        raise ResultValidationError("Canonical evidence lacks its frozen artifact manifest")
    for item in artifact_records:
        path = (artifact_root / str(item.get("path", ""))).resolve()
        try:
            path.relative_to(artifact_root)
        except ValueError as error:
            raise ResultValidationError("Artifact provenance escapes the campaign directory") from error
        if not path.is_file() or not _is_sha256(item.get("sha256")):
            raise ResultValidationError(f"Missing or invalid frozen artifact: {item.get('path')}")
        if _sha256(path) != item["sha256"]:
            raise ResultValidationError(f"Frozen artifact hash mismatch: {item.get('path')}")
    recordings_path = next(
        artifact_root / item["path"]
        for item in artifact_records
        if item.get("role") == "recordings_manifest"
    )
    recording_manifest = json.loads(recordings_path.read_text(encoding="utf-8"))
    manifest_truth = {
        str(item["recording_id"]): str(item["label"])
        for item in recording_manifest.get("recordings", [])
    }
    if not all(
        item.get("metadata", {}).get(
            "temporal_adjacency_subsumed_by_split_group_verified", False
        )
        for item in recording_manifest.get("recordings", [])
    ):
        raise ResultValidationError(
            "Frozen recording manifest lacks verified temporal-group coverage"
        )
    rows = payload.get("tables", {}).get("tab_primary_results", {}).get("rows")
    if not isinstance(rows, list) or [row.get("condition_id") for row in rows] != RECOGNITION_CONDITIONS:
        raise ResultValidationError("Primary result rows do not match the frozen 15-condition registry")
    required_row = {
        "condition_id",
        "recording_count",
        "mean_macro_f1",
        "ci95_low",
        "ci95_high",
        "worst_domain_macro_f1",
        "balanced_accuracy",
        "parameters",
        "macs",
        "latency_ms",
    }
    if any(not required_row.issubset(row) for row in rows):
        raise ResultValidationError("A primary result row is incomplete")
    folds = payload.get("held_domain", {}).get("folds", [])
    if len(folds) < 3:
        raise ResultValidationError("At least three held domains are required")
    recomputed: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in RECOGNITION_CONDITIONS
    }
    for fold in folds:
        if fold.get("target_reads") != 1:
            raise ResultValidationError("Every held fold must have exactly one logical target read")
        conditions = fold.get("conditions", {})
        if not set(RECOGNITION_CONDITIONS).issubset(conditions):
            raise ResultValidationError("Held fold lacks a registered recognition condition")
        for condition in RECOGNITION_CONDITIONS:
            seeds = conditions[condition].get("seeds", [])
            expected = 1 if condition == "fixed_wavelet_subbands" else 5
            if len(seeds) != expected:
                raise ResultValidationError(f"{condition} has an invalid seed count")
            for seed in seeds:
                metric = seed.get("metrics", {})
                if metric.get("replication_unit") != "recording":
                    raise ResultValidationError("Windows cannot be inferential replicates")
                if metric.get("recording_count") != len(metric.get("recording_order", [])):
                    raise ResultValidationError("Recording metric count/order mismatch")
                probability = np.asarray(metric.get("recording_probabilities", []), dtype=float)
                classes = metric.get("probability_class_order", [])
                if probability.shape != (metric.get("recording_count"), len(classes)):
                    raise ResultValidationError("Recording probability tensor/class order mismatch")
            reference = seeds[0]["metrics"]
            recording_order = list(map(str, reference.get("recording_order", [])))
            if len(recording_order) != len(set(recording_order)):
                raise ResultValidationError("Target recording order contains duplicates")
            if any(item not in manifest_truth for item in recording_order):
                raise ResultValidationError("Target prediction references a recording outside the manifest")
            expected_truth = [manifest_truth[item] for item in recording_order]
            if list(map(str, reference.get("recording_truth", []))) != expected_truth:
                raise ResultValidationError("Target recording truth does not match the frozen manifest")
            if any(
                seed["metrics"].get("recording_order") != reference.get("recording_order")
                or seed["metrics"].get("recording_truth") != reference.get("recording_truth")
                or seed["metrics"].get("probability_class_order")
                != reference.get("probability_class_order")
                for seed in seeds
            ):
                raise ResultValidationError("Five-seed prediction vectors are not aligned")
            truth = np.asarray(reference["recording_truth"], dtype=object)
            classes = np.asarray(reference["probability_class_order"], dtype=object)
            probability = np.mean(
                [np.asarray(seed["metrics"]["recording_probabilities"], dtype=float) for seed in seeds],
                axis=0,
            )
            predicted = classes[np.argmax(probability, axis=1)]
            recomputed[condition].append(
                {
                    **classification_metrics(truth, predicted),
                    "recording_count": len(truth),
                }
            )
            cost = conditions[condition].get("cost", {})
            if condition != "fixed_wavelet_subbands":
                per_seed = cost.get("per_seed", [])
                if len(per_seed) != 5 or not all(
                    item.get("latency", {}).get("warmups") == 20
                    and item.get("latency", {}).get("repeats") == 100
                    for item in per_seed
                ):
                    raise ResultValidationError("Learned-condition latency is not measured for all five seeds")
                expected_parameters = sum(float(item["parameters"]) for item in per_seed)
                expected_model_macs = sum(
                    float(item.get("model_macs", item["macs"])) for item in per_seed
                )
                expected_model_latency = sum(
                    float(item["latency"].get("model_median_ms", item["latency"]["median_ms"]))
                    for item in per_seed
                )
                if not _close(cost.get("parameters"), expected_parameters):
                    raise ResultValidationError("Five-model parameters are not summed")
                if not _close(cost.get("model_macs"), expected_model_macs):
                    raise ResultValidationError("Five-model MACs are not summed")
                if not _close(cost.get("latency", {}).get("model_median_ms"), expected_model_latency):
                    raise ResultValidationError("Five-model sequential latency is not summed")
    roles = ("source_fit", "source_calibration", "source_selection", "source_validation", "held_target")
    for fold in payload.get("splits", {}).get("folds", []):
        sets = [set(fold.get(role, [])) for role in roles]
        if any(sets[left] & sets[right] for left in range(len(sets)) for right in range(left + 1, len(sets))):
            raise ResultValidationError("Physical recording overlap across frozen roles")
    if set(payload.get("hypotheses", {})) != {"H1", "H2", "H3", "H4"}:
        raise ResultValidationError("H1-H4 decision ledger is incomplete")
    external_rows = payload.get("tables", {}).get("tab_external_sota", {}).get("rows", [])
    expected_external = frozen_external_sota_registry()
    if [item.get("condition_id") for item in external_rows] != [
        item["condition_id"] for item in expected_external
    ]:
        raise ResultValidationError("External SOTA registry is missing or reordered")
    for observed, expected in zip(external_rows, expected_external):
        if observed.get("eligibility") != expected["implementation_status"]:
            raise ResultValidationError("External SOTA eligibility status drifted")
        if observed.get("ineligibility_reason") != expected["reason"]:
            raise ResultValidationError("External SOTA frozen reason drifted")
    h2_order = tuple(
        payload.get("statistics", {}).get("h2", {}).get("configuration_order", [])
    )
    if h2_order != CONFIRMATORY_H2_CONFIGURATION_ORDER:
        raise ResultValidationError("Canonical H2 registry differs from the shared confirmatory order")
    row_by_condition = {row["condition_id"]: row for row in rows}
    for condition, values in recomputed.items():
        row = row_by_condition[condition]
        expected = {
            "mean_macro_f1": np.mean([item["macro_f1"] for item in values]),
            "worst_domain_macro_f1": np.min([item["macro_f1"] for item in values]),
            "balanced_accuracy": np.mean([item["balanced_accuracy"] for item in values]),
            "recording_count": sum(item["recording_count"] for item in values),
        }
        if any(not _close(row[key], value) for key, value in expected.items()):
            raise ResultValidationError(f"{condition} summary does not recompute from seed-ensemble probabilities")
        fold_costs = [fold["conditions"][condition]["cost"] for fold in folds]
        expected_cost = {
            "latency_ms": np.mean([item["latency"]["median_ms"] for item in fold_costs]),
            "macs": np.mean([item["macs"] for item in fold_costs if item.get("macs") is not None]),
        }
        if any(not _close(row[key], value) for key, value in expected_cost.items()):
            raise ResultValidationError(f"{condition} cost summary does not recompute from fold costs")
    eligibility = payload.get("eligibility", {})
    if eligibility.get("eligible_domains", 0) < 3:
        raise ResultValidationError("Canonical evidence has fewer than three eligible domains")
    integrity = payload.get("integrity", {})
    if integrity.get("status") != "passed" or integrity.get("critical_failures"):
        raise ResultValidationError("Canonical evidence failed an integrity gate")
    dataset_checksums = payload.get("provenance", {}).get("dataset_checksums", {})
    if not dataset_checksums or not all(_is_sha256(value) for value in dataset_checksums.values()):
        raise ResultValidationError("Canonical evidence lacks complete dataset SHA-256 provenance")
    statistics = payload.get("statistics", {})
    h1_stored = statistics.get("h1", {})
    if h1_stored.get("status") == "ok":
        candidates = h1_stored.get("candidates", [])
        if not candidates:
            raise ResultValidationError("H1 lacks its registered candidate decision inputs")
        for candidate in candidates:
            expected_iut = (
                max(float(candidate["p_instability"]), float(candidate["p_f1"]))
                if candidate.get("feasible")
                else 1.0
            )
            if not _close(candidate.get("p_iut"), expected_iut):
                raise ResultValidationError("H1 candidate IUT does not recompute")
        h1_recomputed = {
            "status": "ok",
            "raw_p": min(
                1.0,
                len(candidates) * min(float(item["p_iut"]) for item in candidates),
            ),
        }
    else:
        h1_recomputed = {"status": "untestable", "reason": h1_stored.get("reason")}
    h2_stored = statistics.get("h2", {})
    h2_recomputed = {
        "status": h2_stored.get("status"),
        "raw_p": h2_stored.get("raw_p"),
        "reason": h2_stored.get("reason"),
    }
    h3_stored = statistics.get("h3", {})
    h3_recomputed = h3_raw_p(h3_stored.get("components", []))
    h4_stored = statistics.get("h4", {})
    h4_inputs = h4_stored.get("inputs")
    if not isinstance(h4_inputs, dict):
        raise ResultValidationError("H4 lacks its registered decision inputs")
    h4_recomputed = h4_raw_p(**h4_inputs)
    raw_records = {
        "H1": h1_recomputed,
        "H2": h2_recomputed,
        "H3": h3_recomputed,
        "H4": h4_recomputed,
    }
    for key, recomputed_record in raw_records.items():
        stored = statistics.get(key.lower(), {})
        if stored.get("status") != recomputed_record.get("status"):
            raise ResultValidationError(f"{key} testability does not recompute")
        if recomputed_record.get("status") == "ok" and not _close(
            stored.get("raw_p"), recomputed_record.get("raw_p")
        ):
            raise ResultValidationError(f"{key} raw p-value does not recompute")
    expected_decisions = finalize_hypotheses(raw_records)
    observed_holm = statistics.get("holm")
    if expected_decisions["holm"] is None:
        if observed_holm is not None:
            raise ResultValidationError("Stored Holm family must be null when any hypothesis is untestable")
    elif not isinstance(observed_holm, dict) or any(
        not _close(observed_holm.get(key), expected_decisions["holm"][key])
        for key in expected_decisions["holm"]
    ):
        raise ResultValidationError("Stored Holm family does not recompute")
    for key in ("H1", "H2", "H3", "H4"):
        expected = expected_decisions["hypotheses"][key]
        observed = payload["hypotheses"][key]
        if observed.get("status") != expected.get("status"):
            raise ResultValidationError(f"{key} decision status does not recompute")
        if raw_records[key].get("status") == "ok" and not _close(
            observed.get("raw_p"), raw_records[key].get("raw_p")
        ):
            raise ResultValidationError(f"{key} ledger raw p-value does not recompute")

    mechanism = payload.get("mechanism", {}).get("m1", {})
    m1_inputs = mechanism.get("state_machine_inputs", {})
    compute_gate = mechanism.get("compute_gate", {})
    m1_recomputed = m1_state_machine(
        eligible=bool(m1_inputs.get("eligible")),
        integrity_pass=bool(m1_inputs.get("integrity_pass")),
        calibration_pass=bool(m1_inputs.get("calibration_pass")),
        relation_coverage_by_nuisance=m1_inputs.get("relation_coverage_by_nuisance", {}),
        relation_recording_count=int(m1_inputs.get("relation_recording_count", 0)),
        delta_gate=float(mechanism["delta_gate"]["estimate"]),
        delta_gate_ci_low=float(mechanism["delta_gate"]["ci95"][0]),
        delta_shuffle=float(mechanism["delta_shuffle"]["estimate"]),
        delta_shuffle_ci_low=float(mechanism["delta_shuffle"]["ci95"][0]),
        worst_domain_ci_low=float(mechanism["worst_domain_difference"]["ci95"][0]),
        compute_measurement_valid=bool(m1_inputs.get("compute_measurement_valid")),
        parameter_ratio=float(compute_gate["parameter_ratio"]),
        mac_ratio=float(compute_gate["mac_ratio"]),
        latency_ratio=float(compute_gate["latency_ratio"]),
    )
    if mechanism.get("status") != m1_recomputed.get("status") or mechanism.get(
        "reason"
    ) != m1_recomputed.get("reason"):
        raise ResultValidationError("M1 state-machine decision does not recompute")
    pcrd_row = row_by_condition["pcrd"]
    gate_row = row_by_condition["ordinary_gate"]
    expected_worst = pcrd_row["worst_domain_macro_f1"] - gate_row["worst_domain_macro_f1"]
    if not _close(mechanism["worst_domain_difference"]["estimate"], expected_worst):
        raise ResultValidationError("M1 worst-domain estimand is inconsistent with the protocol")
    primary = payload.get("primary_metric", {})
    if primary.get("best_condition") != "pcrd" or not _close(
        primary.get("value"), pcrd_row["mean_macro_f1"]
    ):
        raise ResultValidationError("Primary metric is not the source-frozen PCRD aggregate")
    if independent_recompute:
        _independent_recompute(payload, artifact_root)


def _comparison_projection(payload: dict[str, Any]) -> dict[str, Any]:
    projection = deepcopy(payload)
    projection.get("run", {}).pop("created_at", None)
    return projection


def _independent_recompute(payload: dict[str, Any], artifact_root: Path) -> None:
    """Re-run canonical finalization from sealed source/model/target artifacts.

    This deliberately uses the same versioned estimator implementation but a
    fresh read of every frozen artifact.  It catches a modified or stale
    results.json even when its internal summaries and decision arithmetic are
    self-consistent.
    """
    from scars.cli.finalize_results import run as finalize_results

    temporary = artifact_root / f".validator-recomputed-{uuid4().hex}.json"
    try:
        finalize_results(
            campaign_dir=artifact_root,
            output=temporary,
            resamples=10_000,
            validate_output=False,
        )
        recomputed = json.loads(temporary.read_text(encoding="utf-8"))
    finally:
        temporary.unlink(missing_ok=True)
    if _comparison_projection(payload) != _comparison_projection(recomputed):
        raise ResultValidationError(
            "Canonical results differ from an independent finalization of frozen artifacts"
        )


def _validate_real_confirmatory(payload: dict[str, Any], artifact_root: Path | None) -> None:
    """Fail closed before a file may call itself canonical real evidence."""
    if payload.get("run", {}).get("status") != "finalized":
        raise ResultValidationError("real_confirmatory run.status must be finalized")
    if artifact_root is None:
        raise ResultValidationError("Real validation requires the result artifact directory")
    artifact_root = Path(artifact_root).resolve()
    provenance = payload.get("provenance", {})
    for key in (
        "recordings_manifest_sha256",
        "acquisition_manifest_sha256",
        "combined_config_hash",
        "ontology_sha256",
        "label_map_sha256",
        "source_tree_sha256",
    ):
        if not _is_sha256(provenance.get(key)):
            raise ResultValidationError(f"Missing or invalid provenance hash: {key}")
    gate = payload.get("datasets", {}).get("acquisition_gate")
    if not isinstance(gate, dict):
        raise ResultValidationError("Missing acquisition gate")
    domain_gate = gate.get("domain_gate", gate)
    if not (domain_gate.get("gate_open") or domain_gate.get("confirmatory_open")):
        raise ResultValidationError("Confirmatory acquisition gate is not open")
    eligible = domain_gate.get("eligible_domain_ids", payload.get("datasets", {}).get("domains", []))
    complete_count = domain_gate.get("complete_domain_count", len(eligible or []))
    if complete_count < 3 or len(eligible or []) < 3:
        raise ResultValidationError("At least three complete eligible domains are required")
    dataset_hashes = provenance.get("dataset_manifest_sha256")
    if not isinstance(dataset_hashes, dict) or set(dataset_hashes) != set(eligible):
        raise ResultValidationError("Dataset-manifest hashes do not cover every eligible domain")
    if not all(_is_sha256(value) for value in dataset_hashes.values()):
        raise ResultValidationError("Every eligible dataset manifest needs a SHA-256")
    compute_gate = gate.get("compute_gate")
    if not isinstance(compute_gate, dict) or not compute_gate.get("gate_open"):
        raise ResultValidationError("Confirmatory compute gate is not open")

    order = tuple(payload.get("representations", {}).get("configuration_order", []))
    if order != H2_BANK_ORDER:
        raise ResultValidationError("Real result does not contain the exact frozen H2 bank order")
    held_folds = payload.get("held_domain", {}).get("folds", [])
    source_folds = payload.get("source_fitting", {}).get("folds", [])
    selection_folds = payload.get("source_selection", {}).get("folds", [])
    split_folds = payload.get("splits", {}).get("folds", [])
    fold_count = len(held_folds)
    if fold_count < 3 or not (len(source_folds) == len(selection_folds) == len(split_folds) == fold_count):
        raise ResultValidationError("Real fold records are missing or have inconsistent counts")
    if not payload.get("splits", {}).get("manifest_hash"):
        raise ResultValidationError("Missing immutable split manifest hash")
    for fold in held_folds:
        if fold.get("target_reads") != 1:
            raise ResultValidationError(f"Fold {fold.get('fold_id')} must log exactly one target read")
        if fold.get("target_unlocks") != 1:
            raise ResultValidationError(f"Fold {fold.get('fold_id')} must log exactly one target unlock")
        if tuple(fold.get("configuration_order", [])) != H2_BANK_ORDER:
            raise ResultValidationError(f"Fold {fold.get('fold_id')} lacks the registered H2 order")
        if set(fold.get("conditions", {})) != set(H2_BANK_ORDER):
            raise ResultValidationError(f"Fold {fold.get('fold_id')} has an incomplete H2 bank")
        matching = next((item for item in split_folds if item.get("fold_id") == fold.get("fold_id")), None)
        if matching is None or set(fold.get("target_recording_ids", [])) != set(
            matching.get("held_target_recordings", [])
        ):
            raise ResultValidationError("Target read ledger does not cover the frozen held recordings")
        target_ids = set(fold["target_recording_ids"])
        if fold.get("selected_theta_hat") not in H2_BANK_ORDER:
            raise ResultValidationError("Held fold lacks its source-frozen selected configuration")
        for stable_id, metrics in fold["conditions"].items():
            required = {
                "macro_f1",
                "recording_count",
                "recording_order",
                "recording_truth",
                "recording_prediction",
                "replication_unit",
                "confusion_matrix",
                "label_order",
                "per_class_recall",
            }
            if not isinstance(metrics, dict) or not required.issubset(metrics):
                raise ResultValidationError(f"Incomplete recording metrics for {stable_id}")
            if metrics["replication_unit"] != "recording":
                raise ResultValidationError("Windows may not be treated as held-domain replicates")
            if set(metrics["recording_order"]) != target_ids:
                raise ResultValidationError(f"Condition {stable_id} does not cover every target recording")
            if metrics["recording_count"] != len(target_ids):
                raise ResultValidationError(f"Condition {stable_id} has inconsistent recording count")
            if not (
                len(metrics["recording_truth"])
                == len(metrics["recording_prediction"])
                == len(metrics["recording_order"])
            ):
                raise ResultValidationError(f"Condition {stable_id} has misaligned recording outputs")
            recomputed_f1 = macro_f1(
                np.asarray(metrics["recording_truth"]), np.asarray(metrics["recording_prediction"])
            )
            if not _close(metrics["macro_f1"], recomputed_f1):
                raise ResultValidationError(f"Condition {stable_id} macro-F1 does not recompute")
    for fold in source_folds:
        if fold.get("h2_bank_status") != "complete":
            raise ResultValidationError("Source H2 bank status is not complete")
        if tuple(fold.get("configuration_order", [])) != H2_BANK_ORDER:
            raise ResultValidationError("Source H2 configuration order drifted")
        if tuple(record.get("stable_id") for record in fold.get("records", [])) != H2_BANK_ORDER:
            raise ResultValidationError("Source fold lacks the exact ordered twelve H2 records")
        if not fold.get("source_freeze_sha256") or not fold.get("source_freeze_path"):
            raise ResultValidationError("Missing durable source-freeze provenance")
        relative = Path(fold["source_freeze_path"])
        candidate = (artifact_root / relative).resolve()
        try:
            candidate.relative_to(artifact_root)
        except ValueError as error:
            raise ResultValidationError("Source-freeze path escapes artifact directory") from error
        if not candidate.is_file() or _sha256(candidate) != fold["source_freeze_sha256"]:
            raise ResultValidationError("Source-freeze file is missing or its SHA-256 differs")
    for fold, source_fold in zip(selection_folds, source_folds):
        if not fold.get("source_artifact_hash"):
            raise ResultValidationError("Every source selection fold needs a frozen artifact hash")
        if fold["source_artifact_hash"] != source_fold["source_freeze_sha256"]:
            raise ResultValidationError("Source selection hash differs from durable freeze hash")
        if fold.get("selected_theta_hat") not in H2_BANK_ORDER:
            raise ResultValidationError("Source selection lacks a registered selected configuration")
    selection_by_fold = {fold["fold_id"]: fold for fold in selection_folds}
    selected_scores = []
    focal_id = payload.get("held_domain", {}).get("focal_condition_id")
    if focal_id != "wst_cyclic_resolution_16":
        raise ResultValidationError("Focal held-domain condition differs from the frozen H2 bank")
    focal_scores = []
    for fold in held_folds:
        source_selection = selection_by_fold.get(fold["fold_id"])
        if source_selection is None or fold["selected_theta_hat"] != source_selection["selected_theta_hat"]:
            raise ResultValidationError("Held selected condition differs from source-frozen selection")
        selected_scores.append(float(fold["conditions"][fold["selected_theta_hat"]]["macro_f1"]))
        focal_scores.append(float(fold["conditions"][focal_id]["macro_f1"]))
    if not _close(payload["held_domain"].get("focal_average_macro_f1"), np.mean(focal_scores)):
        raise ResultValidationError("Focal average macro-F1 does not equal the fold-level mean")
    if not _close(payload["held_domain"].get("focal_worst_macro_f1"), np.min(focal_scores)):
        raise ResultValidationError("Focal worst-domain macro-F1 does not equal the fold-level minimum")
    if not _close(payload["held_domain"].get("selected_average_macro_f1"), np.mean(selected_scores)):
        raise ResultValidationError("Selected average macro-F1 does not equal the fold-level mean")
    if not _close(payload["held_domain"].get("selected_worst_macro_f1"), np.min(selected_scores)):
        raise ResultValidationError("Selected worst-domain macro-F1 does not equal the fold-level minimum")
    if not _close(payload["held_domain"].get("average_macro_f1"), np.mean(selected_scores)):
        raise ResultValidationError("Held average macro-F1 does not equal the fold-level mean")
    if not _close(payload["held_domain"].get("worst_macro_f1"), np.min(selected_scores)):
        raise ResultValidationError("Held worst-domain macro-F1 does not equal the fold-level minimum")

    h2 = payload.get("statistics", {}).get("h2")
    if not isinstance(h2, dict) or h2.get("status") not in {"ok", "blocked"}:
        raise ResultValidationError("Missing finalized H2 statistical record")
    if tuple(h2.get("configuration_order", [])) != H2_BANK_ORDER:
        raise ResultValidationError("H2 statistical order differs from the registered bank")
    if h2.get("configuration_count") != len(H2_BANK_ORDER) or h2.get("no_deletion_gate") != "passed":
        raise ResultValidationError("H2 no-deletion gate is incomplete")
    if h2.get("status") == "ok" and (h2.get("permutations") != 99_999 or h2.get("seed") != 24_021):
        raise ResultValidationError("H2 permutation count/seed differs from registration")
    if h2.get("status") == "ok":
        if h2.get("inferential_use") != "forbidden_diagnostic_only":
            raise ResultValidationError("Current H2 residual permutation must be labeled diagnostic-only")
        interval = payload.get("statistics", {}).get("block_bootstrap", {}).get("H2")
        if not isinstance(interval, dict) or interval.get("seed") != 24_022:
            raise ResultValidationError("H2 nested bootstrap provenance is missing or unregistered")
        if interval.get("outer_unit") != "domain" or interval.get("inner_unit") != "recording_or_event":
            raise ResultValidationError("H2 bootstrap units differ from registration")
    measurement = payload.get("costs", {}).get("measurement_manifest")
    if not isinstance(measurement, dict) or not measurement.get("deployment_budget_gate_open"):
        raise ResultValidationError("Missing open deployment/timing measurement manifest")
    for key in ("h1", "h3", "h4"):
        if not isinstance(payload.get("statistics", {}).get(key), dict):
            raise ResultValidationError(f"Missing registered {key.upper()} arm")
    learned = payload["statistics"]["h3"].get("learned_probe_seed_status", {})
    if learned.get("status") != "complete" or learned.get("completed") != [11, 23, 37, 53, 71]:
        raise ResultValidationError("H3 five-seed learned-probe arm is incomplete")
    ablations = payload.get("ablations", {})
    if len(ablations.get("channel_masks", [])) < 3 or len(ablations.get("resolution", [])) != 3:
        raise ResultValidationError("Required channel/resolution ablations are incomplete")
    if not isinstance(ablations.get("controls"), dict) or "degeneracy" not in ablations["controls"]:
        raise ResultValidationError("Missing representation degeneracy control")
    audits = payload.get("leakage_audits", {})
    for key in ("duplicates", "near_duplicates", "temporal", "domain_probe", "label_shuffle", "background_only"):
        if audits.get(key) is None:
            raise ResultValidationError(f"Missing leakage/control audit: {key}")
    if audits["duplicates"].get("cross_domain_hashes"):
        raise ResultValidationError("Cross-domain exact duplicates invalidate confirmatory evidence")
    if audits["near_duplicates"].get("cross_domain_pair_count", 0) > 0:
        raise ResultValidationError("Cross-domain near duplicates invalidate confirmatory evidence")
    if audits["temporal"].get("status") != "passed" or audits["temporal"].get(
        "cross_partition_recording_violations"
    ):
        raise ResultValidationError("Temporal/group leakage audit failed")
    if len(audits["label_shuffle"].get("folds", [])) != fold_count:
        raise ResultValidationError("Label-shuffle audit does not cover every fold")
    invalid_status = [
        key
        for key in ("H1", "H2", "H3", "H4")
        if payload.get("hypotheses", {}).get(key, {}).get("status")
        not in {"supported", "unsupported", "blocked"}
    ]
    if invalid_status:
        raise ResultValidationError(f"Real finalized hypotheses have invalid/open status: {invalid_status}")
    arm_by_hypothesis = {"H1": "h1", "H2": "h2", "H3": "h3", "H4": "h4"}
    decided = [
        key
        for key in ("H1", "H2", "H3", "H4")
        if payload["hypotheses"][key]["status"] in {"supported", "unsupported"}
    ]
    if decided:
        if PUBLICATION_DECISION_ENGINE_VERSION is None:
            raise ResultValidationError(
                "Decided real hypotheses are refused: the registered publication decision engine is not implemented"
            )
        if set(decided) != {"H1", "H2", "H3", "H4"}:
            raise ResultValidationError("The registered Holm family cannot be partially decided")
        holm = payload.get("statistics", {}).get("holm")
        if not isinstance(holm, dict) or set(holm) != {"H1", "H2", "H3", "H4"}:
            raise ResultValidationError("A decided hypothesis requires the registered H1-H4 Holm family")
        raw_p_values: dict[str, float] = {}
        for hypothesis, arm_key in arm_by_hypothesis.items():
            arm = payload["statistics"].get(arm_key)
            if not isinstance(arm, dict) or arm.get("status") != "ok":
                raise ResultValidationError(f"Decided {hypothesis} requires a successful inferential arm")
            p_value = arm.get("p_value")
            if not isinstance(p_value, (int, float)) or not 0.0 <= float(p_value) <= 1.0:
                raise ResultValidationError(f"Decided {hypothesis} lacks a valid raw p-value")
            raw_p_values[hypothesis] = float(p_value)
        recomputed_holm = holm_correction(raw_p_values)
        for hypothesis in arm_by_hypothesis:
            if not _close(holm.get(hypothesis), recomputed_holm[hypothesis]):
                raise ResultValidationError("Stored Holm values do not recompute from registered raw p-values")
            hypothesis_record = payload["hypotheses"][hypothesis]
            if not _close(hypothesis_record.get("holm_adjusted_p"), holm[hypothesis]):
                raise ResultValidationError(f"{hypothesis} hypothesis record does not match Holm output")
            direction_holds = bool(payload["statistics"][arm_by_hypothesis[hypothesis]].get("direction_holds"))
            expected = "supported" if direction_holds and holm[hypothesis] <= 0.05 else "unsupported"
            if hypothesis_record["status"] != expected:
                raise ResultValidationError(f"{hypothesis} decision conflicts with its frozen rule")
    else:
        inconsistent = [
            hypothesis
            for hypothesis, arm_key in arm_by_hypothesis.items()
            if payload["hypotheses"][hypothesis]["status"] == "blocked"
            and isinstance(payload["statistics"].get(arm_key), dict)
            and payload["statistics"][arm_key].get("final_decision") in {"supported", "unsupported"}
        ]
        if inconsistent:
            raise ResultValidationError(f"Blocked hypotheses contain phantom final decisions: {inconsistent}")
