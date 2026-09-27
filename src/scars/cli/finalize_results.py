from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scars.baselines import frozen_external_sota_registry
from scars.evaluation.classification import calibration_metrics, classification_metrics, macro_f1
from scars.cli.collect_static_metadata import build_static_metadata
from scars.evaluation.detection import detection_metrics
from scars.evaluation.robustness import normalized_curve_area
from scars.evaluation.decisions import (
    DECISION_ENGINE_VERSION,
    H2DomainRecords,
    finalize_hypotheses,
    h1_raw_p,
    h2_hierarchical_bootstrap,
    h3_raw_p,
    h4_raw_p,
    m1_state_machine,
)
from scars.experiment.common import atomic_json, assert_frozen_environment
from scars.results.registry import ABLATION_CONDITIONS, RECOGNITION_CONDITIONS
from scars.results.schema import empty_results
from scars.results.provenance import sha256_file
from scars.results.validator import validate_results
from scars.state import RunPhase, RunState


BOOTSTRAP_SEED = 24022


def _ensemble_seed_rows(rows):
    """Average seed probability vectors before any recording-level metric."""
    if not rows:
        raise ValueError("Cannot ensemble zero seed rows")
    reference = rows[0]["metrics"]
    truth = np.asarray(reference["recording_truth"], dtype=object)
    order = reference["recording_order"]
    classes = np.asarray(reference["probability_class_order"], dtype=object)
    probabilities = []
    for row in rows:
        metric = row["metrics"]
        if metric["recording_truth"] != reference["recording_truth"]:
            raise ValueError("Seed ensemble truth mismatch")
        if metric["recording_order"] != order:
            raise ValueError("Seed ensemble recording-order mismatch")
        if metric["probability_class_order"] != reference["probability_class_order"]:
            raise ValueError("Seed ensemble class-order mismatch")
        probabilities.append(np.asarray(metric["recording_probabilities"], dtype=float))
    probability = np.mean(probabilities, axis=0)
    prediction = classes[np.argmax(probability, axis=1)]
    return truth, prediction, probability, classes


def _stratified_draw(truth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = []
    for label in sorted(set(truth.tolist()), key=str):
        indices = np.flatnonzero(truth == label)
        output.extend(indices[rng.integers(0, len(indices), len(indices))].tolist())
    return np.asarray(output, dtype=int)


def _source_domain_class_draw(
    truth: np.ndarray, domains: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    output = []
    for domain, label in sorted(set(zip(domains.astype(str), truth.astype(str)))):
        indices = np.flatnonzero(
            (domains.astype(str) == domain) & (truth.astype(str) == label)
        )
        output.extend(indices[rng.integers(0, len(indices), len(indices))].tolist())
    return np.asarray(output, dtype=int)


def _target_condition_draws(folds, condition: str, resamples: int, seed: int):
    observed_domains = []
    prepared = []
    for fold in folds:
        rows = fold["conditions"][condition]["seeds"]
        if not rows:
            raise ValueError(f"No target seed records for {condition}")
        truth, prediction, _, _ = _ensemble_seed_rows(rows)
        observed_domains.append(macro_f1(truth, prediction))
        prepared.append((truth, prediction))
    rng = np.random.default_rng(seed)
    draws = np.zeros(resamples, dtype=np.float64)
    for draw_index in range(resamples):
        domain_values = []
        for truth, prediction in prepared:
            indices = _stratified_draw(truth, rng)
            domain_values.append(macro_f1(truth[indices], prediction[indices]))
        draws[draw_index] = np.mean(domain_values)
    return np.asarray(observed_domains), draws


def _target_contrast_draws(folds, left: str, right: str, resamples: int, seed: int):
    prepared = []
    observed_domains = []
    for fold in folds:
        left_rows = fold["conditions"][left]["seeds"]
        right_rows = fold["conditions"][right]["seeds"]
        truth, left_prediction, _, left_classes = _ensemble_seed_rows(left_rows)
        right_truth, right_prediction, _, right_classes = _ensemble_seed_rows(right_rows)
        if truth.tolist() != right_truth.tolist() or left_classes.tolist() != right_classes.tolist():
            raise ValueError("Paired control ensemble alignment mismatch")
        if any(
            row["metrics"]["recording_truth"] != left_rows[0]["metrics"]["recording_truth"]
            for row in right_rows
        ):
            raise ValueError("Paired controls do not share recording truth/order")
        value = macro_f1(truth, left_prediction) - macro_f1(truth, right_prediction)
        observed_domains.append(value)
        prepared.append((truth, left_prediction, right_prediction))
    rng = np.random.default_rng(seed)
    draws = np.zeros(resamples)
    worst_draws = np.zeros(resamples)
    for draw_index in range(resamples):
        differences = []
        left_domain = []
        right_domain = []
        for truth, left_prediction, right_prediction in prepared:
            indices = _stratified_draw(truth, rng)
            left_value = macro_f1(truth[indices], left_prediction[indices])
            right_value = macro_f1(truth[indices], right_prediction[indices])
            differences.append(left_value - right_value)
            left_domain.append(left_value)
            right_domain.append(right_value)
        draws[draw_index] = np.mean(differences)
        worst_draws[draw_index] = np.min(left_domain) - np.min(right_domain)
    return np.asarray(observed_domains), draws, worst_draws


def _centered_p(draws: np.ndarray, observed: float, null_boundary: float = 0.0) -> float:
    centered = draws - observed
    return float((1 + np.sum(centered >= observed - null_boundary)) / (len(draws) + 1))


def _recording_detection_summary(folds, condition: str, resamples: int, seed: int):
    prepared, observed_domains = [], []
    keys = ("auroc", "auprc", "far", "miss_rate")
    for fold in folds:
        rows = fold["conditions"][condition]["seeds"]
        truth, _, probability, classes = _ensemble_seed_rows(rows)
        background = np.flatnonzero(classes.astype(str) == "background")
        if len(background) != 1:
            raise RuntimeError("Detection requires one canonical background class")
        thresholds = [
            float(row["metrics"]["detection"]["threshold"])
            for row in rows
            if row["metrics"]["detection"].get("status") != "ineligible_no_canonical_background"
        ]
        if len(thresholds) != len(rows):
            raise RuntimeError("Detection threshold missing for a seed")
        threshold = float(np.mean(thresholds))
        y = (truth.astype(str) != "background").astype(int)
        scores = 1.0 - probability[:, int(background[0])]
        observed_domains.append(detection_metrics(y, scores, threshold))
        prepared.append((truth, y, scores, threshold))
    rng = np.random.default_rng(seed)
    draws = {key: np.zeros(resamples) for key in keys}
    for draw_index in range(resamples):
        domain_values = {key: [] for key in keys}
        for truth, y, scores, threshold in prepared:
            indices = _stratified_draw(truth, rng)
            for key in keys:
                domain_values[key].append(
                    detection_metrics(y[indices], scores[indices], threshold)[key]
                )
        for key in keys:
            draws[key][draw_index] = np.mean(domain_values[key])
    return {key: float(np.mean([item[key] for item in observed_domains])) for key in keys}, draws


def _per_class_summary(folds, label: str, resamples: int, seed: int):
    prepared, observed_domains = [], []
    for fold in folds:
        seed_records = fold["conditions"]["pcrd"]["seeds"]
        truth, prediction, _, _ = _ensemble_seed_rows(seed_records)
        observed_domains.append(
            {
                key: float(classification_metrics(truth, prediction)["per_class"][label][key])
                for key in ("precision", "recall", "f1")
            }
        )
        prepared.append((truth, prediction))
    rng = np.random.default_rng(seed)
    f1_draws = np.zeros(resamples)
    for draw_index in range(resamples):
        domain_values = []
        for truth, prediction in prepared:
            indices = _stratified_draw(truth, rng)
            domain_values.append(
                classification_metrics(truth[indices], prediction[indices])["per_class"][label]["f1"]
            )
        f1_draws[draw_index] = np.mean(domain_values)
    return {
        key: float(np.mean([item[key] for item in observed_domains]))
        for key in ("precision", "recall", "f1")
    }, f1_draws


def _robustness_bootstrap(folds, condition: str, axis: str, resamples: int, seed: int):
    grid = folds[0]["robustness"][condition][axis]["grid_db"]
    prepared, observed_curves = [], []
    for fold in folds:
        by_severity = fold["robustness"][condition][axis]["recording_metrics"]
        truth = np.asarray(by_severity[0][0]["recording_truth"], dtype=object)
        predictions = []
        for seed_metrics in by_severity:
            reference = seed_metrics[0]
            classes = np.asarray(reference["probability_class_order"], dtype=object)
            if any(
                metric["recording_truth"] != reference["recording_truth"]
                or metric["recording_order"] != reference["recording_order"]
                or metric["probability_class_order"] != reference["probability_class_order"]
                for metric in seed_metrics
            ):
                raise ValueError("Robustness seed ensemble alignment mismatch")
            probability = np.mean(
                [np.asarray(metric["recording_probabilities"], dtype=float) for metric in seed_metrics],
                axis=0,
            )
            predictions.append(classes[np.argmax(probability, axis=1)])
        observed_curves.append(
            [float(macro_f1(truth, prediction)) for prediction in predictions]
        )
        prepared.append((truth, predictions))
    rng = np.random.default_rng(seed)
    curve_draws = np.zeros((resamples, len(grid)))
    auc_draws = np.zeros(resamples)
    for draw_index in range(resamples):
        domain_curves = []
        for truth, predictions in prepared:
            indices = _stratified_draw(truth, rng)
            domain_curves.append(
                [float(macro_f1(truth[indices], prediction[indices])) for prediction in predictions]
            )
        curve_draws[draw_index] = np.mean(domain_curves, axis=0)
        auc_draws[draw_index] = normalized_curve_area(grid, curve_draws[draw_index])
    return np.mean(observed_curves, axis=0), curve_draws, auc_draws


def _source_metric_contrast(source_freezes, candidate: str, comparator: str, resamples: int, seed: int):
    prepared = []
    observed = []
    for freeze in source_freezes:
        candidate_metric = freeze["source_validation_fixed_h1_records"][candidate]["metrics"]
        comparator_metric = freeze["source_validation_fixed_h1_records"][comparator]["metrics"]
        if candidate_metric["recording_order"] != comparator_metric["recording_order"]:
            raise ValueError("H1 source comparator recording order mismatch")
        truth = np.asarray(candidate_metric["recording_truth"], dtype=object)
        recording_order = candidate_metric["recording_order"]
        domains = np.asarray(
            [
                freeze["source_validation_fixed_h1_records"][candidate]["recording_domains"][recording_id]
                for recording_id in recording_order
            ],
            dtype=object,
        )
        candidate_prediction = np.asarray(candidate_metric["recording_prediction"], dtype=object)
        comparator_prediction = np.asarray(comparator_metric["recording_prediction"], dtype=object)
        observed.append(
            macro_f1(truth, candidate_prediction) - macro_f1(truth, comparator_prediction)
        )
        prepared.append((truth, domains, candidate_prediction, comparator_prediction))
    rng = np.random.default_rng(seed)
    draws = np.zeros(resamples)
    for draw_index in range(resamples):
        values = []
        for truth, domains, candidate_prediction, comparator_prediction in prepared:
            indices = _source_domain_class_draw(truth, domains, rng)
            values.append(
                macro_f1(truth[indices], candidate_prediction[indices])
                - macro_f1(truth[indices], comparator_prediction[indices])
            )
        draws[draw_index] = np.mean(values)
    estimate = float(np.mean(observed))
    return {"estimate": estimate, "raw_p": _centered_p(draws, estimate), "draws": draws}


def _instability_map(record: dict[str, object]) -> dict[str, tuple[str, float]]:
    output: dict[str, tuple[str, float]] = {}
    for rows in record["instability"]["recording_values"].values():
        for row in rows:
            current = output.get(row["recording_id"])
            value = float(row["recording_max"])
            if current is None or value > current[1]:
                output[row["recording_id"]] = (row["domain"], value)
    return output


def _source_instability_contrast(source_freezes, candidate: str, comparator: str, resamples: int, seed: int):
    prepared = []
    observed = []
    for freeze in source_freezes:
        candidate_map = _instability_map(
            freeze["source_validation_fixed_h1_records"][candidate]
        )
        comparator_map = _instability_map(
            freeze["source_validation_fixed_h1_records"][comparator]
        )
        common = sorted(set(candidate_map) & set(comparator_map))
        by_domain = {}
        for recording_id in common:
            domain, candidate_value = candidate_map[recording_id]
            comparator_domain, comparator_value = comparator_map[recording_id]
            if domain != comparator_domain:
                raise ValueError("Instability comparator domain mismatch")
            by_domain.setdefault(domain, []).append(comparator_value - candidate_value)
        prepared.append({key: np.asarray(value) for key, value in by_domain.items()})
        observed.append(np.mean([np.mean(value) for value in by_domain.values()]))
    rng = np.random.default_rng(seed)
    draws = np.zeros(resamples)
    for index in range(resamples):
        fold_values = []
        for by_domain in prepared:
            domain_values = []
            for values in by_domain.values():
                sample = values[rng.integers(0, len(values), len(values))]
                domain_values.append(np.mean(sample))
            fold_values.append(np.mean(domain_values))
        draws[index] = np.mean(fold_values)
    estimate = float(np.mean(observed))
    return {
        "estimate": estimate,
        "raw_p": _centered_p(draws, estimate, null_boundary=-0.01),
        "draws": draws,
    }


def _h2_records(source_freezes, target_folds, order):
    output = []
    for freeze, target in zip(source_freezes, target_folds):
        first = freeze["h2_source_validation_records"][order[0]]
        source_order = first["metrics"]["recording_order"]
        source_truth = np.asarray(first["metrics"]["recording_truth"], dtype=object)
        source_prediction = np.column_stack(
            [
                np.asarray(
                    freeze["h2_source_validation_records"][condition]["metrics"]["recording_prediction"],
                    dtype=object,
                )
                for condition in order
            ]
        )
        source_instability = np.zeros((len(source_order), len(order)))
        for condition_index, condition in enumerate(order):
            values = _instability_map(
                freeze["h2_source_validation_records"][condition]
            )
            source_instability[:, condition_index] = [values[item][1] for item in source_order]
        source_domain = np.asarray(
            [first["recording_domains"][item] for item in source_order], dtype=object
        )
        held_first = target["h2_bank"][order[0]]["metrics"]
        held_truth = np.asarray(held_first["recording_truth"], dtype=object)
        held_prediction = np.column_stack(
            [
                np.asarray(target["h2_bank"][condition]["metrics"]["recording_prediction"], dtype=object)
                for condition in order
            ]
        )
        output.append(
            H2DomainRecords(
                source_truth,
                source_prediction,
                source_instability,
                source_domain,
                held_truth,
                held_prediction,
            )
        )
    return output


def run(
    *,
    campaign_dir: Path,
    output: Path,
    resamples: int,
    validate_output: bool = True,
) -> Path:
    if resamples != 10_000:
        raise ValueError(
            "Confirmatory finalization is frozen at exactly 10,000 bootstrap resamples"
        )
    campaign = json.loads((campaign_dir / "source_campaign.json").read_text(encoding="utf-8"))
    project_root = Path(__file__).resolve().parents[3]
    assert_frozen_environment(campaign, project_root)
    target_campaign = json.loads((campaign_dir / "target_campaign.json").read_text(encoding="utf-8"))
    if campaign.get("status") != "target_evaluated" or target_campaign.get("status") != "complete":
        raise PermissionError("Target campaign is incomplete")
    source_freezes = []
    target_folds = []
    model_freezes = []
    for index, fold in enumerate(campaign["folds"]):
        fold_dir = campaign_dir / fold["directory"]
        source_path = fold_dir / "source_freeze.json"
        model_path = fold_dir / "model_freeze.json"
        target_record = target_campaign["folds"][index]
        target_path = campaign_dir / target_record["path"]
        if sha256_file(source_path) != fold["source_freeze_sha256"]:
            raise RuntimeError("Source-freeze artifact changed before finalization")
        model_record = campaign["model_freezes"][index]
        if sha256_file(model_path) != model_record["sha256"]:
            raise RuntimeError("Model-freeze artifact changed before finalization")
        if sha256_file(target_path) != target_record["sha256"]:
            raise RuntimeError("Target-metric artifact changed before finalization")
        source_freezes.append(json.loads(source_path.read_text(encoding="utf-8")))
        target_folds.append(json.loads(target_path.read_text(encoding="utf-8")))
        model_freezes.append(json.loads(model_path.read_text(encoding="utf-8")))
    if any(freeze.get("status") != "complete" for freeze in model_freezes):
        raise RuntimeError("A fold shrank below a PCRD-comparable active set; M1 is untestable")
    preflight = json.loads((campaign_dir / "preflight.json").read_text(encoding="utf-8"))

    if preflight.get("campaign_mode") in {"pilot_two_dataset", "pilot_three_dataset"}:
        from scars.cli.finalize_pilot import build_pilot_results
        pilot_payload = build_pilot_results(campaign_dir, campaign, preflight,
                                           source_freezes, model_freezes, target_folds, resamples)
        atomic_json(output, pilot_payload)
        return output

    evidence_type = (
        "pilot_two_dataset"
        if preflight.get("campaign_mode") == "pilot_two_dataset"
        else "real_confirmatory"
    )
    payload = empty_results(evidence_type, campaign_dir.name)
    payload["static_metadata"] = build_static_metadata(project_root)
    payload["run"].update(
        {
            "status": "finalized",
            "device": target_folds[0]["conditions"]["pcrd"]["cost"]["latency"]["device"],
            "seeds": [11, 23, 37, 53, 71],
        }
    )
    payload["statistics"]["decision_engine_version"] = DECISION_ENGINE_VERSION
    payload["statistics"]["registered_resamples"] = resamples
    payload["splits"] = {
        "folds": [
            {
                "fold_id": source["fold_id"],
                **source["source_roles"],
                "held_target": source["held_target_recordings"],
            }
            for source in source_freezes
        ],
        "manifest_hash": sha256_file(campaign_dir / "splits.json"),
    }
    payload["datasets"]["domains"] = [item["fold_id"].split("=", 1)[-1] for item in source_freezes]
    payload["source_selection"]["folds"] = [
        {
            "fold_id": item["fold_id"],
            "selected_stable_id": item["selected_stable_id"],
            "pareto_front": item["pareto_front"],
            "source_artifact_hash": item["selected_artifact_sha256"],
        }
        for item in source_freezes
    ]
    payload["held_domain"]["folds"] = target_folds

    rows = []
    for condition in RECOGNITION_CONDITIONS:
        domains, draws = _target_condition_draws(
            target_folds, condition, resamples, BOOTSTRAP_SEED + len(rows)
        )
        low, high = np.quantile(draws, [0.025, 0.975])
        costs = [fold["conditions"][condition]["cost"] for fold in target_folds]
        row = {
            "condition_id": condition,
            "held_domain": "equal_domain",
            "recording_count": int(
                sum(fold["conditions"][condition]["seeds"][0]["metrics"]["recording_count"] for fold in target_folds)
            ),
            "mean_macro_f1": float(np.mean(domains)),
            "ci95_low": float(low),
            "ci95_high": float(high),
            "worst_domain_macro_f1": float(np.min(domains)),
            "balanced_accuracy": float(
                np.mean(
                    [
                        classification_metrics(
                            _ensemble_seed_rows(fold["conditions"][condition]["seeds"])[0],
                            _ensemble_seed_rows(fold["conditions"][condition]["seeds"])[1],
                        )["balanced_accuracy"]
                        for fold in target_folds
                    ]
                )
            ),
            "parameters": float(np.mean([cost["parameters"] for cost in costs if cost["parameters"] is not None])) if any(cost["parameters"] is not None for cost in costs) else None,
            "macs": float(np.mean([cost["macs"] for cost in costs if cost["macs"] is not None])) if any(cost["macs"] is not None for cost in costs) else None,
            "latency_ms": float(np.mean([cost["latency"]["median_ms"] for cost in costs])),
            "bytes_per_sample": float(np.mean([cost["bytes_per_sample"] for cost in costs])),
            "tensor_bytes": float(np.mean([cost["bytes_per_sample"] for cost in costs])),
            "representation_latency_ms": float(
                np.mean([cost["latency"].get("representation_median_ms", 0.0) for cost in costs])
            ),
            "model_latency_ms": float(
                np.mean([cost["latency"].get("model_median_ms", cost["latency"]["median_ms"]) for cost in costs])
            ),
            "peak_memory_bytes": (
                float(np.mean([cost["peak_inference_memory_bytes"] for cost in costs if cost.get("peak_inference_memory_bytes") is not None]))
                if any(cost.get("peak_inference_memory_bytes") is not None for cost in costs)
                else 0.0
            ),
        }
        rows.append(row)
    payload["tables"]["tab_primary_results"]["rows"] = rows
    payload["figures"]["fig_main_recognition"]["series"] = rows
    payload["costs"]["conditions"] = {row["condition_id"]: {key: row[key] for key in ("parameters", "macs", "latency_ms", "bytes_per_sample")} for row in rows}
    payload["statistics"]["seed_variability"] = {
        condition: {
            "per_domain_seed_macro_f1_std": [
                float(
                    np.std(
                        [
                            macro_f1(
                                np.asarray(seed["metrics"]["recording_truth"], dtype=object),
                                np.asarray(seed["metrics"]["recording_prediction"], dtype=object),
                            )
                            for seed in fold["conditions"][condition]["seeds"]
                        ],
                        ddof=1,
                    )
                )
                if len(fold["conditions"][condition]["seeds"]) > 1
                else 0.0
                for fold in target_folds
            ],
            "interpretation": "descriptive_seed_variability_not_the_primary_ensemble_estimand",
        }
        for condition in RECOGNITION_CONDITIONS
    }

    delta_gate_domains, delta_gate_draws, worst_draws = _target_contrast_draws(
        target_folds, "pcrd", "ordinary_gate", resamples, 24101
    )
    delta_shuffle_domains, delta_shuffle_draws, _ = _target_contrast_draws(
        target_folds, "pcrd", "shuffled_pcrd", resamples, 24102
    )
    delta_gate = float(np.mean(delta_gate_domains))
    delta_shuffle = float(np.mean(delta_shuffle_domains))
    gate_ci = np.quantile(delta_gate_draws, [0.025, 0.975])
    shuffle_ci = np.quantile(delta_shuffle_draws, [0.025, 0.975])
    worst_ci = np.quantile(worst_draws, [0.025, 0.975])
    pcrd_cost = next(row for row in rows if row["condition_id"] == "pcrd")
    early_cost = next(row for row in rows if row["condition_id"] == "early_fusion_resnet")
    ordinary_gate_cost = next(row for row in rows if row["condition_id"] == "ordinary_gate")
    worst_domain_difference = float(
        pcrd_cost["worst_domain_macro_f1"]
        - ordinary_gate_cost["worst_domain_macro_f1"]
    )
    has_mac_matched = all(
        "early_fusion_resnet_mac_matched" in fold["conditions"] for fold in target_folds
    )
    mac_reference_cost = (
        {
            "macs": float(
                np.mean(
                    [
                        fold["conditions"]["early_fusion_resnet_mac_matched"]["cost"].get(
                            "model_macs",
                            fold["conditions"]["early_fusion_resnet_mac_matched"]["cost"]["macs"],
                        )
                        for fold in target_folds
                    ]
                )
            )
        }
        if has_mac_matched
        else early_cost
    )
    coverage_by_nuisance = {
        nuisance: min(
            freeze["relation_coverage"]["by_nuisance"].get(nuisance, 0.0)
            for freeze in model_freezes
        )
        for nuisance in model_freezes[0]["relation_coverage"]["by_nuisance"]
    }
    integrity_failures = []
    for fold in target_folds:
        integrity = fold.get("integrity", {})
        if integrity.get("duplicates", {}).get("cross_domain_hashes"):
            integrity_failures.append(f"{fold['fold_id']}:exact_duplicate")
        if integrity.get("near_duplicates", {}).get("cross_domain_pair_count", 0):
            integrity_failures.append(f"{fold['fold_id']}:near_duplicate")
        if integrity.get("group_overlap", {}).get("status") != "passed":
            integrity_failures.append(f"{fold['fold_id']}:group_overlap")
        if integrity.get("temporal_group_adjacency", {}).get("status") != "passed":
            integrity_failures.append(f"{fold['fold_id']}:temporal_group_adjacency")
        if integrity.get("target_firewall", {}).get("target_reads") != 1:
            integrity_failures.append(f"{fold['fold_id']}:target_read_ledger")
    for source in source_freezes:
        for audit, value in source["negative_controls"].items():
            if audit == "background_only" and value.get("status") == "not_applicable":
                continue
            if value.get("status") != "passed":
                integrity_failures.append(f"{source['fold_id']}:{audit}")
    for freeze in model_freezes:
        if not freeze.get("relation_coverage", {}).get("complete_cells", False):
            integrity_failures.append(f"{freeze['fold_id']}:incomplete_relation_cells")
    integrity_pass = not integrity_failures
    compute_measurement_valid = all(
        fold["conditions"][condition]["cost"]["latency"].get("measurement_valid", False)
        for fold in target_folds
        for condition in ("pcrd", "early_fusion_resnet")
    )
    eligible = bool(
        preflight.get("status") == "ready"
        and len(preflight.get("datasets", [])) >= 3
        and len(source_freezes) >= 3
    )
    calibration_pass = all(
        np.isfinite(record.get("temperature", np.nan))
        and 0.05 <= float(record["temperature"]) <= 20.0
        and record.get("calibration_role") == "source_calibration"
        for freeze in model_freezes
        for record in freeze.get("teachers", {}).values()
    )
    m1 = m1_state_machine(
        eligible=eligible,
        integrity_pass=integrity_pass,
        calibration_pass=calibration_pass,
        relation_coverage_by_nuisance=coverage_by_nuisance,
        relation_recording_count=min(freeze["relation_coverage"]["recording_count"] for freeze in model_freezes),
        delta_gate=delta_gate,
        delta_gate_ci_low=float(gate_ci[0]),
        delta_shuffle=delta_shuffle,
        delta_shuffle_ci_low=float(shuffle_ci[0]),
        worst_domain_ci_low=float(worst_ci[0]),
        compute_measurement_valid=compute_measurement_valid,
        parameter_ratio=float(pcrd_cost["parameters"] / early_cost["parameters"]),
        mac_ratio=float(
            np.mean(
                [fold["conditions"]["pcrd"]["cost"].get("model_macs", fold["conditions"]["pcrd"]["cost"]["macs"]) for fold in target_folds]
            )
            / mac_reference_cost["macs"]
        ),
        latency_ratio=float(
            np.mean(
                [fold["conditions"]["pcrd"]["cost"]["latency"]["model_median_ms"] for fold in target_folds]
            )
            / np.mean(
                [fold["conditions"]["early_fusion_resnet"]["cost"]["latency"]["model_median_ms"] for fold in target_folds]
            )
        ),
    )
    payload["mechanism"]["m1"] = {
        "delta_gate": {"estimate": delta_gate, "ci95": gate_ci.tolist()},
        "delta_shuffle": {"estimate": delta_shuffle, "ci95": shuffle_ci.tolist()},
        "worst_domain_difference": {"estimate": worst_domain_difference, "ci95": worst_ci.tolist()},
        "state_machine_inputs": {
            "eligible": eligible,
            "integrity_pass": integrity_pass,
            "calibration_pass": calibration_pass,
            "relation_coverage_by_nuisance": coverage_by_nuisance,
            "relation_recording_count": min(
                freeze["relation_coverage"]["recording_count"] for freeze in model_freezes
            ),
            "compute_measurement_valid": compute_measurement_valid,
        },
        "compute_gate": {
            "parameter_ratio": pcrd_cost["parameters"] / early_cost["parameters"],
            "mac_ratio": np.mean(
                [fold["conditions"]["pcrd"]["cost"].get("model_macs", fold["conditions"]["pcrd"]["cost"]["macs"]) for fold in target_folds]
            ) / mac_reference_cost["macs"],
            "mac_scope": "model_only_for_registered_M1_ratio",
            "latency_ratio": np.mean(
                [fold["conditions"]["pcrd"]["cost"]["latency"]["model_median_ms"] for fold in target_folds]
            ) / np.mean(
                [fold["conditions"]["early_fusion_resnet"]["cost"]["latency"]["model_median_ms"] for fold in target_folds]
            ),
            "latency_scope": "batch1_model_only_for_registered_M1_ratio",
            "parameter_reference": "early_fusion_resnet",
            "mac_reference": "early_fusion_resnet_mac_matched" if has_mac_matched else "early_fusion_resnet",
            "measurement_valid": compute_measurement_valid,
        },
        **m1,
    }

    h1_candidates = []
    for candidate in (
        "W+C",
        "norm_percentile",
        "norm_zscore",
        "norm_none",
        "resolution_8",
        "resolution_16",
        "resolution_32",
    ):
        instability = {
            comparator: _source_instability_contrast(
                source_freezes, candidate, comparator, resamples, 24200 + len(h1_candidates)
            )
            for comparator in ("W", "STFT")
        }
        f1 = {
            comparator: _source_metric_contrast(
                source_freezes, candidate, comparator, resamples, 24300 + len(h1_candidates)
            )
            for comparator in ("W", "STFT")
        }
        feasible = all(freeze["candidates"][candidate]["feasible"] for freeze in source_freezes)
        measurement_valid = all(
            freeze["candidates"][candidate]["cost"].get("measurement_valid", False)
            for freeze in source_freezes
        )
        h1_candidates.append(
            {
                "candidate_id": candidate,
                "p_instability_vs_w": instability["W"]["raw_p"],
                "p_instability_vs_stft": instability["STFT"]["raw_p"],
                "p_f1_vs_w": f1["W"]["raw_p"],
                "p_f1_vs_stft": f1["STFT"]["raw_p"],
                "feasible": feasible,
                "measurement_valid": measurement_valid,
                "instability_effect_vs_w": instability["W"]["estimate"],
                "instability_effect_vs_stft": instability["STFT"]["estimate"],
                "f1_effect_vs_w": f1["W"]["estimate"],
                "f1_effect_vs_stft": f1["STFT"]["estimate"],
            }
        )
    h1 = h1_raw_p(h1_candidates)

    order = source_freezes[0]["h2_configuration_order"]
    log_macs = np.log(
        [np.mean([freeze["h2_candidates"][condition]["cost"]["estimated_macs"] for freeze in source_freezes]) for condition in order]
    )
    resolutions = np.asarray(
        [
            json.loads(
                (campaign_dir / campaign["folds"][0]["directory"] / source_freezes[0]["h2_candidates"][condition]["path"]).read_text(encoding="utf-8")
            )["config"]["output_bins"]
            for condition in order
        ],
        dtype=float,
    )
    h2 = h2_hierarchical_bootstrap(
        _h2_records(source_freezes, target_folds, order),
        log_macs,
        resolutions,
        seed=BOOTSTRAP_SEED,
        resamples=resamples,
    )
    h2["configuration_order"] = list(order)

    h3_components = []
    ablation_rows = []
    representation_ablation_ids = [
        condition
        for condition in ABLATION_CONDITIONS
        if not condition.startswith("active_minus_")
    ]
    for condition_index, condition in enumerate(representation_ablation_ids):
        virtual_folds = [
            {
                **fold,
                "conditions": {
                    **fold["conditions"],
                    condition: {
                        "seeds": [
                            {"metrics": fold["ablation_bank"][condition]["metrics"]}
                        ]
                    },
                },
            }
            for fold in target_folds
        ]
        ablation_domains, ablation_draws = _target_condition_draws(
            virtual_folds, condition, resamples, 24600 + condition_index
        )
        contrast_domains, contrast_draws, _ = _target_contrast_draws(
            virtual_folds, "pcrd", condition, resamples, 24700 + condition_index
        )
        contrast_estimate = float(np.mean(contrast_domains))
        mean_low, mean_high = np.quantile(ablation_draws, [0.025, 0.975])
        delta_low, delta_high = np.quantile(-contrast_draws, [0.025, 0.975])
        similarities = [
            fold["ablation_bank"][condition]["representation_similarity"]
            for fold in target_folds
        ]
        candidate_ids = [
            fold["ablation_bank"][condition]["candidate_id"] for fold in target_folds
        ]
        costs = [
            source["candidates"][candidate]["cost"]
            for source, candidate in zip(source_freezes, candidate_ids)
        ]
        ablation_rows.append(
            {
                "condition_id": condition,
                "held_domain": "equal_domain",
                "recording_count": sum(
                    fold["ablation_bank"][condition]["metrics"]["recording_count"]
                    for fold in target_folds
                ),
                "mean_macro_f1": float(np.mean(ablation_domains)),
                "ci95_low": float(mean_low),
                "ci95_high": float(mean_high),
                "delta_vs_pcrd": -contrast_estimate,
                "delta_ci95_low": float(delta_low),
                "delta_ci95_high": float(delta_high),
                "cosine": float(np.mean([item["cosine"] for item in similarities])),
                "nmae": float(np.mean([item["normalized_mae"] for item in similarities])),
                "domain_probe_accuracy": float(
                    np.mean(
                        [
                            fold["ablation_bank"][condition]["domain_probe_accuracy"]
                            for fold in target_folds
                        ]
                    )
                ),
                "parameters": 0,
                "macs": float(np.mean([item["estimated_macs"] for item in costs])),
                "latency_ms": float(np.mean([item["batch1_latency_ms"] for item in costs])),
                "interpretation": "descriptive_pending_joint_decision",
            }
        )
    for family in model_freezes[0]["active_families"]:
        condition = f"active_minus_{family}"
        domains, draws, _ = _target_contrast_draws(
            target_folds, "pcrd", condition, resamples, 24400 + len(h3_components)
        )
        estimate = float(np.mean(domains))
        similarity = [fold["conditions"][condition]["representation_similarity"] for fold in target_folds]
        cosine = float(np.mean([item["cosine"] for item in similarity]))
        nmae = float(np.mean([item["normalized_mae"] for item in similarity]))
        raw_p = _centered_p(draws, estimate)
        h3_components.append(
            {"family": family, "estimate": estimate, "raw_p": raw_p, "cosine": cosine, "nmae": nmae}
        )
        low, high = np.quantile(draws, [0.025, 0.975])
        active_domains, active_draws = _target_condition_draws(
            target_folds, condition, resamples, 24800 + len(h3_components)
        )
        active_low, active_high = np.quantile(active_draws, [0.025, 0.975])
        ablation_rows.append(
            {
                "condition_id": condition,
                "held_domain": "equal_domain",
                "recording_count": sum(fold["conditions"][condition]["seeds"][0]["metrics"]["recording_count"] for fold in target_folds),
                "mean_macro_f1": float(np.mean(active_domains)),
                "ci95_low": float(active_low),
                "ci95_high": float(active_high),
                "delta_vs_pcrd": -estimate,
                "delta_ci95_low": float(-high),
                "delta_ci95_high": float(-low),
                "cosine": cosine,
                "nmae": nmae,
                "domain_probe_accuracy": float(
                    np.mean(
                        [
                            source["candidates"][source["selected_stable_id"]]["domain_probe_accuracy"]
                            for source in source_freezes
                        ]
                    )
                ),
                "parameters": target_folds[0]["conditions"][condition]["cost"]["parameters"],
                "macs": target_folds[0]["conditions"][condition]["cost"]["macs"],
                "latency_ms": np.mean([fold["conditions"][condition]["cost"]["latency"]["median_ms"] for fold in target_folds]),
                "interpretation": "pending_Holm",
            }
        )
    h3 = h3_raw_p(h3_components)
    row_lookup = {row["condition_id"]: row for row in ablation_rows}
    selected_domain_probe = float(
        np.mean(
            [
                source["candidates"][source["selected_stable_id"]]["domain_probe_accuracy"]
                for source in source_freezes
            ]
        )
    )
    for family in ("W", "C", "E", "S"):
        condition = f"active_minus_{family}"
        if condition not in row_lookup:
            row_lookup[condition] = {
                "condition_id": condition,
                "held_domain": "equal_domain",
                "recording_count": pcrd_cost["recording_count"],
                "mean_macro_f1": pcrd_cost["mean_macro_f1"],
                "ci95_low": pcrd_cost["ci95_low"],
                "ci95_high": pcrd_cost["ci95_high"],
                "delta_vs_pcrd": 0.0,
                "delta_ci95_low": 0.0,
                "delta_ci95_high": 0.0,
                "cosine": 1.0,
                "nmae": 0.0,
                "domain_probe_accuracy": selected_domain_probe,
                "parameters": pcrd_cost["parameters"],
                "macs": pcrd_cost["macs"],
                "latency_ms": pcrd_cost["latency_ms"],
                "interpretation": "structural_identity_family_absent_from_frozen_active_set",
            }
    payload["tables"]["tab_ablations"]["rows"] = [
        row_lookup[condition] for condition in ABLATION_CONDITIONS
    ]

    perf_domains, perf_draws, _ = _target_contrast_draws(
        [
            {**fold, "conditions": {**fold["conditions"], "resolution_8": {"seeds": [{"metrics": fold["ablation_bank"]["resolution_8"]["metrics"]}]}, "resolution_32": {"seeds": [{"metrics": fold["ablation_bank"]["resolution_32"]["metrics"]}]}}}
            for fold in target_folds
        ],
        "resolution_8",
        "resolution_32",
        resamples,
        24500,
    )
    perf_observed = float(np.mean(perf_domains))
    p_perf = _centered_p(perf_draws, perf_observed, null_boundary=-0.01)
    latency_differences = np.asarray(
        [
            fold["candidates"]["resolution_32"]["cost"]["batch1_latency_ms"]
            - fold["candidates"]["resolution_8"]["cost"]["batch1_latency_ms"]
            for fold in source_freezes
        ]
    )
    rng = np.random.default_rng(24501)
    latency_draws = np.asarray(
        [np.mean(latency_differences[rng.integers(0, len(latency_differences), len(latency_differences))]) for _ in range(resamples)]
    )
    latency_observed = float(np.mean(latency_differences))
    p_latency = _centered_p(latency_draws, latency_observed)
    h4 = h4_raw_p(
        p_performance_noninferiority=p_perf,
        p_latency_superiority=p_latency,
        bytes_strictly_reduced=all(
            fold["candidates"]["resolution_8"]["cost"]["bytes_per_sample"]
            < fold["candidates"]["resolution_32"]["cost"]["bytes_per_sample"]
            for fold in source_freezes
        ),
        macs_strictly_reduced=all(
            fold["candidates"]["resolution_8"]["cost"]["estimated_macs"]
            < fold["candidates"]["resolution_32"]["cost"]["estimated_macs"]
            for fold in source_freezes
        ),
        measurement_valid=all(
            fold["candidates"][condition]["cost"].get("measurement_valid")
            for fold in source_freezes
            for condition in ("resolution_8", "resolution_32")
        ),
    )
    h4["inputs"] = {
        "p_performance_noninferiority": p_perf,
        "p_latency_superiority": p_latency,
        "bytes_strictly_reduced": all(
            fold["candidates"]["resolution_8"]["cost"]["bytes_per_sample"]
            < fold["candidates"]["resolution_32"]["cost"]["bytes_per_sample"]
            for fold in source_freezes
        ),
        "macs_strictly_reduced": all(
            fold["candidates"]["resolution_8"]["cost"]["estimated_macs"]
            < fold["candidates"]["resolution_32"]["cost"]["estimated_macs"]
            for fold in source_freezes
        ),
        "measurement_valid": all(
            fold["candidates"][condition]["cost"].get("measurement_valid")
            for fold in source_freezes
            for condition in ("resolution_8", "resolution_32")
        ),
    }
    raw = {"H1": h1, "H2": h2, "H3": h3, "H4": h4}
    decisions = finalize_hypotheses(raw)
    payload["statistics"].update({"h1": h1, "h2": h2, "h3": h3, "h4": h4, "holm": decisions["holm"]})
    h1_effect = max(
        min(
            candidate["instability_effect_vs_w"],
            candidate["instability_effect_vs_stft"],
            candidate["f1_effect_vs_w"],
            candidate["f1_effect_vs_stft"],
        )
        for candidate in h1_candidates
        if candidate["feasible"]
    )
    estimates = {
        "H1": float(h1_effect),
        "H2": h2.get("estimate"),
        "H3": float(min(component["estimate"] for component in h3_components)),
        "H4": perf_observed,
    }
    hypothesis_records = {}
    for hypothesis in ("H1", "H2", "H3", "H4"):
        decision = decisions["hypotheses"][hypothesis]
        hypothesis_records[hypothesis] = {
            "estimate": estimates[hypothesis],
            "raw_p": raw[hypothesis].get("raw_p"),
            "holm_p": None if decisions["holm"] is None else decisions["holm"][hypothesis],
            "status": decision["status"],
            "decision": decision["status"],
            "reason": decision.get("reason", "registered_rule_and_joint_Holm"),
        }
    payload["hypotheses"] = hypothesis_records
    payload["tables"]["tab_hypotheses"].update(
        {
            "row_fields": ["id", "estimand", "raw_p", "holm_p", "status", "decision_reason"],
            "rows": [
                {
                    "id": key,
                    "estimand": value["estimate"],
                    "raw_p": value["raw_p"],
                    "holm_p": value["holm_p"],
                    "status": value["status"],
                    "decision_reason": value["reason"],
                }
                for key, value in hypothesis_records.items()
            ],
        }
    )
    payload["tables"]["tab_m1"].update(
        {
            "row_fields": ["component", "estimate", "interval_or_threshold", "status"],
            "rows": [
                {"component": "PCRD-minus-gate", "estimate": delta_gate, "interval_or_threshold": gate_ci.tolist(), "status": m1["status"]},
                {"component": "PCRD-minus-shuffled", "estimate": delta_shuffle, "interval_or_threshold": shuffle_ci.tolist(), "status": m1["status"]},
                {"component": "worst-domain-noninferiority", "estimate": worst_domain_difference, "interval_or_threshold": worst_ci.tolist(), "status": m1["status"]},
            ],
        }
    )
    payload["mechanism"]["relation_coverage"] = {
        "aggregate": min(coverage_by_nuisance.values()),
        "by_nuisance": coverage_by_nuisance,
        "recording_count": min(freeze["relation_coverage"]["recording_count"] for freeze in model_freezes),
    }

    detection_rows = []
    detection_eligible = bool(preflight.get("detection_eligible", False))
    if detection_eligible:
        for condition_index, condition in enumerate(RECOGNITION_CONDITIONS):
            values, detection_draws = _recording_detection_summary(
                target_folds, condition, resamples, 25000 + condition_index
            )
            thresholds = [
                float(seed["metrics"]["detection"]["threshold"])
                for fold in target_folds
                for seed in fold["conditions"][condition]["seeds"]
            ]
            auroc_ci = np.quantile(detection_draws["auroc"], [0.025, 0.975]).tolist()
            detection_rows.append(
                {
                    "condition_id": condition,
                    **values,
                    "ci95": auroc_ci,
                    "recording_count": sum(
                        fold["conditions"][condition]["seeds"][0]["metrics"]["recording_count"]
                        for fold in target_folds
                    ),
                    "threshold": float(np.mean(thresholds)),
                    "interpretation": "descriptive_source_threshold_recording_block_interval",
                }
            )
        pcrd_detection = next(row for row in detection_rows if row["condition_id"] == "pcrd")
        payload["detection"] = {
            "eligible": True,
            "folds": [fold["conditions"]["pcrd"] for fold in target_folds],
            "summary": {key: pcrd_detection[key] for key in ("auroc", "auprc", "far", "miss_rate")},
        }
    else:
        payload["detection"] = {
            "eligible": False,
            "status": "ineligible_no_shared_canonical_background",
            "folds": [],
            "summary": None,
        }
    payload["tables"]["tab_detection"].update(
        {
            "row_fields": ["condition_id", "auroc", "auprc", "far", "miss_rate", "ci95", "recording_count", "threshold", "interpretation"],
            "rows": detection_rows,
        }
    )

    robustness_rows = []
    robustness_series = []
    for condition_index, condition in enumerate(("pcrd", "W", "C", "STFT")):
        summaries = {}
        for axis_index, axis in enumerate(("snr", "sir")):
            summaries[axis] = _robustness_bootstrap(
                target_folds,
                condition,
                axis,
                resamples,
                25100 + 10 * condition_index + axis_index,
            )
        snr_curve, snr_curve_draws, snr_auc_draws = summaries["snr"]
        sir_curve, sir_curve_draws, sir_auc_draws = summaries["sir"]
        all_values = np.concatenate([snr_curve, sir_curve]).tolist()
        clean = next(row["mean_macro_f1"] for row in rows if row["condition_id"] == "pcrd") if condition == "pcrd" else float(np.mean([fold["candidate_bank"][condition]["metrics"]["macro_f1"] for fold in target_folds]))
        worst = min(all_values)
        combined = np.concatenate([snr_auc_draws, sir_auc_draws])
        ci = np.quantile(combined, [0.025, 0.975])
        robustness_rows.append(
            {
                "condition_id": condition,
                "snr_auc": float(normalized_curve_area(target_folds[0]["robustness"][condition]["snr"]["grid_db"], snr_curve)),
                "sir_auc": float(normalized_curve_area(target_folds[0]["robustness"][condition]["sir"]["grid_db"], sir_curve)),
                "worst_nuisance": "SNR_or_SIR_registered_grid",
                "worst_degradation": float(clean - worst),
                "recording_count": sum(fold["candidate_bank"]["W"]["metrics"]["recording_count"] for fold in target_folds),
                "ci95_low": float(ci[0]),
                "ci95_high": float(ci[1]),
                "interpretation": "descriptive",
            }
        )
        for axis, (curve, curve_draws, _) in summaries.items():
            grid = target_folds[0]["robustness"][condition][axis]["grid_db"]
            curve_ci_low = np.quantile(curve_draws, 0.025, axis=0)
            curve_ci_high = np.quantile(curve_draws, 0.975, axis=0)
            robustness_series.append(
                {
                    "series_id": f"{condition}_{axis}",
                    "panel_id": axis,
                    "x_name": f"{axis}_db",
                    "x_unit": "dB",
                    "x": grid,
                    "y_name": "recording_macro_f1",
                    "y_unit": "fraction",
                    "y": np.asarray(curve).tolist(),
                    "ci95_low": curve_ci_low.tolist(),
                    "ci95_high": curve_ci_high.tolist(),
                    "recording_support": [robustness_rows[-1]["recording_count"]] * len(grid),
                    "held_domain": "equal_domain",
                }
            )
    payload["tables"]["tab_robustness"].update(
        {
            "row_fields": ["condition_id", "snr_auc", "sir_auc", "worst_nuisance", "worst_degradation", "recording_count", "ci95_low", "ci95_high", "interpretation"],
            "rows": robustness_rows,
        }
    )
    payload["robustness"] = {
        "snr": [item for item in robustness_series if item["panel_id"] == "snr"],
        "sir": [item for item in robustness_series if item["panel_id"] == "sir"],
        "normalized_curve_auc": {item["condition_id"]: {"snr": item["snr_auc"], "sir": item["sir_auc"]} for item in robustness_rows},
    }

    per_class_rows = []
    class_labels = sorted(
        target_folds[0]["conditions"]["pcrd"]["seeds"][0]["metrics"]["per_class"]
    )
    for label_index, label in enumerate(class_labels):
        metrics, f1_draws = _per_class_summary(
            target_folds, label, resamples, 25200 + label_index
        )
        ci = np.quantile(f1_draws, [0.025, 0.975])
        per_class_rows.append(
            {
                "class_domain": label,
                **metrics,
                "recording_support": int(
                    sum(
                        fold["conditions"]["pcrd"]["seeds"][0]["metrics"]["per_class"][label]["support"]
                        for fold in target_folds
                    )
                ),
                "ci95_low": float(ci[0]),
                "ci95_high": float(ci[1]),
            }
        )
    payload["tables"]["tab_per_class"].update(
        {"row_fields": ["class_domain", "precision", "recall", "f1", "recording_support", "ci95_low", "ci95_high"], "rows": per_class_rows}
    )

    gate_weights = np.asarray(
        [
            seed["mean_gate_weights"]
            for fold in target_folds
            for seed in fold["conditions"]["pcrd"]["seeds"]
        ]
    )
    entropy = -np.sum(gate_weights * np.log(np.maximum(gate_weights, 1.0e-12)), axis=1)
    routing_rows = [
        {"diagnostic": "mean_gate_entropy", "estimate": float(np.mean(entropy)), "interval_or_threshold": np.quantile(entropy, [0.025, 0.975]).tolist(), "recording_support": int(sum(fold["conditions"]["pcrd"]["seeds"][0]["metrics"]["recording_count"] for fold in target_folds)), "interpretation": "descriptive_not_channel_attribution"},
        {"diagnostic": "relation_coverage", "estimate": min(coverage_by_nuisance.values()), "interval_or_threshold": 0.5, "recording_support": min(freeze["relation_coverage"]["recording_count"] for freeze in model_freezes), "interpretation": m1["status"]},
    ]
    payload["tables"]["tab_routing_calibration"].update(
        {"row_fields": ["diagnostic", "estimate", "interval_or_threshold", "recording_support", "interpretation"], "rows": routing_rows}
    )

    recording_manifest = json.loads((campaign_dir / "recordings_recognition.json").read_text(encoding="utf-8"))
    dataset_rows = []
    for dataset in preflight["datasets"]:
        records = [item for item in recording_manifest["recordings"] if item["dataset"] == dataset]
        dataset_rows.append(
            {"dataset_id": dataset, "recording_count": len(records), "group_count": len({item["split_group"] for item in records}), "class_count": len({item["label"] for item in records}), "has_background": any(item["is_background"] for item in records), "split_axes": "dataset_x_class_physical_group", "eligibility": "eligible", "license": "verified_by_label_contract", "checksum_manifest": preflight["content_hashes_complete"]}
        )
    payload["eligibility"] = {
        "eligible_domains": len(dataset_rows),
        "datasets": dataset_rows,
        "recognition_eligible": True,
        "detection_eligible": detection_eligible,
    }
    payload["tables"]["tab_datasets"].update({"rows": dataset_rows})
    payload["tables"]["tab_channel_contract"]["rows"] = [
        {"family": family, "dtype": "float32", "shape": "H_x_W", "model_input": True}
        for family in ("W", "C", "E", "S")
    ]
    payload["tables"]["tab_baselines"]["rows"] = [
        {"condition_id": condition, "source_only": True, "target_tuning": False}
        for condition in RECOGNITION_CONDITIONS
    ]
    if has_mac_matched:
        payload["tables"]["tab_baselines"]["rows"].append(
            {
                "condition_id": "early_fusion_resnet_mac_matched",
                "source_only": True,
                "target_tuning": False,
                "reporting_role": "capacity_sensitivity_not_primary_registry",
            }
        )
    payload["tables"]["tab_external_sota"]["rows"] = [
        {
            "condition_id": item["condition_id"],
            "eligibility": item["implementation_status"],
            "ineligibility_reason": item["reason"],
            "target_privilege": item["target_privilege"],
            "native_input": None,
            "observation_samples": None,
            "recording_count": None,
            "mean_macro_f1": None,
            "ci95_low": None,
            "ci95_high": None,
            "worst_domain_macro_f1": None,
            "parameters": None,
            "macs": None,
            "latency_ms": None,
        }
        for item in frozen_external_sota_registry()
    ]

    integrity_rows = []
    for fold in target_folds:
        for audit, value in fold["integrity"].items():
            status = "passed"
            if audit == "duplicates" and value.get("cross_domain_hashes"):
                status = "failed"
            if audit == "near_duplicates" and value.get("cross_domain_pair_count"):
                status = "failed"
            if audit == "group_overlap":
                status = value["status"]
            if value.get("status") == "failed":
                status = "failed"
            integrity_rows.append(
                {"audit": f"{fold['fold_id']}:{audit}", "status": status, "critical": True, "evidence_path_hash": hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()}
            )
    for source in source_freezes:
        for audit, value in source["negative_controls"].items():
            status = value["status"]
            integrity_rows.append(
                {
                    "audit": f"{source['fold_id']}:{audit}",
                    "status": status,
                    "critical": True,
                    "evidence_path_hash": hashlib.sha256(
                        json.dumps(value, sort_keys=True).encode()
                    ).hexdigest(),
                }
            )
    integrity_pass = not integrity_failures
    payload["integrity"] = {"status": "passed" if integrity_pass else "failed", "critical_failures": integrity_failures, "audits": integrity_rows}
    payload["tables"]["tab_integrity"].update(
        {"row_fields": ["audit", "status", "critical", "evidence_path_hash"], "rows": integrity_rows}
    )

    condition_names = [row["condition_id"] for row in rows]
    payload["figures"]["fig_main_recognition"] = {
        "series": [{"series_id": "equal_domain", "panel_id": "recognition", "x_name": "condition", "x_unit": "category", "x": condition_names, "y_name": "recording_macro_f1", "y_unit": "fraction", "y": [row["mean_macro_f1"] for row in rows], "ci95_low": [row["ci95_low"] for row in rows], "ci95_high": [row["ci95_high"] for row in rows], "recording_support": [row["recording_count"] for row in rows], "held_domain": "equal_domain"}]
    }
    payload["figures"]["fig_m1_effects"] = {
        "series": [{"series_id": "m1", "panel_id": "effects", "x_name": "contrast", "x_unit": "macro_f1", "x": ["gate", "shuffle"], "y_name": "effect", "y_unit": "fraction", "y": [delta_gate, delta_shuffle], "ci95_low": [float(gate_ci[0]), float(shuffle_ci[0])], "ci95_high": [float(gate_ci[1]), float(shuffle_ci[1])], "recording_support": [pcrd_cost["recording_count"]] * 2, "held_domain": "equal_domain"}]
    }
    payload["figures"]["fig_domain_heatmap"] = {
        "matrix": [
            [
                float(
                    macro_f1(
                        _ensemble_seed_rows(fold["conditions"][condition]["seeds"])[0],
                        _ensemble_seed_rows(fold["conditions"][condition]["seeds"])[1],
                    )
                )
                for fold in target_folds
            ]
            for condition in RECOGNITION_CONDITIONS
        ],
        "x_labels": payload["datasets"]["domains"],
        "y_labels": RECOGNITION_CONDITIONS,
    }
    payload["figures"]["fig_robustness"] = {"series": robustness_series}
    payload["figures"]["fig_routing_diagnostics"] = {
        "panel_fields": ["panel_id", "nuisance", "severity", "family", "weights", "recording_ids", "entropy", "relation_coverage"],
        "panels": [{"panel_id": "aggregate", "nuisance": "all", "severity": "all", "family": model_freezes[0]["active_families"], "weights": np.mean(gate_weights, axis=0).tolist(), "recording_ids": [item for fold in target_folds for item in fold["conditions"]["pcrd"]["seeds"][0]["metrics"]["recording_order"]], "entropy": float(np.mean(entropy)), "relation_coverage": min(coverage_by_nuisance.values())}],
    }
    first_source = source_freezes[0]
    if h2.get("status") == "ok":
        h2_residual_instability = np.mean(
            [item["residual_instability"] for item in h2["per_domain"]], axis=0
        ).tolist()
        h2_residual_degradation = np.mean(
            [item["residual_degradation"] for item in h2["per_domain"]], axis=0
        ).tolist()
    else:
        h2_residual_instability = None
        h2_residual_degradation = None
    selection_order = first_source["h2_configuration_order"]
    h2_source_records = first_source["h2_source_validation_records"]
    h2_artifacts = first_source["h2_candidates"]
    payload["figures"]["fig_scars_selection_h2"] = {
        "panel_fields": ["panel_id", "configuration_id", "source_instability", "source_macro_f1", "cost", "feasible", "pareto", "selected", "h2_residual_instability", "h2_residual_degradation", "held_domain"],
        "panels": [{"panel_id": "h2_diagnostic_bank", "configuration_id": selection_order, "source_instability": [h2_source_records[item]["instability"]["value"] for item in selection_order], "source_macro_f1": [h2_source_records[item]["metrics"]["macro_f1"] for item in selection_order], "cost": [h2_artifacts[item]["cost"]["estimated_macs"] for item in selection_order], "feasible": [h2_artifacts[item]["feasible"] for item in selection_order], "pareto": [False for _ in selection_order], "selected": [False for _ in selection_order], "h2_residual_instability": h2_residual_instability, "h2_residual_degradation": h2_residual_degradation, "held_domain": first_source["fold_id"]}],
    }
    confusion_panels = []
    for fold in target_folds:
        truth, prediction, _, classes = _ensemble_seed_rows(
            fold["conditions"]["pcrd"]["seeds"]
        )
        metrics = classification_metrics(truth, prediction)
        confusion_panels.append(
            {
                "panel_id": fold["fold_id"],
                "condition_id": "pcrd",
                "held_domain": fold["fold_id"],
                "class_labels": metrics["label_order"],
                "matrix": metrics["normalized_confusion_matrix"],
                "recording_support": len(truth),
            }
        )
    payload["figures"]["fig_confusion_matrices"] = {
        "panel_fields": ["panel_id", "condition_id", "held_domain", "class_labels", "matrix", "recording_support"],
        "panels": confusion_panels,
    }
    calibration_panels = []
    for fold in target_folds:
        truth, _, probability, classes = _ensemble_seed_rows(
            fold["conditions"]["pcrd"]["seeds"]
        )
        calibration = calibration_metrics(truth, probability, classes)
        interval = []
        for item in calibration["bins"]:
            proportion = float(item["empirical_accuracy"])
            count = int(item["support"])
            denominator = 1.0 + 1.96**2 / count
            center = (proportion + 1.96**2 / (2 * count)) / denominator
            half = 1.96 * np.sqrt(proportion * (1.0 - proportion) / count + 1.96**2 / (4 * count**2)) / denominator
            interval.append((max(0.0, center - half), min(1.0, center + half)))
        calibration_panels.append({"panel_id": fold["fold_id"], "condition_id": "pcrd", "held_domain": fold["fold_id"], "confidence": [item["confidence"] for item in calibration["bins"]], "empirical_accuracy": [item["empirical_accuracy"] for item in calibration["bins"]], "ci95_low": [item[0] for item in interval], "ci95_high": [item[1] for item in interval], "bin_support": [item["support"] for item in calibration["bins"]], "ece": calibration["ece"], "brier": calibration["brier"], "nll": calibration["nll"]})
    payload["figures"]["fig_calibration"] = {"panel_fields": ["panel_id", "condition_id", "held_domain", "confidence", "empirical_accuracy", "ci95_low", "ci95_high", "bin_support", "ece", "brier", "nll"], "panels": calibration_panels}
    payload["figures"]["fig_compute_tradeoff"] = {
        "x_names": ["latency_ms", "macs", "bytes"],
        "series": [{"series_id": key, "panel_id": key, "x_name": key, "x_unit": "measured_or_counted", "x": [row["bytes_per_sample"] if key == "bytes" else row[key] for row in rows], "y_name": "recording_macro_f1", "y_unit": "fraction", "y": [row["mean_macro_f1"] for row in rows], "recording_support": [row["recording_count"] for row in rows], "held_domain": "equal_domain"} for key in ("latency_ms", "macs", "bytes")],
    }
    payload["provenance"] = {
        "dataset_checksums": {item["recording_id"]: item["sha256"] for item in recording_manifest["recordings"]},
        "source_role_manifests": payload["splits"],
        "transform_hash": [item["canonical_tensor_artifact"]["sha256"] for item in source_freezes],
        "global_pareto_candidate_hash": [item["selected_artifact_sha256"] for item in source_freezes],
        "teacher_hashes": [{family: record["sha256"] for family, record in freeze["teachers"].items()} for freeze in model_freezes],
        "calibration_hash": hashlib.sha256(json.dumps([{family: record["temperature"] for family, record in freeze["teachers"].items()} for freeze in model_freezes], sort_keys=True).encode()).hexdigest(),
        "relation_cache_hash": [freeze["relation_cache"]["sha256"] for freeze in model_freezes],
        "student_hash": [[record["sha256"] for record in freeze["models"]["pcrd"]] for freeze in model_freezes],
        "environment": campaign["provenance"],
        "hardware_power_manifest": [
            {
                "fold_id": fold["fold_id"],
                "pcrd": fold["conditions"]["pcrd"]["cost"]["latency"],
                "early_fusion_resnet": fold["conditions"]["early_fusion_resnet"]["cost"]["latency"],
            }
            for fold in target_folds
        ],
        "target_access_ledger": [fold["integrity"]["target_firewall"] for fold in target_folds],
        "artifact_manifest": [
            {
                "role": "recordings_manifest",
                "path": "recordings_recognition.json",
                "sha256": sha256_file(campaign_dir / "recordings_recognition.json"),
            },
            {
                "role": "split_manifest",
                "path": "splits.json",
                "sha256": sha256_file(campaign_dir / "splits.json"),
            },
            *[
                {
                    "role": "source_freeze",
                    "fold_id": fold["fold_id"],
                    "path": str(Path(record["directory"]) / "source_freeze.json"),
                    "sha256": record["source_freeze_sha256"],
                }
                for fold, record in zip(source_freezes, campaign["folds"])
            ],
            *[
                {
                    "role": "model_freeze",
                    "fold_id": fold["fold_id"],
                    "path": str(Path(record["directory"]) / "model_freeze.json"),
                    "sha256": campaign["model_freezes"][index]["sha256"],
                }
                for index, (fold, record) in enumerate(zip(model_freezes, campaign["folds"]))
            ],
            *[
                {
                    "role": "target_metrics",
                    "fold_id": fold["fold_id"],
                    "path": target_campaign["folds"][index]["path"],
                    "sha256": target_campaign["folds"][index]["sha256"],
                }
                for index, fold in enumerate(target_folds)
            ],
        ],
    }
    payload["primary_metric"] = {
        "name": "equal_domain_recording_macro_f1",
        "value": pcrd_cost["mean_macro_f1"],
        "direction": "maximize",
        "best_condition": "pcrd",
    }
    payload["synthetic_dev"] = False
    payload["warnings"] = []
    if not detection_eligible:
        payload["warnings"].append(
            "Recognition results are complete; the registered detection arm is ineligible because shared real background is absent, so detection fields remain TBD."
        )
    if validate_output:
        validate_results(payload, output.parent, independent_recompute=False)
    atomic_json(output, payload)
    if validate_output:
        for fold in campaign["folds"]:
            state = RunState(campaign_dir / fold["directory"] / "run_state.json")
            if state.phase == RunPhase.TARGET_EVALUATED:
                state.transition(RunPhase.RESULTS_FINALIZED)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute H1-H4/M1 and write canonical results.json")
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    return parser


def main() -> int:
    print(run(**vars(build_parser().parse_args())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
