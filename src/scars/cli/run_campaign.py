from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any

import numpy as np
import yaml

from scars.audits.degeneracy import representation_similarity
from scars.audits.domain_probe import domain_probe_accuracy
from scars.audits.leakage import duplicate_audit, near_duplicate_audit, split_group_leakage_audit
from scars.baselines import H2_BANK_ORDER, h2_configuration_factory
from scars.data.manifest import load_manifest
from scars.data.splits import leave_one_dataset_out
from scars.data.windowing import WindowBatch, window_recordings
from scars.evaluation.classification import macro_f1, recording_level_metrics
from scars.evaluation.detection import detection_metrics, source_threshold
from scars.evaluation.robustness import normalized_curve_area
from scars.evaluation.statistics import cohens_d_paired, coherent_freedman_lane
from scars.probes.ridge import RidgeProbe
from scars.probes.small_resnet import SmallResNetProbe
from scars.results.provenance import environment_manifest, sha256_file
from scars.results.schema import empty_results
from scars.results.validator import PUBLICATION_DECISION_ENGINE_VERSION
from scars.results.writer import write_results
from scars.selection.nuisance import awgn, ism_mixing, registered_nuisance_cases
from scars.selection.pareto import ObjectiveRecord, select_unique
from scars.selection.sensitivity import source_instability
from scars.state import RunPhase, RunState


def _features(tensor: np.ndarray) -> np.ndarray:
    return tensor.reshape(len(tensor), -1)


def _deterministic(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _deterministic(item)
            for key, item in value.items()
            if key not in {"fit_timestamp", "created_at", "wallclock_time"}
        }
    if isinstance(value, list):
        return [_deterministic(item) for item in value]
    return value


def _artifact_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(_deterministic(value), sort_keys=True).encode()).hexdigest()


def _json_ready(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _gate_check(acquisition: dict[str, Any], budget: dict[str, Any]) -> None:
    domain = acquisition.get("domain_gate", {})
    compute = acquisition.get("compute_gate", {})
    eligible = domain.get("eligible_domain_ids", [])
    if not domain.get("gate_open") or len(eligible) < 3 or domain.get("complete_domain_count", 0) < 3:
        raise PermissionError("Acquisition gate is closed or has fewer than three complete domains")
    required_reviews = (
        "shared_ontology_verified",
        "grouping_verified",
        "licenses_verified",
        "label_mapping_verified",
        "checksum_manifests_verified",
        "sampling_policy_verified",
    )
    if not all(domain.get(key) for key in required_reviews):
        raise PermissionError("Ontology/group/license/checksum/sampling acquisition checks are incomplete")
    ontology = acquisition.get("ontology", {})
    if not ontology.get("shared_label_intersection") or not ontology.get("label_mapping_file"):
        raise PermissionError("Frozen ontology and label-mapping file are required")
    if not _is_sha256_text(ontology.get("label_mapping_sha256")):
        raise PermissionError("Frozen label-mapping SHA-256 is required")
    eligible_corpora = [
        corpus for corpus in acquisition.get("corpora", []) if corpus.get("dataset_id") in eligible
    ]
    if len(eligible_corpora) != len(eligible):
        raise PermissionError("Every eligible domain requires one acquisition corpus record")
    for corpus in eligible_corpora:
        if not corpus.get("complete") or not corpus.get("sample_rate_hz"):
            raise PermissionError("Eligible corpus is incomplete or lacks a sample rate")
        if not _is_sha256_text(corpus.get("manifest_sha256")) or not _is_sha256_text(
            corpus.get("file_checksums_sha256")
        ):
            raise PermissionError("Eligible corpus lacks reviewed manifest/checksum SHA-256 values")
    if not compute.get("gate_open"):
        raise PermissionError("Acquisition-manifest compute gate is closed")
    constraints = budget.get("hard_constraints", {})
    if not budget.get("gate", {}).get("allow_pareto_freeze") or any(
        constraints.get(key) is None
        for key in ("max_bytes_per_sample", "max_batch1_latency_ms", "max_estimated_macs")
    ):
        raise PermissionError("Deployment envelope is not frozen")


def _feasible(cost: dict[str, float], budget: dict[str, Any]) -> bool:
    limits = budget["hard_constraints"]
    return bool(
        cost["bytes_per_sample"] <= limits["max_bytes_per_sample"]
        and cost["batch1_latency_ms"] <= limits["max_batch1_latency_ms"]
        and cost["estimated_macs"] <= limits["max_estimated_macs"]
    )


def _is_sha256_text(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _recording_binary_scores(
    batch: WindowBatch,
    probability: np.ndarray,
    classes: np.ndarray,
    background_labels: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    background_indices = [i for i, label in enumerate(classes) if str(label) in background_labels]
    if not background_indices:
        raise ValueError("Probe class set has no registered background label")
    score = 1.0 - probability[:, background_indices].sum(axis=1)
    y, s = [], []
    for recording_id in sorted(set(batch.recording_ids.tolist())):
        mask = batch.recording_ids == recording_id
        labels = {str(label) for label in batch.labels[mask]}
        if len(labels) != 1:
            raise ValueError("A recording crosses labels")
        y.append(0 if next(iter(labels)) in background_labels else 1)
        s.append(float(np.mean(score[mask])))
    return np.asarray(y, dtype=int), np.asarray(s, dtype=float)


def _robustness_curve(
    representation,
    probe: RidgeProbe,
    batch: WindowBatch,
    transform,
    grid: list[float],
    seed: int,
) -> list[dict[str, float]]:
    output = []
    for index, severity in enumerate(grid):
        rng = np.random.default_rng(seed + index)
        perturbed = np.stack([transform(x, rng, severity) for x in batch.iq])
        probability = probe.predict_proba(_features(representation.transform(perturbed)))
        metric = recording_level_metrics(
            batch.labels, batch.recording_ids, probability, probe.classes_
        )
        output.append({"severity": severity, "macro_f1": float(metric["macro_f1"])})
    return output


def _combine_batches(batches: list[WindowBatch]) -> WindowBatch:
    return WindowBatch(
        iq=np.concatenate([batch.iq for batch in batches]),
        labels=np.concatenate([batch.labels for batch in batches]),
        recording_ids=np.concatenate([batch.recording_ids for batch in batches]),
        domains=np.concatenate([batch.domains for batch in batches]),
        starts=np.concatenate([batch.starts for batch in batches]),
        ends=np.concatenate([batch.ends for batch in batches]),
    )


def _label_shuffle_control(
    representation,
    train: WindowBatch,
    validation: WindowBatch,
    observed_macro_f1: float,
    seed: int = 24_023,
    permutations: int = 10_000,
) -> dict[str, Any]:
    train_features = _features(representation.transform(train.iq))
    validation_features = _features(representation.transform(validation.iq))
    recording_order = sorted(set(train.recording_ids.tolist()))
    labels = []
    for recording_id in recording_order:
        values = np.unique(train.labels[train.recording_ids == recording_id])
        if len(values) != 1:
            raise ValueError("Training recording crosses labels")
        labels.append(values[0])
    labels = np.asarray(labels, dtype=object)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        shuffled = rng.permutation(labels)
        mapping = dict(zip(recording_order, shuffled))
        window_labels = np.asarray([mapping[value] for value in train.recording_ids], dtype=object)
        probe = RidgeProbe().fit(train_features, window_labels)
        probability = probe.predict_proba(validation_features)
        metric = recording_level_metrics(
            validation.labels, validation.recording_ids, probability, probe.classes_
        )
        exceedances += metric["macro_f1"] >= observed_macro_f1 - 1.0e-15
    return {
        "status": "ok",
        "unit": "source_training_recording_label",
        "observed_macro_f1": observed_macro_f1,
        "p_value": float((1 + exceedances) / (permutations + 1)),
        "permutations": permutations,
        "seed": seed,
    }


def _resampled_recording_f1(metric: dict[str, Any], draw: np.ndarray) -> float:
    truth = np.asarray(metric["recording_truth"], dtype=object)[draw]
    prediction = np.asarray(metric["recording_prediction"], dtype=object)[draw]
    return macro_f1(truth, prediction)


def _finite_effect(value: float) -> dict[str, Any]:
    return (
        {"value": float(value), "status": "ok"}
        if np.isfinite(value)
        else {"value": None, "status": "undefined_zero_variance"}
    )


def _nested_h2_bootstrap(
    source_cache: list[tuple],
    held_folds: list[dict[str, Any]],
    resamples: int = 10_000,
    seed: int = 24_022,
) -> dict[str, Any]:
    """Nested domain/recording bootstrap with paired draws across configurations."""
    rng = np.random.default_rng(seed)
    domain_count = len(source_cache)
    estimates: list[float] = []
    blocked_replicates = 0
    for _ in range(resamples):
        outer = rng.integers(0, domain_count, size=domain_count)
        l_rows, f_rows, held_rows = [], [], []
        for fold_index in outer:
            records = source_cache[int(fold_index)][7]
            held_conditions = held_folds[int(fold_index)]["conditions"]
            first_case = next(iter(records[0]["stability"]["recording_values"].values()))
            domain_ids: dict[str, list[str]] = {}
            for item in first_case:
                domain_ids.setdefault(item["domain"], []).append(item["recording_id"])
            source_draws = {
                domain: rng.choice(sorted(set(ids)), size=len(set(ids)), replace=True).tolist()
                for domain, ids in domain_ids.items()
            }
            validation_count = len(records[0]["validation"]["recording_order"])
            held_count = len(held_conditions[H2_BANK_ORDER[0]]["recording_order"])
            validation_draw = rng.integers(0, validation_count, size=validation_count)
            held_draw = rng.integers(0, held_count, size=held_count)
            fold_l, fold_f, fold_held = [], [], []
            for record, stable_id in zip(records, H2_BANK_ORDER):
                case_values = []
                for details in record["stability"]["recording_values"].values():
                    lookup = {
                        (item["domain"], item["recording_id"]): item["recording_max"]
                        for item in details
                    }
                    case_values.append(
                        float(
                            np.mean(
                                [
                                    np.median([lookup[(domain, rid)] for rid in draw])
                                    for domain, draw in source_draws.items()
                                ]
                            )
                        )
                    )
                fold_l.append(max(case_values))
                fold_f.append(_resampled_recording_f1(record["validation"], validation_draw))
                fold_held.append(
                    _resampled_recording_f1(held_conditions[stable_id], held_draw)
                )
            l_rows.append(fold_l)
            f_rows.append(fold_f)
            held_rows.append(fold_held)
        l_sample = np.asarray(l_rows)
        f_sample = np.asarray(f_rows)
        held_sample = np.asarray(held_rows)
        statistic = coherent_freedman_lane(
            l_sample,
            f_sample - held_sample,
            f_sample,
            seed=24_021,
            permutations=0,
        )
        if statistic["status"] != "ok":
            blocked_replicates += 1
        else:
            estimates.append(float(statistic["estimate"]))
    if blocked_replicates:
        return {
            "status": "blocked",
            "reason": "degenerate_partial_rank_in_nested_bootstrap",
            "blocked_replicates": blocked_replicates,
            "requested_resamples": resamples,
            "seed": seed,
            "outer_unit": "domain",
            "inner_unit": "recording_or_event",
        }
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "status": "ok",
        "interval": [float(low), float(high)],
        "confidence": 0.95,
        "resamples": resamples,
        "seed": seed,
        "outer_unit": "domain",
        "inner_unit": "recording_or_event",
        "paired_configuration_draws": True,
    }


def run(
    recordings_manifest: Path,
    acquisition_manifest: Path,
    deployment_budget: Path,
    output_dir: Path,
    seed: int,
    window_samples: int,
    hop_samples: int,
    authorize_target: bool,
) -> Path:
    if not authorize_target:
        raise PermissionError("Refused: --authorize-target is required for irreversible target access")
    raise PermissionError(
        "Refused: legacy monolithic runner is retired. Use preflight_mat -> freeze_source -> "
        "train_source_models -> evaluate_target -> finalize_results."
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Refused: campaign output directory must be new and empty")
    started = perf_counter()
    acquisition = json.loads(acquisition_manifest.read_text(encoding="utf-8"))
    budget = yaml.safe_load(deployment_budget.read_text(encoding="utf-8"))
    _gate_check(acquisition, budget)
    recordings = load_manifest(recordings_manifest)
    eligible = set(acquisition["domain_gate"]["eligible_domain_ids"])
    recordings = [record for record in recordings if record.dataset in eligible]
    if {record.dataset for record in recordings} != eligible:
        raise ValueError("Recordings manifest does not cover every gated eligible domain")
    folds = leave_one_dataset_out(recordings, seed=seed)
    if len(folds) < 3:
        raise ValueError("Confirmatory campaign requires at least three held domains")

    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[3]
    config_paths = sorted((project_root / "configs").glob("*.yaml"))
    config_paths += sorted((project_root / "configs").glob("*.json"))
    results = empty_results("real_confirmatory", output_dir.name)
    results["run"].update(
        {
            "command": " ".join(sys.argv),
            "device": f"cpu:{platform.processor() or platform.machine()}",
            "seeds": [seed],
        }
    )
    results["provenance"] = environment_manifest(project_root, config_paths)
    results["provenance"]["recordings_manifest_sha256"] = sha256_file(recordings_manifest)
    results["provenance"]["acquisition_manifest_sha256"] = sha256_file(acquisition_manifest)
    results["provenance"]["ontology_sha256"] = _artifact_hash(acquisition.get("ontology"))
    results["provenance"]["label_map_sha256"] = acquisition["ontology"][
        "label_mapping_sha256"
    ]
    # Raw-file checksums are imported from the already-reviewed acquisition
    # manifests. Re-reading held waveform files here would itself violate the
    # target lock before the global source-only barrier.
    results["provenance"]["dataset_manifest_sha256"] = {
        corpus["dataset_id"]: corpus.get("manifest_sha256")
        for corpus in acquisition.get("corpora", [])
        if corpus.get("dataset_id") in eligible
    }
    results["datasets"] = {
        "domains": sorted(eligible),
        "ontology": acquisition.get("ontology"),
        "acquisition_gate": acquisition,
    }
    results["representations"]["configuration_order"] = list(H2_BANK_ORDER)
    split_records = [
        {
            "fold_id": fold.fold_id,
            "source_train_recordings": [record.recording_id for record in fold.source_train],
            "source_validation_recordings": [record.recording_id for record in fold.source_validation],
            "held_target_recordings": [record.recording_id for record in fold.held_target],
            "coverage": fold.coverage,
            "windowing_after_split": True,
        }
        for fold in folds
    ]
    results["splits"]["folds"] = split_records
    results["splits"]["manifest_hash"] = _artifact_hash(split_records)
    results["provenance"]["split_manifest_hash"] = results["splits"]["manifest_hash"]

    source_cache = []
    nuisance_cases = registered_nuisance_cases()
    timing = budget.get("measurement_policy", {})
    warmups = int(timing.get("warmup_batches", 20))
    repeats = int(timing.get("measured_batches", 100))
    timing_manifest = {
        "deployment_budget_gate_open": True,
        "hardware": results["run"]["device"],
        "warmup_batches": warmups,
        "measured_batches": repeats,
        "batch_size": 1,
        "statistic": timing.get("statistic", "median"),
        "software": results["provenance"].get("packages"),
    }
    background_by_domain = {
        domain: {
            str(record.label)
            for record in recordings
            if record.dataset == domain and record.is_background and record.label is not None
        }
        for domain in eligible
    }
    detection_gate_open = all(background_by_domain[domain] for domain in eligible)
    background_labels = (
        set().union(*(background_by_domain[domain] for domain in eligible))
        if detection_gate_open
        else set()
    )
    learned_seeds = [11, 23, 37, 53, 71]
    learned_h3_ids = {
        "W": "wst_j4q2_16",
        "C": "cyclic_a8_16",
        "W+C": "wstj4q2_cyclica8_16",
    }
    for fold_index, fold in enumerate(folds):
        state = RunState(output_dir / "folds" / str(fold_index) / "run_state.json")
        state.save()
        state.transition(RunPhase.SPLITS_FROZEN)
        train = window_recordings(fold.source_train, window_samples, hop_samples)
        validation = window_recordings(fold.source_validation, window_samples, hop_samples)
        split_record = split_records[fold_index]
        fitted: dict[str, tuple[Any, RidgeProbe]] = {}
        records = []
        objectives = []
        common_nuisance_seed = seed + 10_000 * fold_index
        bank = h2_configuration_factory()
        for representation in bank:
            stable_id = representation.config.stable_id
            representation.fit(
                train.iq,
                fold.fold_id,
                recording_ids=train.recording_ids,
            )
            probe = RidgeProbe().fit(_features(representation.transform(train.iq)), train.labels)
            validation_probability = probe.predict_proba(
                _features(representation.transform(validation.iq))
            )
            validation_metric = recording_level_metrics(
                validation.labels,
                validation.recording_ids,
                validation_probability,
                probe.classes_,
            )
            instability = source_instability(
                representation,
                train.iq,
                train.recording_ids,
                train.domains,
                nuisance_cases,
                common_nuisance_seed,
            )
            cost = representation.measured_cost(train.iq, warmups=warmups, repeats=repeats)
            objective = ObjectiveRecord(
                stable_id=stable_id,
                nuisance_sensitivity=float(instability["value"]),
                source_selection_macro_f1=float(validation_metric["macro_f1"]),
                batch1_latency_ms=float(cost["batch1_latency_ms"]),
                bytes_per_sample=float(cost["bytes_per_sample"]),
                estimated_macs=float(cost["estimated_macs"]),
                feasible=_feasible(cost, budget),
            )
            objectives.append(objective)
            fitted[stable_id] = (representation, probe)
            records.append(
                {
                    "stable_id": stable_id,
                    "objective": asdict(objective),
                    "stability": instability,
                    "validation": validation_metric,
                    "representation_artifact": representation.source_artifact(),
                    "probe_artifact": probe.source_artifact(),
                    "nuisance_seed": common_nuisance_seed,
                }
            )
            results["costs"]["conditions"].setdefault(stable_id, []).append(cost)
        if tuple(record["stable_id"] for record in records) != H2_BANK_ORDER:
            raise AssertionError("Source H2 bank is incomplete")
        learned_models: dict[str, dict[int, tuple[Any, SmallResNetProbe]]] = {}
        learned_records = []
        for mask, stable_id in learned_h3_ids.items():
            representation, _ = fitted[stable_id]
            train_tensor = representation.transform(train.iq)
            validation_tensor = representation.transform(validation.iq)
            learned_models[mask] = {}
            for learned_seed in learned_seeds:
                learned = SmallResNetProbe(learned_seed).fit(
                    train_tensor,
                    train.labels,
                    validation_tensor,
                    validation.labels,
                    validation.recording_ids,
                )
                artifact_path = (
                    output_dir
                    / "folds"
                    / str(fold_index)
                    / "learned"
                    / f"{mask.replace('+', 'plus')}-seed-{learned_seed}.pt"
                )
                artifact = learned.save_artifact(artifact_path)
                artifact["path"] = str(artifact_path.relative_to(output_dir))
                learned_records.append(
                    {"mask": mask, "stable_id": stable_id, "seed": learned_seed, **artifact}
                )
                learned_models[mask][learned_seed] = (representation, learned)
        selected, front = select_unique(objectives)
        selected_representation, selected_probe = fitted[selected]
        observed_selected_source_f1 = next(
            record["validation"]["macro_f1"] for record in records if record["stable_id"] == selected
        )
        label_shuffle = _label_shuffle_control(
            selected_representation,
            train,
            validation,
            observed_selected_source_f1,
            seed=24_023,
        )
        detection_threshold = None
        if background_labels:
            validation_probability = selected_probe.predict_proba(
                _features(selected_representation.transform(validation.iq))
            )
            y_val, s_val = _recording_binary_scores(
                validation, validation_probability, selected_probe.classes_, background_labels
            )
            detection_threshold = source_threshold(y_val, s_val)
        freeze_payload = {
            "fold_id": fold.fold_id,
            "split": split_record,
            "configuration_order": list(H2_BANK_ORDER),
            "h2_bank_status": "complete",
            "records": records,
            "pareto": front,
            "selected": selected,
            "source_detection_threshold": detection_threshold,
            "label_shuffle_control": label_shuffle,
            "learned_h3_artifacts": learned_records,
            "timing_manifest": timing_manifest,
            "provenance": {
                "recordings_manifest_sha256": results["provenance"]["recordings_manifest_sha256"],
                "acquisition_manifest_sha256": results["provenance"]["acquisition_manifest_sha256"],
                "dataset_manifest_sha256": results["provenance"]["dataset_manifest_sha256"],
                "combined_config_hash": results["provenance"]["combined_config_hash"],
                "config_hashes": results["provenance"]["config_hashes"],
                "split_manifest_hash": results["splits"]["manifest_hash"],
            },
            "nuisance_case_order": [case.id for case in nuisance_cases],
        }
        state.transition(RunPhase.SOURCE_FITTING_COMPLETE)
        freeze_path = output_dir / "folds" / str(fold_index) / "source_freeze.json"
        _atomic_json(freeze_path, freeze_payload)
        state.source_artifact_hash = sha256_file(freeze_path)
        state.transition(RunPhase.PARETO_FROZEN)
        results["source_fitting"]["folds"].append(
            {
                "fold_id": fold.fold_id,
                "h2_bank_status": "complete",
                "configuration_order": list(H2_BANK_ORDER),
                "source_freeze_path": str(freeze_path.relative_to(output_dir)),
                "source_freeze_sha256": state.source_artifact_hash,
                "records": records,
            }
        )
        results["source_selection"]["folds"].append(
            {
                "fold_id": fold.fold_id,
                "pareto": front,
                "selected_theta_hat": selected,
                "source_artifact_hash": state.source_artifact_hash,
                "source_detection_threshold": detection_threshold,
            }
        )
        source_cache.append(
            (
                fold_index,
                fold,
                train,
                validation,
                fitted,
                state,
                selected,
                records,
                freeze_path,
                detection_threshold,
                label_shuffle,
                learned_models,
                learned_records,
            )
        )

    # Global source-only barrier: no target is materialized until all folds and
    # all twelve H2 configurations have immutable artifacts.
    results["source_selection"]["policy"] = {
        "axes": ["nuisance_sensitivity", "source_selection_macro_f1", "bytes_per_sample", "estimated_macs", "batch1_latency_ms"],
        "source_f1_absolute_tolerance": 0.01,
        "target_metrics_used": False,
        "configuration_order": list(H2_BANK_ORDER),
    }
    for (
        *_,
        state,
        selected,
        records,
        freeze_path,
        detection_threshold,
        label_shuffle,
        learned_models,
        learned_records,
    ) in source_cache:
        del selected, records, detection_threshold, label_shuffle, learned_models, learned_records
        persisted = RunState(state.path)
        if persisted.phase != RunPhase.PARETO_FROZEN:
            raise RuntimeError("Global barrier found a fold that is not PARETO_FROZEN")
        if persisted.source_artifact_hash != sha256_file(freeze_path):
            raise RuntimeError("Global barrier source-freeze hash mismatch")

    selected_scores = []
    target_batches: list[WindowBatch] = []
    h3_similarity = []
    domain_probe_records = []
    background_control_records = []
    learned_h3_folds = []
    snr_grid = [20.0, 10.0, 0.0, -10.0]
    sir_grid = [20.0, 10.0, 0.0, -10.0]
    for (
        fold_index,
        fold,
        train,
        validation,
        fitted,
        state,
        selected,
        source_records,
        freeze_path,
        detection_threshold,
        label_shuffle,
        learned_models,
        learned_records,
    ) in source_cache:
        del freeze_path, learned_records
        state.transition(RunPhase.TARGET_UNLOCKED)
        target = window_recordings(
            fold.held_target,
            window_samples,
            hop_samples,
            access_role="held_target",
            state=state,
        )
        condition_results = {}
        for stable_id in H2_BANK_ORDER:
            representation, probe = fitted[stable_id]
            probability = probe.predict_proba(_features(representation.transform(target.iq)))
            condition_results[stable_id] = recording_level_metrics(
                target.labels, target.recording_ids, probability, probe.classes_
            )
        selected_scores.append(float(condition_results[selected]["macro_f1"]))
        target_batches.append(target)
        w_representation, _ = fitted["wst_j4q2_16"]
        c_representation, _ = fitted["cyclic_a8_16"]
        similarity = representation_similarity(
            w_representation.transform(target.iq), c_representation.transform(target.iq)
        )
        h3_similarity.append({"fold_id": fold.fold_id, **similarity})
        learned_condition_results: dict[str, dict[str, Any]] = {}
        for mask in ("W", "C", "W+C"):
            learned_condition_results[mask] = {}
            for learned_seed in learned_seeds:
                representation, learned_probe = learned_models[mask][learned_seed]
                probability = learned_probe.predict_proba(representation.transform(target.iq))
                learned_condition_results[mask][str(learned_seed)] = recording_level_metrics(
                    target.labels,
                    target.recording_ids,
                    probability,
                    learned_probe.classes_,
                )
        learned_h3_folds.append(
            {"fold_id": fold.fold_id, "conditions": learned_condition_results}
        )
        wc_representation, _ = fitted["wstj4q2_cyclica8_16"]
        domain_probe_records.append(
            {
                "fold_id": fold.fold_id,
                "accuracy": domain_probe_accuracy(
                    _features(wc_representation.transform(train.iq)),
                    train.domains,
                    train.recording_ids,
                ),
                "replication_unit": "recording",
                "recording_disjoint_split": True,
            }
        )
        selected_representation, selected_probe = fitted[selected]
        snr_curve = _robustness_curve(
            selected_representation, selected_probe, target, awgn, snr_grid, seed + 100_000 + fold_index
        )
        sir_curve = _robustness_curve(
            selected_representation, selected_probe, target, ism_mixing, sir_grid, seed + 200_000 + fold_index
        )
        results["robustness"]["snr"].append({"fold_id": fold.fold_id, "curve": snr_curve})
        results["robustness"]["sir"].append({"fold_id": fold.fold_id, "curve": sir_curve})
        if background_labels:
            if detection_threshold is None:
                raise RuntimeError("Missing source-frozen detection threshold")
            target_probability = selected_probe.predict_proba(
                _features(selected_representation.transform(target.iq))
            )
            y_target, s_target = _recording_binary_scores(
                target, target_probability, selected_probe.classes_, background_labels
            )
            results["detection"]["folds"].append(
                {
                    "fold_id": fold.fold_id,
                    **detection_metrics(y_target, s_target, detection_threshold),
                }
            )
            background_control_records.append(
                {
                    "fold_id": fold.fold_id,
                    "background_recording_count": int(np.sum(y_target == 0)),
                    "mean_uav_score_on_background": float(np.mean(s_target[y_target == 0]))
                    if np.any(y_target == 0)
                    else None,
                    "maximum_uav_score_on_background": float(np.max(s_target[y_target == 0]))
                    if np.any(y_target == 0)
                    else None,
                }
            )
        state.transition(RunPhase.TARGET_EVALUATED)
        state.transition(RunPhase.RESULTS_FINALIZED)
        results["held_domain"]["folds"].append(
            {
                "fold_id": fold.fold_id,
                "selected_theta_hat": selected,
                "conditions": condition_results,
                "configuration_order": list(H2_BANK_ORDER),
                "target_reads": state.target_reads,
                "target_unlocks": state.target_unlocks,
                "target_recording_ids": list(state.target_recording_ids),
                "source_label_shuffle": label_shuffle,
            }
        )

    focal_id = "wstj4q2_cyclica8_16"
    focal_scores = [
        float(fold["conditions"][focal_id]["macro_f1"])
        for fold in results["held_domain"]["folds"]
    ]
    results["held_domain"].update(
        {
            "focal_condition_id": focal_id,
            "focal_average_macro_f1": float(np.mean(focal_scores)),
            "focal_worst_macro_f1": float(np.min(focal_scores)),
            "selected_average_macro_f1": float(np.mean(selected_scores)),
            "selected_worst_macro_f1": float(np.min(selected_scores)),
            # Retained schema aliases; these are never used to choose the focal row.
            "average_macro_f1": float(np.mean(selected_scores)),
            "worst_macro_f1": float(np.min(selected_scores)),
        }
    )
    l = np.asarray(
        [[record["stability"]["value"] for record in item[7]] for item in source_cache]
    )
    source_f1 = np.asarray(
        [[record["validation"]["macro_f1"] for record in item[7]] for item in source_cache]
    )
    held_f1 = np.asarray(
        [
            [fold["conditions"][stable_id]["macro_f1"] for stable_id in H2_BANK_ORDER]
            for fold in results["held_domain"]["folds"]
        ]
    )
    h2 = coherent_freedman_lane(l, source_f1 - held_f1, source_f1)
    h2["configuration_order"] = list(H2_BANK_ORDER)
    h2["configuration_count"] = len(H2_BANK_ORDER)
    h2["no_deletion_gate"] = "passed"
    results["statistics"]["h2"] = h2
    if h2["status"] == "ok":
        interval = _nested_h2_bootstrap(source_cache, results["held_domain"]["folds"])
        results["statistics"]["block_bootstrap"]["H2"] = interval
        results["hypotheses"]["H2"] = {
            "status": "blocked",
            "decision": None,
            "reason": (
                "H2 numerical test complete but H1-H4 Holm family is not yet complete"
                if interval["status"] == "ok"
                else interval["reason"]
            ),
        }
    else:
        results["hypotheses"]["H2"] = {
            "status": "blocked",
            "decision": None,
            "reason": h2.get("reason"),
        }
    # H1 source-frontier arm (descriptive decision inputs; four-way Holm remains pending).
    wc_ids = {
        "wstj4q2_cyclica8_8",
        "wstj4q2_cyclica8_16",
        "wstj4q2_cyclica8_32",
    }
    h1_folds = []
    for source_fold, selection_fold in zip(
        results["source_fitting"]["folds"], results["source_selection"]["folds"]
    ):
        record_by_id = {record["stable_id"]: record for record in source_fold["records"]}
        eligible_wc = [
            stable_id
            for stable_id in wc_ids.intersection(selection_fold["pareto"])
            if record_by_id[stable_id]["objective"]["feasible"]
        ]
        h1_folds.append(
            {
                "fold_id": source_fold["fold_id"],
                "eligible_nondominated_wc": sorted(eligible_wc),
                "direction_holds": bool(eligible_wc),
            }
        )
    results["statistics"]["h1"] = {
        "folds": h1_folds,
        "all_folds_direction_holds": all(item["direction_holds"] for item in h1_folds),
        "inferential_status": "pending_four_hypothesis_holm",
    }
    results["hypotheses"]["H1"] = {
        "status": "blocked",
        "decision": None,
        "reason": "source-frontier arm computed; registered four-hypothesis multiplicity family pending",
    }

    # H3 deterministic ridge deletion arm and degeneracy diagnostics.
    h3_ids = {
        "W": "wst_j4q2_16",
        "C": "cyclic_a8_16",
        "W+C": "wstj4q2_cyclica8_16",
    }
    h3_scores = {
        name: np.asarray(
            [fold["conditions"][stable_id]["macro_f1"] for fold in results["held_domain"]["folds"]]
        )
        for name, stable_id in h3_ids.items()
    }
    h3_deltas = {
        "combined_minus_W": h3_scores["W+C"] - h3_scores["W"],
        "combined_minus_C": h3_scores["W+C"] - h3_scores["C"],
    }
    results["statistics"]["h3"] = {
        key: {
            "per_domain": values.tolist(),
            "average": float(np.mean(values)),
            "worst": float(np.min(values)),
            "positive_domain_fraction": float(np.mean(values > 0)),
            "cohens_d_paired": _finite_effect(
                cohens_d_paired(
                    h3_scores["W+C"], h3_scores["W" if key.endswith("W") else "C"]
                )
            ),
        }
        for key, values in h3_deltas.items()
    }
    results["statistics"]["h3"]["learned_probe_seed_status"] = {
        "required": learned_seeds,
        "completed": learned_seeds,
        "status": "complete",
    }
    learned_deltas = {"combined_minus_W": [], "combined_minus_C": []}
    for fold in learned_h3_folds:
        for learned_seed in learned_seeds:
            seed_key = str(learned_seed)
            combined = fold["conditions"]["W+C"][seed_key]["macro_f1"]
            learned_deltas["combined_minus_W"].append(
                combined - fold["conditions"]["W"][seed_key]["macro_f1"]
            )
            learned_deltas["combined_minus_C"].append(
                combined - fold["conditions"]["C"][seed_key]["macro_f1"]
            )
    results["statistics"]["h3"]["learned_probe"] = {
        key: {
            "domain_seed_deltas": values,
            "average": float(np.mean(values)),
            "worst": float(np.min(values)),
            "positive_fraction": float(np.mean(np.asarray(values) > 0)),
            "cohens_d_paired": _finite_effect(
                cohens_d_paired(np.asarray(values), np.zeros(len(values)))
            ),
        }
        for key, values in learned_deltas.items()
    }
    results["ablations"]["learned_h3_folds"] = learned_h3_folds
    results["ablations"]["channel_masks"] = [
        {
            "stable_id": name,
            "h2_bank_id": stable_id,
            "per_domain_macro_f1": h3_scores[name].tolist(),
            "average_macro_f1": float(np.mean(h3_scores[name])),
            "worst_domain_macro_f1": float(np.min(h3_scores[name])),
        }
        for name, stable_id in h3_ids.items()
    ]
    results["ablations"]["controls"] = {
        "degeneracy": {
            "folds": h3_similarity,
            "cosine": float(np.mean([item["cosine"] for item in h3_similarity])),
            "normalized_mae": float(
                np.mean([item["normalized_mae"] for item in h3_similarity])
            ),
            "any_degenerate": any(item["degenerate"] for item in h3_similarity),
        }
    }
    results["hypotheses"]["H3"] = {
        "status": "blocked",
        "decision": None,
        "reason": "ridge and five-seed learned deletion arms complete; Holm gate pending",
    }

    # H4 resolution/performance/cost inputs.
    resolution_ids = {
        8: "wstj4q2_cyclica8_8",
        16: "wstj4q2_cyclica8_16",
        32: "wstj4q2_cyclica8_32",
    }
    resolution_records = []
    for resolution, stable_id in resolution_ids.items():
        scores = np.asarray(
            [fold["conditions"][stable_id]["macro_f1"] for fold in results["held_domain"]["folds"]]
        )
        costs = results["costs"]["conditions"][stable_id]
        resolution_records.append(
            {
                "resolution": resolution,
                "stable_id": stable_id,
                "per_domain_macro_f1": scores.tolist(),
                "average_macro_f1": float(np.mean(scores)),
                "worst_domain_macro_f1": float(np.min(scores)),
                "bytes_per_sample": float(np.mean([item["bytes_per_sample"] for item in costs])),
                "batch1_latency_ms": float(np.mean([item["batch1_latency_ms"] for item in costs])),
                "estimated_macs": float(np.mean([item["estimated_macs"] for item in costs])),
            }
        )
    high = next(item for item in resolution_records if item["resolution"] == 32)
    dominators = [
        item["resolution"]
        for item in resolution_records
        if item["resolution"] < 32
        and item["average_macro_f1"] >= high["average_macro_f1"] - 0.01
        and item["worst_domain_macro_f1"] >= high["worst_domain_macro_f1"] - 0.01
        and item["bytes_per_sample"] < high["bytes_per_sample"]
        and item["batch1_latency_ms"] < high["batch1_latency_ms"]
        and item["estimated_macs"] <= high["estimated_macs"]
    ]
    results["ablations"]["resolution"] = resolution_records
    results["statistics"]["h4"] = {
        "lower_resolution_dominators": dominators,
        "direction_holds": bool(dominators),
        "performance_tolerance": 0.01,
        "inferential_status": "paired_interval_and_holm_pending",
    }
    results["hypotheses"]["H4"] = {
        "status": "blocked",
        "decision": None,
        "reason": "resolution/cost arm computed; paired interval and Holm gates pending",
    }
    corpus_batch = _combine_batches(target_batches)
    results["leakage_audits"]["duplicates"] = duplicate_audit(corpus_batch)
    results["leakage_audits"]["near_duplicates"] = near_duplicate_audit(corpus_batch)
    results["leakage_audits"]["temporal"] = split_group_leakage_audit(split_records)
    results["leakage_audits"]["domain_probe"] = {
        "folds": domain_probe_records,
        "aggregate_accuracy": float(
            np.mean([record["accuracy"] for record in domain_probe_records])
        ),
        "probe": "source_only_ridge_on_wstj4q2_cyclica8_16",
    }
    results["leakage_audits"]["label_shuffle"] = {
        "folds": [fold["source_label_shuffle"] for fold in results["held_domain"]["folds"]],
        "registered_permutations_per_fold": 10_000,
    }
    results["leakage_audits"]["background_only"] = (
        {
            "status": "ok",
            "folds": background_control_records,
        }
        if background_control_records
        else {
            "status": "not_applicable",
            "reason": "no acquisition-gated background label",
        }
    )
    snr_means = [
        float(np.mean([fold["curve"][i]["macro_f1"] for fold in results["robustness"]["snr"]]))
        for i in range(len(snr_grid))
    ]
    sir_means = [
        float(np.mean([fold["curve"][i]["macro_f1"] for fold in results["robustness"]["sir"]]))
        for i in range(len(sir_grid))
    ]
    results["robustness"]["normalized_curve_auc"] = {
        "snr": normalized_curve_area(snr_grid, snr_means),
        "sir": normalized_curve_area(sir_grid, sir_means),
    }
    results["detection"]["summary"] = (
        None
        if not results["detection"]["folds"]
        else {
            key: float(np.mean([fold[key] for fold in results["detection"]["folds"]]))
            for key in ("auroc", "auprc", "far", "miss_rate")
        }
    )
    results["costs"]["measurement_manifest"] = timing_manifest
    results["baselines"]["folds"] = [
        {
            "fold_id": fold["fold_id"],
            "condition_ids": list(H2_BANK_ORDER),
            "note": "common H2 ridge-probe bank; learned baselines are a separate sensitivity arm",
        }
        for fold in results["held_domain"]["folds"]
    ]
    results["warnings"] = [] if detection_gate_open else [
        "Detection arm not estimable: every eligible corpus must contain an acquisition-gated background label."
    ]
    results["run"]["elapsed_sec"] = float(perf_counter() - started)
    results["run"]["status"] = "finalized"
    path = output_dir / "results.json"
    write_results(path, results)
    (output_dir / "manifest.json").write_text(
        json.dumps(results["provenance"], indent=2, sort_keys=True), encoding="utf-8"
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the acquisition-gated SCARS campaign")
    parser.add_argument("--recordings-manifest", type=Path, required=True)
    parser.add_argument("--acquisition-manifest", type=Path, required=True)
    parser.add_argument("--deployment-budget", type=Path, default=Path("configs/deployment_budget.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=24021)
    parser.add_argument("--window-samples", type=int, default=4096)
    parser.add_argument("--hop-samples", type=int, default=2048)
    parser.add_argument("--authorize-target", action="store_true")
    arguments = parser.parse_args()
    print(run(**vars(arguments)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
