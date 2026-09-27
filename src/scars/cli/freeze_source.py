from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from scars.ablation import channel_mask_grid
from scars.baselines import effective_h2_signature, frozen_external_sota_registry, h2_configuration_factory
from scars.data.manifest import load_manifest
from scars.data.windowing import window_recordings
from scars.evaluation.classification import macro_f1, recording_level_metrics
from scars.audits.degeneracy import representation_similarity
from scars.audits.domain_probe import domain_probe_accuracy, domain_probe_shuffle_threshold
from scars.experiment.common import atomic_json, flatten, load_split_plan, one_window_per_recording
from scars.probes.ridge import RidgeProbe
from scars.representations.tensor import RepresentationConfig, SourceFittedTensor
from scars.results.provenance import environment_manifest, sha256_file
from scars.results.registry import CONFIRMATORY_H2_CONFIGURATION_ORDER
from scars.selection.feasibility import DeploymentBudget
from scars.selection.nuisance import registered_nuisance_cases
from scars.selection.pareto import ObjectiveRecord, select_unique
from scars.selection.sensitivity import source_instability
from scars.state import RunPhase, RunState


H2_CONFIGURATION_ORDER = CONFIRMATORY_H2_CONFIGURATION_ORDER


def _candidates() -> list[RepresentationConfig]:
    masks = channel_mask_grid(16)
    candidates = [masks[name] for name in ("W", "C", "W+C", "W+C+E", "W+C+E+S")]
    candidates.extend(
        [
            RepresentationConfig("wst_J3Q1", use_c=False, output_bins=16, wst_j=3, wst_q=1),
            RepresentationConfig("wst_J3Q2", use_c=False, output_bins=16, wst_j=3, wst_q=2),
            RepresentationConfig("wst_J3Q4", use_c=False, output_bins=16, wst_j=3, wst_q=4),
            RepresentationConfig("wst_J4Q1", use_c=False, output_bins=16, wst_j=4, wst_q=1),
            RepresentationConfig("wst_J4Q2", use_c=False, output_bins=16, wst_j=4, wst_q=2),
            RepresentationConfig("wst_J4Q4", use_c=False, output_bins=16, wst_j=4, wst_q=4),
            RepresentationConfig("wst_J5Q1", use_c=False, output_bins=16, wst_j=5, wst_q=1),
            RepresentationConfig("wst_J5Q2", use_c=False, output_bins=16, wst_j=5, wst_q=2),
            RepresentationConfig("wst_J5Q4", use_c=False, output_bins=16, wst_j=5, wst_q=4),
            RepresentationConfig("cyclic_A4", use_w=False, output_bins=16, cyclic_count=4),
            RepresentationConfig("cyclic_A8", use_w=False, output_bins=16, cyclic_count=8),
            RepresentationConfig("cyclic_A16", use_w=False, output_bins=16, cyclic_count=16),
            RepresentationConfig("cyclic_permuted", use_w=False, output_bins=16, cyclic_count=8, cyclic_permutation=True),
            RepresentationConfig("norm_percentile", output_bins=16, normalization="percentile"),
            RepresentationConfig("norm_zscore", output_bins=16, normalization="zscore"),
            RepresentationConfig("norm_none", output_bins=16, normalization="none"),
            RepresentationConfig("resolution_8", output_bins=8),
            RepresentationConfig("resolution_16", output_bins=16),
            RepresentationConfig("resolution_32", output_bins=32),
            RepresentationConfig(
                "STFT", use_w=False, use_c=False, use_s=True, output_bins=16
            ),
        ]
    )
    if len({item.stable_id for item in candidates}) != len(candidates):
        raise AssertionError("Source candidate IDs are not unique")
    return candidates


def _source_channel_decisions(
    representation: SourceFittedTensor,
    fit_batch,
    selection_batch,
    *,
    seed: int,
    allowed_families: tuple[str, ...],
    allow_insufficient_domain_probe: bool = False,
) -> dict[str, object]:
    """Freeze family retain/exclude decisions without source-validation or target use."""
    fit_tensor = representation.transform(fit_batch.iq)
    selection_tensor = representation.transform(selection_batch.iq)
    families = representation.config.active_families()
    full_probe = RidgeProbe().fit(flatten(fit_tensor), fit_batch.labels)
    full_probability = full_probe.predict_proba(flatten(selection_tensor))
    full_f1 = recording_level_metrics(
        selection_batch.labels,
        selection_batch.recording_ids,
        full_probability,
        full_probe.classes_,
    )["macro_f1"]
    records = []
    for family_index, family in enumerate(families):
        zeroed = selection_tensor.copy()
        zeroed[:, family_index] = 0.0
        similarity = representation_similarity(selection_tensor, zeroed)
        family_features = flatten(selection_tensor[:, family_index : family_index + 1])
        observed_domain_probe = domain_probe_accuracy(
            family_features,
            selection_batch.domains,
            selection_batch.recording_ids,
            allow_insufficient=allow_insufficient_domain_probe,
        )
        shuffle = domain_probe_shuffle_threshold(
            family_features,
            selection_batch.domains,
            selection_batch.recording_ids,
            permutations=10_000,
            seed=seed + family_index,
            allow_insufficient=allow_insufficient_domain_probe,
        )
        retained_indices = [index for index in range(len(families)) if index != family_index]
        if retained_indices:
            deletion_probe = RidgeProbe().fit(
                flatten(fit_tensor[:, retained_indices]), fit_batch.labels
            )
            deletion_probability = deletion_probe.predict_proba(
                flatten(selection_tensor[:, retained_indices])
            )
            deletion_f1 = recording_level_metrics(
                selection_batch.labels,
                selection_batch.recording_ids,
                deletion_probability,
                deletion_probe.classes_,
            )["macro_f1"]
            independent_value = float(full_f1 - deletion_f1)
            independent_pass = independent_value > 0.0
        else:
            deletion_f1 = None
            independent_value = None
            independent_pass = True
        shortcut_pass = observed_domain_probe <= float(shuffle["threshold"])
        selected_by_pareto = family in allowed_families
        retain = (
            selected_by_pareto
            and independent_pass
            and not bool(similarity["degenerate"])
            and shortcut_pass
        )
        records.append(
            {
                "family": family,
                "source_selection_full_macro_f1": float(full_f1),
                "source_selection_deleted_macro_f1": deletion_f1,
                "independent_value_delta": independent_value,
                "independent_value_pass": independent_pass,
                "similarity": similarity,
                "domain_probe_accuracy": float(observed_domain_probe),
                "domain_probe_shuffle_null": shuffle,
                "shortcut_pass": shortcut_pass,
                "selected_by_global_pareto": selected_by_pareto,
                "retain": retain,
                "rule": "global_pareto_family AND independent_delta_gt_0 AND not_degenerate AND domain_probe_le_shuffle_q95",
            }
        )
    active = [record["family"] for record in records if record["retain"]]
    return {
        "input_families": list(families),
        "active_families": active,
        "active_indices": [families.index(family) for family in active],
        "records": records,
        "fit_role": "source_fit",
        "decision_role": "source_selection",
        "source_validation_used": False,
        "target_used": False,
    }


def _recording_arrays(features: np.ndarray, batch):
    output_features, output_labels, output_domains, output_ids = [], [], [], []
    for recording_id in sorted(set(batch.recording_ids.tolist()), key=str):
        mask = batch.recording_ids == recording_id
        labels = np.unique(batch.labels[mask])
        domains = np.unique(batch.domains[mask])
        if len(labels) != 1 or len(domains) != 1:
            raise ValueError("A source recording crosses label/domain during a negative control")
        output_features.append(np.mean(features[mask], axis=0))
        output_labels.append(labels[0])
        output_domains.append(domains[0])
        output_ids.append(str(recording_id))
    return (
        np.asarray(output_features),
        np.asarray(output_labels, dtype=object),
        np.asarray(output_domains, dtype=object),
        output_ids,
    )


def _source_negative_controls(
    representation: SourceFittedTensor,
    fit_batch,
    selection_batch,
    *,
    seed: int,
    active_indices: list[int],
) -> dict[str, object]:
    if not active_indices:
        return {
            "label_shuffle": {"status": "failed", "reason": "empty_active_family_set"},
            "background_only": {"status": "failed", "reason": "empty_active_family_set"},
        }
    fit_features = flatten(representation.transform(fit_batch.iq)[:, active_indices])
    selection_features = flatten(
        representation.transform(selection_batch.iq)[:, active_indices]
    )
    fit_x, fit_y, fit_domain, fit_ids = _recording_arrays(fit_features, fit_batch)
    selection_x, selection_y, selection_domain, selection_ids = _recording_arrays(
        selection_features, selection_batch
    )
    real_prediction = RidgeProbe().fit(fit_x, fit_y).predict(selection_x)
    real_macro_f1 = macro_f1(selection_y, real_prediction)
    rng = np.random.default_rng(seed)
    label_shuffle_values = []
    for _ in range(10_000):
        shuffled = fit_y[rng.permutation(len(fit_y))]
        prediction = RidgeProbe().fit(fit_x, shuffled).predict(selection_x)
        label_shuffle_values.append(macro_f1(selection_y, prediction))

    fit_background = fit_y.astype(str) == "background"
    selection_background = selection_y.astype(str) == "background"
    if not np.any(fit_background) or not np.any(selection_background):
        background_control = {
            "status": "not_applicable",
            "reason": "background_missing_detection_arm_ineligible",
        }
    else:
        observed_prediction = RidgeProbe(l2=0.05).fit(
            fit_x[fit_background], fit_domain[fit_background]
        ).predict(selection_x[selection_background])
        observed = float(np.mean(observed_prediction == selection_domain[selection_background]))
        null_values = []
        background_domains = fit_domain[fit_background]
        for _ in range(10_000):
            shuffled = background_domains[rng.permutation(len(background_domains))]
            prediction = RidgeProbe(l2=0.05).fit(
                fit_x[fit_background], shuffled
            ).predict(selection_x[selection_background])
            null_values.append(float(np.mean(prediction == selection_domain[selection_background])))
        threshold = float(np.quantile(null_values, 0.95, method="higher"))
        background_control = {
            "status": "passed" if observed <= threshold else "failed",
            "domain_probe_accuracy": observed,
            "shuffle_q95": threshold,
            "permutations": 10_000,
            "fit_role": "source_fit_background_only",
            "evaluation_role": "source_selection_background_only",
        }
    shuffle_q95 = float(np.quantile(label_shuffle_values, 0.95, method="higher"))
    return {
        "label_shuffle": {
            "status": "passed"
            if (
                len(label_shuffle_values) == 10_000
                and np.all(np.isfinite(label_shuffle_values))
                and shuffle_q95 < real_macro_f1
            )
            else "failed",
            "permutations": 10_000,
            "seed": seed,
            "macro_f1_median": float(np.median(label_shuffle_values)),
            "macro_f1_q95": shuffle_q95,
            "unshuffled_source_selection_macro_f1": float(real_macro_f1),
            "pass_rule": "shuffle_q95_strictly_below_unshuffled_macro_f1",
            "unit": "physical_recording",
        },
        "background_only": background_control,
    }


def run(
    *,
    preflight_dir: Path,
    output_dir: Path,
    seed: int,
    window_samples: int,
    hop_samples: int,
    max_windows_per_recording: int,
) -> Path:
    preflight = json.loads((preflight_dir / "preflight.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "ready":
        raise PermissionError("Preflight is blocked; target-safe source fitting is not admissible")
    if seed != 24021:
        raise ValueError("The confirmatory source-freeze seed is fixed at 24021")
    expected_windowing = {
        "window_samples": window_samples,
        "hop_samples": hop_samples,
        "max_windows_per_recording": max_windows_per_recording,
    }
    observed_windowing = {
        key: preflight["memory_profile"][key] for key in expected_windowing
    }
    if observed_windowing != expected_windowing:
        raise ValueError("Windowing arguments differ from the preflight manifest")
    campaign_path = output_dir / "source_campaign.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        if existing.get("status") in {"all_source_folds_frozen", "all_source_models_frozen", "target_evaluated"}:
            return campaign_path
        raise RuntimeError("Existing source campaign has an incompatible status")
    output_dir.mkdir(parents=True, exist_ok=True)
    recordings_path = preflight_dir / preflight["artifacts"]["recordings_manifest"]
    splits_path = preflight_dir / preflight["artifacts"]["split_manifest"]
    recordings = load_manifest(recordings_path)
    folds = load_split_plan(splits_path, recordings)
    budget = DeploymentBudget(4096.0, 100.0, 1.0e8)
    project_root = Path(__file__).resolve().parents[3]
    config_paths = sorted((project_root / "configs").glob("*.yaml")) + sorted(
        (project_root / "configs").glob("*.json")
    )
    campaign_records = []
    started = perf_counter()
    for fold_index, fold in enumerate(folds):
        fold_dir = output_dir / f"fold-{fold_index:02d}"
        completed_freeze = fold_dir / "source_freeze.json"
        if completed_freeze.is_file():
            frozen = json.loads(completed_freeze.read_text(encoding="utf-8"))
            if frozen.get("fold_id") != fold["fold_id"]:
                raise RuntimeError("Resume refused: source fold identity mismatch")
            artifact_records = (
                list(frozen.get("candidates", {}).values())
                + list(frozen.get("h2_candidates", {}).values())
                + [
                frozen.get("canonical_tensor_artifact", {})
                ]
            )
            if any(
                not record.get("path")
                or not (fold_dir / record["path"]).is_file()
                or sha256_file(fold_dir / record["path"]) != record.get("sha256")
                for record in artifact_records
            ):
                raise RuntimeError("Resume refused: a frozen representation artifact drifted")
            campaign_records.append(
                {
                    "fold_id": fold["fold_id"],
                    "directory": fold_dir.name,
                    "source_freeze_sha256": sha256_file(completed_freeze),
                    "selected_stable_id": frozen["selected_stable_id"],
                    "resumed": True,
                }
            )
            continue
        if fold_dir.exists() and any(fold_dir.iterdir()):
            raise RuntimeError(
                f"Incomplete source fold {fold_dir.name} is not resumable; preserve it for audit and restart in a new output directory"
            )
        fold_dir.mkdir(exist_ok=True)
        state = RunState(fold_dir / "run_state.json")
        state.save()
        state.transition(RunPhase.SPLITS_FROZEN)
        batches = {
            role: window_recordings(
                fold[role],
                window_samples,
                hop_samples,
                max_windows_per_recording=max_windows_per_recording,
            )
            for role in (
                "source_fit",
                "source_calibration",
                "source_selection",
                "source_validation",
            )
        }
        instability_batch = one_window_per_recording(batches["source_selection"])
        h1_instability_batch = one_window_per_recording(batches["source_validation"])
        objective_records = []
        artifacts = {}
        source_validation_records = {}
        h2_source_validation_records = {}
        paired_instability_seed = seed + 10_000 * fold_index
        for config in _candidates():
            representation = SourceFittedTensor(config).fit(
                batches["source_fit"].iq,
                fold["fold_id"],
                "source_fit",
                batches["source_fit"].recording_ids,
            )
            fit_tensor = representation.transform(batches["source_fit"].iq)
            selection_tensor = representation.transform(batches["source_selection"].iq)
            probe = RidgeProbe().fit(flatten(fit_tensor), batches["source_fit"].labels)
            selection_probability = probe.predict_proba(flatten(selection_tensor))
            source_metric = recording_level_metrics(
                batches["source_selection"].labels,
                batches["source_selection"].recording_ids,
                selection_probability,
                probe.classes_,
            )
            instability = source_instability(
                representation,
                instability_batch.iq,
                instability_batch.recording_ids,
                instability_batch.domains,
                registered_nuisance_cases(),
                paired_instability_seed,
            )
            cost = representation.measured_cost(
                batches["source_fit"].iq[:1], warmups=20, repeats=100
            )
            feasible, failures = budget.check(cost)
            artifact = representation.source_artifact()
            artifact_path = fold_dir / f"representation-{config.stable_id}.json"
            artifact_hash = atomic_json(artifact_path, artifact)
            artifacts[config.stable_id] = {
                "path": artifact_path.name,
                "sha256": artifact_hash,
                "source_selection_macro_f1": source_metric["macro_f1"],
                "instability": instability,
                "cost": cost,
                "feasible": feasible,
                "feasibility_failures": failures,
                "probe": probe.source_artifact(),
                "domain_probe_accuracy": domain_probe_accuracy(
                    flatten(selection_tensor),
                    batches["source_selection"].domains,
                    batches["source_selection"].recording_ids,
                    allow_insufficient=preflight.get("campaign_mode") == "pilot_three_dataset",
                ),
            }
            objective_records.append(
                ObjectiveRecord(
                    config.stable_id,
                    float(instability["value"]),
                    float(source_metric["macro_f1"]),
                    float(cost["batch1_latency_ms"]),
                    float(cost["bytes_per_sample"]),
                    float(cost["estimated_macs"]),
                    feasible,
                )
            )
            validation_tensor = representation.transform(batches["source_validation"].iq)
            validation_probability = probe.predict_proba(flatten(validation_tensor))
            source_validation_records[config.stable_id] = {
                "metrics": recording_level_metrics(
                    batches["source_validation"].labels,
                    batches["source_validation"].recording_ids,
                    validation_probability,
                    probe.classes_,
                ),
                "instability": source_instability(
                    representation,
                    h1_instability_batch.iq,
                    h1_instability_batch.recording_ids,
                    h1_instability_batch.domains,
                    registered_nuisance_cases(),
                paired_instability_seed,
                ),
                "recording_domains": {
                    str(recording_id): str(
                        np.unique(
                            batches["source_validation"].domains[
                                batches["source_validation"].recording_ids == recording_id
                            ]
                        )[0]
                    )
                    for recording_id in sorted(
                        set(batches["source_validation"].recording_ids.tolist()), key=str
                    )
                },
            }

        # H2 is a separate fixed-probe diagnostic bank.  It must not change the
        # global SCARS candidate frontier, but every member is source-fitted and
        # frozen before target access under the same source roles.
        h2_artifacts = {}
        for h2_representation in h2_configuration_factory():
            h2_id = h2_representation.config.stable_id
            h2_representation.fit(
                batches["source_fit"].iq,
                fold["fold_id"],
                "source_fit",
                batches["source_fit"].recording_ids,
            )
            h2_fit = h2_representation.transform(batches["source_fit"].iq)
            h2_probe = RidgeProbe().fit(flatten(h2_fit), batches["source_fit"].labels)
            h2_validation = h2_representation.transform(batches["source_validation"].iq)
            h2_probability = h2_probe.predict_proba(flatten(h2_validation))
            h2_instability = source_instability(
                h2_representation,
                h1_instability_batch.iq,
                h1_instability_batch.recording_ids,
                h1_instability_batch.domains,
                registered_nuisance_cases(),
                paired_instability_seed,
            )
            h2_cost = h2_representation.measured_cost(
                batches["source_fit"].iq[:1], warmups=20, repeats=100
            )
            h2_feasible, h2_failures = budget.check(h2_cost)
            h2_artifact_path = fold_dir / f"h2-representation-{h2_id}.json"
            h2_artifact_hash = atomic_json(
                h2_artifact_path, h2_representation.source_artifact()
            )
            h2_artifacts[h2_id] = {
                "path": h2_artifact_path.name,
                "sha256": h2_artifact_hash,
                "cost": h2_cost,
                "feasible": h2_feasible,
                "feasibility_failures": h2_failures,
                "probe": h2_probe.source_artifact(),
                "effective_signature": effective_h2_signature(h2_representation),
            }
            h2_source_validation_records[h2_id] = {
                "metrics": recording_level_metrics(
                    batches["source_validation"].labels,
                    batches["source_validation"].recording_ids,
                    h2_probability,
                    h2_probe.classes_,
                ),
                "instability": h2_instability,
                "recording_domains": {
                    str(recording_id): str(
                        np.unique(
                            batches["source_validation"].domains[
                                batches["source_validation"].recording_ids == recording_id
                            ]
                        )[0]
                    )
                    for recording_id in sorted(
                        set(batches["source_validation"].recording_ids.tolist()), key=str
                    )
                },
            }
        selected_id, front = select_unique(objective_records)
        feasible_records = [record for record in objective_records if record.feasible]
        matrix = np.asarray(
            [
                [
                    record.nuisance_sensitivity,
                    -record.source_selection_macro_f1,
                    record.bytes_per_sample,
                    record.estimated_macs,
                    record.batch1_latency_ms,
                ]
                for record in feasible_records
            ],
            dtype=float,
        )
        span = np.maximum(np.ptp(matrix, axis=0), 1.0e-12)
        scalar_scores = np.sum((matrix - np.min(matrix, axis=0)) / span, axis=1)
        scalar_selected_id = feasible_records[int(np.argmin(scalar_scores))].stable_id
        selected_record = artifacts[selected_id]
        selected_representation = SourceFittedTensor.from_source_artifact(
            json.loads((fold_dir / selected_record["path"]).read_text(encoding="utf-8"))
        )
        selected_config = selected_representation.config
        canonical_config = RepresentationConfig(
            stable_id="canonical_WCES",
            use_w=True,
            use_c=True,
            use_e=True,
            use_s=True,
            output_bins=selected_config.output_bins,
            wst_j=selected_config.wst_j,
            wst_q=selected_config.wst_q,
            cyclic_count=selected_config.cyclic_count,
            frame_samples=selected_config.frame_samples,
            hop_samples=selected_config.hop_samples,
            normalization=selected_config.normalization,
        )
        canonical_representation = SourceFittedTensor(canonical_config).fit(
            batches["source_fit"].iq,
            fold["fold_id"],
            "source_fit",
            batches["source_fit"].recording_ids,
        )
        canonical_path = fold_dir / "representation-canonical-WCES.json"
        canonical_hash = atomic_json(
            canonical_path, canonical_representation.source_artifact()
        )
        canonical_cost = canonical_representation.measured_cost(
            batches["source_fit"].iq[:1], warmups=20, repeats=100
        )
        channel_decisions = _source_channel_decisions(
            canonical_representation,
            batches["source_fit"],
            batches["source_selection"],
            seed=seed + 30_000 * fold_index,
            allowed_families=selected_config.active_families(),
            allow_insufficient_domain_probe=preflight.get("campaign_mode") == "pilot_three_dataset",
        )
        negative_controls = _source_negative_controls(
            canonical_representation,
            batches["source_fit"],
            batches["source_selection"],
            seed=seed + 40_000 * fold_index,
            active_indices=channel_decisions["active_indices"],
        )
        state.transition(RunPhase.SOURCE_FITTING_COMPLETE)
        state.source_artifact_hash = canonical_hash
        state.transition(RunPhase.PARETO_FROZEN)
        freeze_payload = {
            "schema_version": "scars-source-freeze-2.0",
            "fold_id": fold["fold_id"],
            "source_roles": {
                role: [record.recording_id for record in fold[role]]
                for role in (
                    "source_fit",
                    "source_calibration",
                    "source_selection",
                    "source_validation",
                )
            },
            "held_target_recordings": [record.recording_id for record in fold["held_target"]],
            "candidate_order": [item.stable_id for item in _candidates()],
            "h2_configuration_order": list(H2_CONFIGURATION_ORDER),
            "h2_candidates": h2_artifacts,
            "candidates": artifacts,
            "pareto_front": front,
            "selected_stable_id": selected_id,
            "scalar_selected_stable_id": scalar_selected_id,
            "selected_artifact_sha256": selected_record["sha256"],
            "canonical_tensor_artifact": {
                "path": canonical_path.name,
                "sha256": canonical_hash,
                "shape_contract": [4, canonical_config.output_bins, canonical_config.output_bins],
                "dtype": "float32",
                "family_order": ["W", "C", "E", "S"],
                "cost": canonical_cost,
            },
            "active_families": channel_decisions["active_families"],
            "active_indices": channel_decisions["active_indices"],
            "channel_decisions": channel_decisions,
            "negative_controls": negative_controls,
            "source_validation_fixed_h1_records": source_validation_records,
            "h2_source_validation_records": h2_source_validation_records,
            "target_reads": 0,
        }
        from scars.experiment.pilot_policy import is_pilot, apply_fixed_families, policy_for
        if is_pilot(preflight):
            apply_fixed_families(freeze_payload, policy_for(preflight))
        freeze_hash = atomic_json(fold_dir / "source_freeze.json", freeze_payload)
        campaign_records.append(
            {
                "fold_id": fold["fold_id"],
                "directory": fold_dir.name,
                "source_freeze_sha256": freeze_hash,
                "selected_stable_id": selected_id,
            }
        )
    shutil.copy2(splits_path, output_dir / "splits.json")
    shutil.copy2(preflight_dir / "preflight.json", output_dir / "preflight.json")
    shutil.copy2(recordings_path, output_dir / "recordings_recognition.json")
    campaign = {
        "schema_version": "scars-source-campaign-2.0",
        "status": "all_source_folds_frozen",
        "target_access_authorized": False,
        "seed": seed,
        "windowing": expected_windowing,
        "external_sota_registry": frozen_external_sota_registry(),
        "folds": campaign_records,
        "elapsed_sec": perf_counter() - started,
        "provenance": environment_manifest(project_root, config_paths),
        "input_hashes": {
            "preflight": sha256_file(preflight_dir / "preflight.json"),
            "recordings": sha256_file(recordings_path),
            "splits": sha256_file(splits_path),
        },
    }
    atomic_json(output_dir / "source_campaign.json", campaign)
    return output_dir / "source_campaign.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze every source fold before target access")
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=24021)
    parser.add_argument("--window-samples", type=int, default=4096)
    parser.add_argument("--hop-samples", type=int, default=2048)
    parser.add_argument("--max-windows-per-recording", type=int, default=64)
    return parser


def main() -> int:
    print(run(**vars(build_parser().parse_args())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
