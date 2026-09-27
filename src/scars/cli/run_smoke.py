from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np

from scars.audits.leakage import duplicate_audit
from scars.data.adapters.synthetic import synthetic_development_corpus
from scars.data.splits import leave_one_dataset_out
from scars.data.windowing import window_recordings
from scars.evaluation.classification import recording_level_metrics
from scars.probes.ridge import RidgeProbe
from scars.representations.tensor import RepresentationConfig, SourceFittedTensor
from scars.results.provenance import environment_manifest
from scars.results.schema import empty_results
from scars.results.writer import write_results
from scars.selection.nuisance import default_smoke_cases
from scars.selection.pareto import ObjectiveRecord, select_unique
from scars.selection.sensitivity import source_instability
from scars.state import RunPhase, RunState


def _features(tensor: np.ndarray) -> np.ndarray:
    return tensor.reshape(len(tensor), -1)


def _artifact_hash(payload: object) -> str:
    def deterministic(value):
        if isinstance(value, dict):
            return {
                key: deterministic(item)
                for key, item in value.items()
                if key not in {"fit_timestamp", "created_at", "wallclock_time"}
            }
        if isinstance(value, list):
            return [deterministic(item) for item in value]
        return value

    return hashlib.sha256(json.dumps(deterministic(payload), sort_keys=True).encode()).hexdigest()


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _configs() -> list[RepresentationConfig]:
    base = {"output_bins": 8, "wst_j": 3, "wst_q": 1, "cyclic_count": 4}
    return [
        RepresentationConfig("W", use_w=True, use_c=False, **base),
        RepresentationConfig("C", use_w=False, use_c=True, **base),
        RepresentationConfig("W+C", use_w=True, use_c=True, **base),
    ]


def run(output_dir: Path, seed: int) -> Path:
    started = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    recordings = synthetic_development_corpus(seed=seed)
    folds = leave_one_dataset_out(recordings, seed=seed)
    conditions = _configs()
    results = empty_results("synthetic_dev", output_dir.name)
    results["run"].update(
        {
            "command": " ".join(sys.argv),
            "device": f"cpu:{platform.processor() or platform.machine()}",
            "seeds": [seed],
        }
    )
    project_root = Path(__file__).resolve().parents[3]
    config_paths = sorted((project_root / "configs").glob("*.yaml"))
    config_paths += sorted((project_root / "configs").glob("*.json"))
    results["provenance"] = environment_manifest(project_root.parent, config_paths)
    results["provenance"]["synthetic_generator"] = {
        "seed": seed,
        "recording_count": len(recordings),
        "code": "scars.data.adapters.synthetic.synthetic_development_corpus",
    }
    results["provenance"]["dataset_hashes"] = {
        record.recording_id: _array_hash(record.iq) for record in recordings if record.iq is not None
    }
    results["datasets"] = {
        "domains": sorted({record.dataset for record in recordings}),
        "ontology": ["class_0", "class_1", "class_2"],
        "acquisition_gate": {
            "confirmatory_open": False,
            "reason": "synthetic engineering corpus",
        },
    }
    results["representations"]["configuration_order"] = [
        condition.stable_id for condition in conditions
    ]
    source_cache = []
    # Phase A: freeze every fold's source artifacts before any synthetic target read.
    for fold_index, fold in enumerate(folds):
        state = RunState(output_dir / "folds" / str(fold_index) / "run_state.json")
        state.save()
        state.transition(RunPhase.SPLITS_FROZEN)
        train = window_recordings(fold.source_train, 512, 512)
        validation = window_recordings(fold.source_validation, 512, 512)
        split_record = {
            "fold_id": fold.fold_id,
            "source_train_recordings": [r.recording_id for r in fold.source_train],
            "source_validation_recordings": [r.recording_id for r in fold.source_validation],
            "held_target_recordings": [r.recording_id for r in fold.held_target],
            "windowing_after_split": True,
            "coverage": fold.coverage,
        }
        results["splits"]["folds"].append(split_record)
        fitted = {}
        objectives = []
        source_records = []
        for condition_index, condition in enumerate(conditions):
            representation = SourceFittedTensor(condition).fit(
                train.iq,
                fold.fold_id,
                "source_train",
                recording_ids=train.recording_ids,
            )
            train_features = _features(representation.transform(train.iq))
            validation_features = _features(representation.transform(validation.iq))
            probe = RidgeProbe().fit(train_features, train.labels)
            validation_probability = probe.predict_proba(validation_features)
            validation_metric = recording_level_metrics(
                validation.labels,
                validation.recording_ids,
                validation_probability,
                probe.classes_,
            )
            audit_count = min(6, len(train.iq))
            stability = source_instability(
                representation,
                train.iq[:audit_count],
                train.recording_ids[:audit_count],
                train.domains[:audit_count],
                default_smoke_cases(),
                seed + fold_index * 100,
            )
            cost = representation.measured_cost(train.iq, warmups=0, repeats=1)
            objective = ObjectiveRecord(
                stable_id=condition.stable_id,
                nuisance_sensitivity=float(stability["value"]),
                source_selection_macro_f1=float(validation_metric["macro_f1"]),
                batch1_latency_ms=cost["batch1_latency_ms"],
                bytes_per_sample=cost["bytes_per_sample"],
                estimated_macs=cost["estimated_macs"],
                feasible=True,
            )
            objectives.append(objective)
            artifact = representation.source_artifact()
            fitted[condition.stable_id] = (representation, probe)
            source_records.append(
                {
                    "stable_id": condition.stable_id,
                    "objective": asdict(objective),
                    "stability": stability,
                    "validation": validation_metric,
                    "artifact": artifact,
                    "probe_artifact": probe.source_artifact(),
                    "nuisance_seed": seed + fold_index * 100,
                }
            )
            results["costs"]["conditions"].setdefault(condition.stable_id, []).append(cost)
        selected, front = select_unique(objectives)
        freeze_payload = {
            "fold_id": fold.fold_id,
            "records": source_records,
            "pareto": front,
            "selected": selected,
            "split": split_record,
            "configuration_order": [condition.stable_id for condition in conditions],
            "h2_bank_status": "synthetic_subset_not_confirmatory_bank",
        }
        state.transition(RunPhase.SOURCE_FITTING_COMPLETE)
        state.source_artifact_hash = _artifact_hash(freeze_payload)
        state.transition(RunPhase.PARETO_FROZEN)
        results["source_fitting"]["folds"].append(
            {"fold_id": fold.fold_id, "records": source_records}
        )
        results["source_selection"]["folds"].append(
            {
                "fold_id": fold.fold_id,
                "pareto": front,
                "selected_theta_hat": selected,
                "source_artifact_hash": state.source_artifact_hash,
            }
        )
        source_cache.append((fold_index, fold, fitted, state, selected))
    results["splits"]["manifest_hash"] = _artifact_hash(results["splits"]["folds"])
    results["provenance"]["split_manifest_hash"] = results["splits"]["manifest_hash"]
    results["source_selection"]["policy"] = {
        "axes": ["nuisance_sensitivity", "source_selection_macro_f1", "bytes_per_sample", "estimated_macs", "batch1_latency_ms"],
        "source_f1_absolute_tolerance": 0.01,
        "target_metrics_used": False,
    }

    selected_scores = []
    for fold_index, fold, fitted, state, selected in source_cache:
        state.transition(RunPhase.TARGET_UNLOCKED)
        target = window_recordings(
            fold.held_target,
            512,
            512,
            access_role="held_target",
            state=state,
        )
        condition_results = {}
        for stable_id, (representation, probe) in fitted.items():
            features = _features(representation.transform(target.iq))
            probability = probe.predict_proba(features)
            condition_results[stable_id] = recording_level_metrics(
                target.labels, target.recording_ids, probability, probe.classes_
            )
        state.transition(RunPhase.TARGET_EVALUATED)
        state.transition(RunPhase.RESULTS_FINALIZED)
        selected_scores.append(condition_results[selected]["macro_f1"])
        results["held_domain"]["folds"].append(
            {
                "fold_id": fold.fold_id,
                "selected_theta_hat": selected,
                "conditions": condition_results,
                "target_reads": state.target_reads,
            }
        )
    results["held_domain"]["average_macro_f1"] = float(np.mean(selected_scores))
    results["held_domain"]["worst_macro_f1"] = float(np.min(selected_scores))
    results["held_domain"]["selected_average_macro_f1"] = float(np.mean(selected_scores))
    results["held_domain"]["selected_worst_macro_f1"] = float(np.min(selected_scores))
    all_windows = window_recordings(recordings, 512, 512)
    results["leakage_audits"]["duplicates"] = duplicate_audit(all_windows)
    results["warnings"] = [
        "SYNTHETIC DEVELOPMENT EVIDENCE ONLY; H1-H4 remain open.",
        "Smoke candidate set is intentionally smaller than the frozen real H2 bank.",
        "Smoke latency policy is reduced and is not deployment evidence.",
    ]
    results["run"]["elapsed_sec"] = float(perf_counter() - started)
    results_path = output_dir / "results.json"
    write_results(results_path, results)
    (output_dir / "manifest.json").write_text(
        json.dumps(results["provenance"], indent=2, sort_keys=True), encoding="utf-8"
    )
    return results_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SCARS synthetic engineering smoke test")
    parser.add_argument("--output-dir", type=Path, default=Path("results/smoke"))
    parser.add_argument("--seed", type=int, default=24021)
    arguments = parser.parse_args()
    print(run(arguments.output_dir, arguments.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
