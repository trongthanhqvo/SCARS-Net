from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from scars.baselines import (
    cwt_images,
    fixed_wavelet_subband_features,
    log_psd_sobel_images,
    log_stft_images,
    magnitude_phase,
    real_imag,
    stft_dct_features,
    torch_model_factory,
    load_h2_representation,
)
from scars.audits.degeneracy import representation_similarity
from scars.audits.leakage import duplicate_audit, near_duplicate_audit, split_group_leakage_audit
from scars.data.manifest import load_manifest
from scars.data.windowing import WindowBatch, window_recordings
from scars.evaluation.classification import calibration_metrics, recording_level_metrics
from scars.evaluation.classification import macro_f1
from scars.evaluation.costs import estimate_model_macs, model_parameter_count, synchronized_batch1_latency
from scars.evaluation.detection import detection_metrics, recording_binary_scores
from scars.evaluation.robustness import normalized_curve_area
from scars.experiment.common import atomic_json, assert_frozen_environment, load_split_plan
from scars.experiment.pretarget import verify_freeze_package, verify_source_campaign_pre_target
from scars.models.scars_net import SCARSNet
from scars.probes.ridge import RidgeProbe
from scars.probes.xgboost_probe import XGBoostProbe
from scars.representations.tensor import SourceFittedTensor
from scars.results.provenance import sha256_file
from scars.state import RunPhase, RunState
from scars.training.baselines import predict_classifier
from scars.training.engine import predict_scars, resolve_device
from scars.selection.nuisance import awgn, ism_mixing, registered_nuisance_cases


def _chunk_name(recording_id: str) -> str:
    return hashlib.sha256(recording_id.encode()).hexdigest()[:20] + ".npz"


def _save_chunk(path: Path, batch: WindowBatch) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            iq=batch.iq,
            labels=batch.labels.astype(str),
            recording_ids=batch.recording_ids.astype(str),
            domains=batch.domains.astype(str),
            starts=batch.starts,
            ends=batch.ends,
        )
    temporary.replace(path)
    return sha256_file(path)


def _load_target_cache(cache_dir: Path, manifest: dict[str, object]) -> WindowBatch:
    values = {key: [] for key in ("iq", "labels", "recording_ids", "domains", "starts", "ends")}
    for record in manifest["recordings"]:
        path = cache_dir / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError("Target cache chunk hash mismatch")
        with np.load(path, allow_pickle=False) as archive:
            for key in values:
                values[key].append(archive[key])
    return WindowBatch(**{key: np.concatenate(items) for key, items in values.items()})


def _validate_target_chunk(path: Path, recording, window_samples: int) -> None:
    with np.load(path, allow_pickle=False) as archive:
        required = {"iq", "labels", "recording_ids", "domains", "starts", "ends"}
        if set(archive.files) != required:
            raise RuntimeError("Target cache chunk fields differ from the frozen schema")
        lengths = {len(archive[key]) for key in required}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise RuntimeError("Target cache chunk has empty or misaligned arrays")
        if archive["iq"].ndim != 2 or archive["iq"].shape[1] != window_samples:
            raise RuntimeError("Target cache IQ window shape mismatch")
        if set(archive["recording_ids"].astype(str).tolist()) != {recording.recording_id}:
            raise RuntimeError("Target cache recording identity mismatch")
        if set(archive["labels"].astype(str).tolist()) != {str(recording.label)}:
            raise RuntimeError("Target cache label differs from the sealed manifest")
        starts = archive["starts"].astype(int)
        ends = archive["ends"].astype(int)
        if np.any(starts < 0) or np.any(ends - starts != window_samples):
            raise RuntimeError("Target cache window boundary contract failed")


def _combine_batches(batches: list[WindowBatch]) -> WindowBatch:
    return WindowBatch(
        **{
            key: np.concatenate([getattr(batch, key) for batch in batches])
            for key in ("iq", "labels", "recording_ids", "domains", "starts", "ends")
        }
    )


def _ensure_target_cache(
    fold,
    fold_dir: Path,
    state: RunState,
    window_samples: int,
    hop_samples: int,
    max_windows_per_recording: int,
) -> WindowBatch:
    cache_dir = fold_dir / "target-cache"
    cache_dir.mkdir(exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    if state.target_read_status == "complete":
        if not manifest_path.is_file() or sha256_file(manifest_path) != state.target_cache_manifest_sha256:
            raise RuntimeError("Committed target cache manifest is missing or changed")
        return _load_target_cache(
            cache_dir, json.loads(manifest_path.read_text(encoding="utf-8"))
        )
    target_ids = [record.recording_id for record in fold["held_target"]]
    state.begin_target_stream(target_ids)
    records = []
    for recording in fold["held_target"]:
        chunk = cache_dir / _chunk_name(recording.recording_id)
        if recording.recording_id in state.target_completed_recording_ids:
            if not chunk.is_file():
                raise RuntimeError("Target ledger says complete but its cache chunk is missing")
            _validate_target_chunk(chunk, recording, window_samples)
        elif chunk.is_file():
            # Never trust an unledgered pre-existing chunk. Re-read this
            # recording inside the already-open logical target stream and
            # atomically replace it before committing its identity.
            batch = window_recordings(
                [recording],
                window_samples,
                hop_samples,
                access_role="held_target_stream",
                state=state,
                max_windows_per_recording=max_windows_per_recording,
            )
            _save_chunk(chunk, batch)
            _validate_target_chunk(chunk, recording, window_samples)
            state.complete_target_recording(recording.recording_id)
        else:
            batch = window_recordings(
                [recording],
                window_samples,
                hop_samples,
                access_role="held_target_stream",
                state=state,
                max_windows_per_recording=max_windows_per_recording,
            )
            _save_chunk(chunk, batch)
            _validate_target_chunk(chunk, recording, window_samples)
            state.complete_target_recording(recording.recording_id)
        records.append(
            {
                "recording_id": recording.recording_id,
                "path": chunk.name,
                "sha256": sha256_file(chunk),
            }
        )
    manifest = {
        "schema_version": "scars-target-cache-1.0",
        "fold_id": fold["fold_id"],
        "recordings": records,
    }
    digest = atomic_json(manifest_path, manifest)
    if state.target_read_status == "in_progress":
        state.commit_target_stream(digest)
    elif state.target_cache_manifest_sha256 != digest:
        raise RuntimeError("Resumed target cache manifest differs from the committed ledger")
    return _load_target_cache(cache_dir, manifest)


def _standardize(values: np.ndarray, artifact: dict[str, object]) -> np.ndarray:
    mean = np.asarray(artifact["mean"], dtype=np.float64)
    scale = np.asarray(artifact["scale"], dtype=np.float64)
    return ((np.asarray(values) - mean) / scale).astype(np.float32)


def _record_metrics(batch, probability, classes, threshold):
    metrics = recording_level_metrics(batch.labels, batch.recording_ids, probability, classes)
    metrics["probability_class_order"] = [str(value) for value in classes]
    record_probability = np.asarray(metrics["recording_probabilities"], dtype=float)
    record_truth = np.asarray(metrics["recording_truth"], dtype=object)
    metrics["calibration"] = calibration_metrics(record_truth, record_probability, classes)
    if threshold is not None and "background" in classes.astype(str).tolist():
        y, scores, _ = recording_binary_scores(
            batch.labels, batch.recording_ids, probability, classes
        )
        metrics["detection"] = detection_metrics(y, scores, float(threshold["threshold"]))
        metrics["detection"]["source_threshold_provenance"] = threshold
    else:
        metrics["detection"] = {"status": "ineligible_no_canonical_background"}
    return metrics


def _ensemble_recording_macro_f1(seed_metrics: list[dict[str, object]]) -> float:
    reference = seed_metrics[0]
    classes = np.asarray(reference["probability_class_order"], dtype=object)
    truth = np.asarray(reference["recording_truth"], dtype=object)
    if any(
        metric["recording_order"] != reference["recording_order"]
        or metric["recording_truth"] != reference["recording_truth"]
        or metric["probability_class_order"] != reference["probability_class_order"]
        for metric in seed_metrics
    ):
        raise ValueError("Robustness seed records are not aligned")
    probability = np.mean(
        [np.asarray(metric["recording_probabilities"], dtype=float) for metric in seed_metrics],
        axis=0,
    )
    prediction = classes[np.argmax(probability, axis=1)]
    return macro_f1(truth, prediction)


def _torch_cost(model, input_array: np.ndarray, device: torch.device) -> dict[str, object]:
    example = torch.as_tensor(input_array[:1], dtype=torch.float32, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    latency = synchronized_batch1_latency(
        model, example, device=device, warmups=20, repeats=100
    )
    return {
        "parameters": model_parameter_count(model),
        "macs": estimate_model_macs(model, example),
        "bytes_per_sample": int(np.asarray(input_array[0]).nbytes),
        "latency": latency,
        "peak_inference_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
    }


def _with_representation_cost(
    model_cost: dict[str, object], representation_cost: dict[str, object]
) -> dict[str, object]:
    output = dict(model_cost)
    output["model_macs"] = float(model_cost["macs"])
    output["representation_macs"] = float(representation_cost["estimated_macs"])
    model_latency = model_cost["latency"]
    representation_latency = float(representation_cost["batch1_latency_ms"])
    output["macs"] = float(model_cost["macs"]) + float(representation_cost["estimated_macs"])
    output["bytes_per_sample"] = float(representation_cost["bytes_per_sample"])
    output["latency"] = {
        **model_latency,
        "median_ms": float(model_latency["median_ms"]) + representation_latency,
        "measurement_valid": bool(
            model_latency.get("measurement_valid", False)
            and representation_cost.get("measurement_valid", False)
        ),
        "scope": "representation_plus_model_stage_sum_proxy",
        "model_median_ms": float(model_latency["median_ms"]),
        "representation_median_ms": representation_latency,
        "representation_device": representation_cost.get("device", "cpu"),
    }
    output["cost_breakdown"] = {
        "representation": representation_cost,
        "model": model_cost,
        "stage_sum_latency_rule": "sum_of_independently_measured_batch1_medians",
        "excluded_from_stage_sum": ["host_to_device_transfer", "orchestration", "file_io"],
    }
    return output


def _cpu_predict_cost(
    callable_predict,
    one_feature: np.ndarray,
    *,
    parameters: int,
    macs: int,
) -> dict[str, object]:
    for _ in range(20):
        callable_predict(one_feature)
    samples = []
    for _ in range(100):
        started = perf_counter()
        callable_predict(one_feature)
        samples.append(1000.0 * (perf_counter() - started))
    first_half = float(np.median(samples[:50]))
    second_half = float(np.median(samples[50:]))
    drift_ratio = max(first_half, second_half) / max(min(first_half, second_half), 1.0e-12)
    return {
        "parameters": int(parameters),
        "macs": int(macs),
        "bytes_per_sample": int(np.asarray(one_feature[0]).nbytes),
        "latency": {
            "median_ms": float(np.median(samples)),
            "samples_ms": samples,
            "warmups": 20,
            "repeats": 100,
            "batch_size": 1,
            "synchronized": True,
            "device": "cpu",
            "timing_stability": {
                "first_half_median_ms": first_half,
                "second_half_median_ms": second_half,
                "drift_ratio": drift_ratio,
                "drift_valid": drift_ratio <= 1.25,
            },
            "measurement_valid": bool(np.all(np.isfinite(samples)) and drift_ratio <= 1.25),
        },
    }


def _measured_preprocessing_cost(callable_transform, one_iq: np.ndarray, estimated_macs: int):
    for _ in range(20):
        output = callable_transform(one_iq)
    samples = []
    for _ in range(100):
        started = perf_counter()
        output = callable_transform(one_iq)
        samples.append(1000.0 * (perf_counter() - started))
    first = float(np.median(samples[:50]))
    second = float(np.median(samples[50:]))
    drift = max(first, second) / max(min(first, second), 1.0e-12)
    return {
        "bytes_per_sample": int(np.asarray(output[0]).nbytes),
        "batch1_latency_ms": float(np.median(samples)),
        "estimated_macs": int(estimated_macs),
        "warmups": 20,
        "repeats": 100,
        "measurement_valid": bool(np.all(np.isfinite(samples)) and drift <= 1.25),
        "device": "cpu",
        "timing_stability": {
            "first_half_median_ms": first,
            "second_half_median_ms": second,
            "drift_ratio": drift,
            "drift_valid": drift <= 1.25,
        },
        "mac_estimator_version": "scars-baseline-preprocessing-v1",
    }


def _baseline_preprocessing_cost(condition, target, preprocessing, fold_dir):
    n = int(target.iq.shape[1])
    frames = max(1, 1 + max(0, n - 128) // 32)
    fft_macs = int(frames * 5 * 128 * np.log2(128))
    if condition == "raw_iq_cnn":
        return _measured_preprocessing_cost(
            lambda iq: _standardize(real_imag(iq), preprocessing), target.iq[:1], 0
        )
    if condition == "magnitude_phase_cnn":
        return _measured_preprocessing_cost(
            lambda iq: _standardize(magnitude_phase(iq), preprocessing), target.iq[:1], 8 * n
        )
    if condition in {"log_stft_cnn", "patch_transformer"}:
        return _measured_preprocessing_cost(
            lambda iq: _standardize(log_stft_images(iq, 16), preprocessing),
            target.iq[:1],
            fft_macs + 16 * 16 * 8,
        )
    if condition == "cwt_cnn":
        return _measured_preprocessing_cost(
            lambda iq: _standardize(cwt_images(iq, 16), preprocessing),
            target.iq[:1],
            16 * 10 * n * int(np.log2(max(n, 2))),
        )
    if condition == "log_psd_sobel":
        return _measured_preprocessing_cost(
            lambda iq: _standardize(log_psd_sobel_images(iq, 16), preprocessing),
            target.iq[:1],
            fft_macs + 18 * 16 * 16,
        )
    raise KeyError(condition)


def _aggregate_seed_costs(costs: list[dict[str, object]]) -> dict[str, object]:
    if not costs:
        raise ValueError("No seed-level cost measurements")
    model_macs = [float(item.get("model_macs", item["macs"])) for item in costs]
    representation_macs = [float(item.get("representation_macs", 0.0)) for item in costs]
    model_latency = [
        float(item["latency"].get("model_median_ms", item["latency"]["median_ms"]))
        for item in costs
    ]
    representation_latency = [
        float(item["latency"].get("representation_median_ms", 0.0)) for item in costs
    ]
    shared_representation_macs = float(np.mean(representation_macs))
    shared_representation_latency = float(np.mean(representation_latency))
    deployed_model_macs = float(np.sum(model_macs))
    deployed_model_latency = float(np.sum(model_latency))
    return {
        "parameters": float(np.sum([item["parameters"] for item in costs])),
        "macs": deployed_model_macs + shared_representation_macs,
        "bytes_per_sample": float(np.mean([item["bytes_per_sample"] for item in costs])),
        "latency": {
            "median_ms": deployed_model_latency + shared_representation_latency,
            "measurement_valid": bool(
                all(item["latency"].get("measurement_valid", False) for item in costs)
            ),
            "device": costs[0]["latency"]["device"],
            "warmups": 20,
            "repeats": 100,
            "aggregation": "sequential_deployed_probability_ensemble_models_plus_one_shared_representation",
            "per_seed": [item["latency"] for item in costs],
            "model_median_ms": deployed_model_latency,
            "representation_median_ms": shared_representation_latency,
        },
        "per_seed": costs,
        "ensemble_size": len(costs),
        "ensemble_semantics": "mean_recording_probability_vector",
        "model_macs": deployed_model_macs,
        "representation_macs": shared_representation_macs,
        "peak_inference_memory_bytes": (
            float(
                np.max(
                    [
                        item["peak_inference_memory_bytes"]
                        for item in costs
                        if item.get("peak_inference_memory_bytes") is not None
                    ]
                )
            )
            if any(item.get("peak_inference_memory_bytes") is not None for item in costs)
            else None
        ),
    }


def _condition_complete(value: dict[str, object] | None, expected_seeds: int) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("cost"), dict):
        return False
    seeds = value.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != expected_seeds:
        return False
    for row in seeds:
        metric = row.get("metrics", {})
        required = {
            "recording_order",
            "recording_truth",
            "recording_probabilities",
            "probability_class_order",
            "replication_unit",
        }
        if not required.issubset(metric) or metric.get("replication_unit") != "recording":
            return False
        if len(metric["recording_order"]) != len(metric["recording_probabilities"]):
            return False
        if expected_seeds > 1 and not isinstance(row.get("cost"), dict):
            return False
    if expected_seeds > 1 and len(value["cost"].get("per_seed", [])) != expected_seeds:
        return False
    return True


def _load_torch(path: Path, condition: str, classes, active):
    payload = torch.load(path, map_location="cpu")
    if condition in {"pcrd", "ordinary_gate", "shuffled_pcrd"} or condition.startswith("active_minus_"):
        model = SCARSNet(active, len(classes))
    else:
        early_width = int(payload.get("early_fusion_width", 32))
        architecture_id = (
            "early_fusion_resnet"
            if condition == "early_fusion_resnet_mac_matched"
            else condition
        )
        model = torch_model_factory(
            architecture_id,
            class_count=len(classes),
            active_families=active,
            early_fusion_width=early_width,
        )()
    model.load_state_dict(payload["state_dict"])
    return model, payload


def _baseline_input(condition: str, batch: WindowBatch, preprocessing, fold_dir: Path):
    if condition == "raw_iq_cnn":
        return _standardize(real_imag(batch.iq), preprocessing)
    if condition == "magnitude_phase_cnn":
        return _standardize(magnitude_phase(batch.iq), preprocessing)
    if condition in {"log_stft_cnn", "patch_transformer"}:
        return _standardize(log_stft_images(batch.iq, 16), preprocessing)
    if condition == "cwt_cnn":
        return _standardize(cwt_images(batch.iq, 16), preprocessing)
    if condition == "log_psd_sobel":
        return _standardize(log_psd_sobel_images(batch.iq, 16), preprocessing)
    if condition in {"wst_only", "cyclic_only"}:
        path = fold_dir / preprocessing["representation_artifact"]
        if sha256_file(path) != preprocessing["representation_sha256"]:
            raise RuntimeError("Baseline representation hash mismatch")
        representation = SourceFittedTensor.from_source_artifact(
            json.loads(path.read_text(encoding="utf-8"))
        )
        return representation.transform(batch.iq)
    raise KeyError(condition)


def run(
    *,
    preflight_dir: Path,
    source_campaign_dir: Path,
    freeze_package: Path | None,
    authorize_target: bool,
    device: str,
    window_samples: int,
    hop_samples: int,
    max_windows_per_recording: int,
    pilot_two_dataset: bool = False,
    pilot_three_dataset: bool = False,
) -> Path:
    if pilot_two_dataset and pilot_three_dataset:
        raise ValueError("Select only one pilot mode")
    if not authorize_target:
        raise PermissionError("Refused: --authorize-target is required")
    campaign_path = source_campaign_dir / "source_campaign.json"
    if pilot_two_dataset or pilot_three_dataset:
        campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        preflight = json.loads((preflight_dir / "preflight.json").read_text(encoding="utf-8"))
        expected_mode = "pilot_three_dataset" if pilot_three_dataset else "pilot_two_dataset"
        if preflight.get("campaign_mode") != expected_mode:
            raise PermissionError("Pilot flag does not match the frozen preflight mode")
        expected_datasets = {"DroneRFa", "DroneRFb-DIR", "DRFF-R2"} if pilot_three_dataset else {"DroneRFa", "DroneRFb-DIR"}
        if set(preflight.get("datasets", [])) != expected_datasets:
            raise PermissionError("Pilot dataset set does not match its explicit authorization mode")
        if (source_campaign_dir / "target_campaign.json").exists():
            raise PermissionError("A pilot target campaign artifact already exists")
    else:
        if freeze_package is None:
            raise PermissionError("Confirmatory target evaluation requires --freeze-package")
        freeze_manifest = verify_freeze_package(freeze_package)
        ledger = json.loads((freeze_package / freeze_manifest["target_access_ledger"]).read_text(encoding="utf-8"))
        if freeze_manifest.get("status") != "ready" or not all(freeze_manifest.get("gates", {}).values()):
            raise PermissionError("Refused: pre-target freeze package is not READY")
        if ledger.get("target_access_count") != 0 or ledger.get("target_performance_inspected") is not False:
            raise PermissionError("Refused: target-access ledger is not pristine")
        campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        preflight = json.loads((preflight_dir / "preflight.json").read_text(encoding="utf-8"))
        if sha256_file(campaign_path) != freeze_manifest.get("source_campaign_sha256"):
            raise RuntimeError("Source campaign differs from the authorized pre-target freeze")
        if freeze_manifest.get("source_tree_sha256") != campaign.get("provenance", {}).get("source_tree_sha256"):
            raise RuntimeError("Freeze-package source tree differs from the source campaign")
        verification = verify_source_campaign_pre_target(
            source_campaign_dir,
            expected_label_contract_sha256=(freeze_manifest.get("source_verification") or {}).get(
                "label_contract_sha256"
            ),
        )
        if verification["campaign_sha256"] != freeze_manifest.get("source_campaign_sha256"):
            raise RuntimeError("Verified source campaign differs from freeze authorization")
    if campaign.get("status") != "all_source_models_frozen":
        raise PermissionError("Every source model fold must be frozen before target unlock")
    project_root = Path(__file__).resolve().parents[3]
    assert_frozen_environment(campaign, project_root)
    if preflight.get("status") != "ready":
        raise PermissionError("Preflight is blocked")
    if sha256_file(preflight_dir / "preflight.json") != campaign["input_hashes"]["preflight"]:
        raise RuntimeError("Target preflight differs from the source-freeze preflight")
    requested_windowing = {
        "window_samples": window_samples,
        "hop_samples": hop_samples,
        "max_windows_per_recording": max_windows_per_recording,
    }
    if requested_windowing != campaign.get("windowing"):
        raise ValueError("Target windowing differs from the frozen source campaign")
    recordings = load_manifest(preflight_dir / preflight["artifacts"]["recordings_manifest"])
    folds = load_split_plan(preflight_dir / preflight["artifacts"]["split_manifest"], recordings)
    resolved = resolve_device(device)
    all_fold_results = []
    for fold, campaign_fold, freeze_record in zip(
        folds, campaign["folds"], campaign["model_freezes"]
    ):
        fold_dir = source_campaign_dir / campaign_fold["directory"]
        freeze_path = source_campaign_dir / freeze_record["path"]
        if sha256_file(freeze_path) != freeze_record["sha256"]:
            raise RuntimeError("Model-freeze hash mismatch")
        model_freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        state = RunState(fold_dir / "run_state.json")
        if state.phase == RunPhase.PARETO_FROZEN:
            state.transition(RunPhase.TARGET_UNLOCKED)
        elif state.phase not in {RunPhase.TARGET_UNLOCKED, RunPhase.TARGET_EVALUATED}:
            raise RuntimeError("Fold is not at a target-safe phase")
        target = _ensure_target_cache(
            fold,
            fold_dir,
            state,
            window_samples,
            hop_samples,
            max_windows_per_recording,
        )
        result_path = fold_dir / "target_metrics.json"
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.is_file()
            else {
                "schema_version": "scars-target-fold-2.0",
                "fold_id": fold["fold_id"],
                "target_recording_ids": [record.recording_id for record in fold["held_target"]],
                "target_reads": 1,
                "model_freeze_sha256": freeze_record["sha256"],
                "source_freeze_sha256": campaign_fold["source_freeze_sha256"],
                "target_cache_manifest_sha256": state.target_cache_manifest_sha256,
                "conditions": {},
                "candidate_bank": {},
                "h2_bank": {},
                "ablation_bank": {},
            }
        )
        result.setdefault("candidate_bank", {})
        if (
            result.get("schema_version") != "scars-target-fold-2.0"
            or result.get("fold_id") != fold["fold_id"]
            or result.get("target_recording_ids")
            != [record.recording_id for record in fold["held_target"]]
            or result.get("model_freeze_sha256") != freeze_record["sha256"]
            or result.get("source_freeze_sha256") != campaign_fold["source_freeze_sha256"]
            or result.get("target_cache_manifest_sha256")
            != state.target_cache_manifest_sha256
        ):
            raise RuntimeError("Existing target result has an incompatible semantic identity")
        if "integrity" not in result:
            source_audit_batches = []
            for role in (
                "source_fit",
                "source_calibration",
                "source_selection",
                "source_validation",
            ):
                role_batch = window_recordings(
                    fold[role],
                    window_samples,
                    hop_samples,
                    max_windows_per_recording=max_windows_per_recording,
                )
                source_audit_batches.append(
                    WindowBatch(
                        iq=role_batch.iq,
                        labels=role_batch.labels,
                        recording_ids=role_batch.recording_ids,
                        domains=np.full(len(role_batch.iq), role, dtype=object),
                        starts=role_batch.starts,
                        ends=role_batch.ends,
                    )
                )
            target_audit = WindowBatch(
                iq=target.iq,
                labels=target.labels,
                recording_ids=target.recording_ids,
                domains=np.full(len(target.iq), "held_target", dtype=object),
                starts=target.starts,
                ends=target.ends,
            )
            combined_audit = _combine_batches([*source_audit_batches, target_audit])
            split_record = {
                "fold_id": fold["fold_id"],
                **{
                    role: [record.recording_id for record in fold[role]]
                    for role in (
                        "source_fit",
                        "source_calibration",
                        "source_selection",
                        "source_validation",
                        "held_target",
                    )
                },
            }
            group_roles = {
                role: {record.group_id for record in fold[role]}
                for role in (
                    "source_fit",
                    "source_calibration",
                    "source_selection",
                    "source_validation",
                    "held_target",
                )
            }
            role_names = tuple(group_roles)
            group_violations = [
                {"left_role": left, "right_role": right, "group_id": group}
                for left_index, left in enumerate(role_names)
                for right in role_names[left_index + 1 :]
                for group in sorted(group_roles[left] & group_roles[right])
            ]
            temporal_grouping_proof = all(
                bool(
                    record.metadata.get(
                        "temporal_adjacency_subsumed_by_split_group_verified"
                    )
                )
                for role in (
                    "source_fit",
                    "source_calibration",
                    "source_selection",
                    "source_validation",
                    "held_target",
                )
                for record in fold[role]
            )
            result["integrity"] = {
                "duplicates": duplicate_audit(combined_audit),
                "near_duplicates": near_duplicate_audit(combined_audit),
                "group_overlap": split_group_leakage_audit([split_record]),
                "temporal_group_adjacency": {
                    "status": "passed"
                    if temporal_grouping_proof and not group_violations
                    else "failed",
                    "method": "manifest_verified_temporal_adjacency_subsumed_by_physical_group_then_cross_role_group_intersection",
                    "manifest_temporal_grouping_proof": temporal_grouping_proof,
                    "cross_partition_group_violations": group_violations,
                    "roles": {
                        role: sorted({record.group_id for record in fold[role]})
                        for role in (
                            "source_fit",
                            "source_calibration",
                            "source_selection",
                            "source_validation",
                            "held_target",
                        )
                    },
                },
                "target_firewall": {
                    "target_reads": state.target_reads,
                    "target_unlocks": state.target_unlocks,
                    "cache_manifest_sha256": state.target_cache_manifest_sha256,
                },
            }
            atomic_json(result_path, result)
        if model_freeze["status"] != "complete":
            result["status"] = model_freeze["status"]
            atomic_json(result_path, result)
            all_fold_results.append(result)
            continue
        active = tuple(model_freeze["active_families"])
        active_indices = list(model_freeze["active_indices"])
        classes = np.asarray(model_freeze["classes"])
        source_freeze = json.loads((fold_dir / "source_freeze.json").read_text(encoding="utf-8"))
        selected_record = source_freeze["canonical_tensor_artifact"]
        if sha256_file(fold_dir / selected_record["path"]) != selected_record["sha256"]:
            raise RuntimeError("Canonical WCES representation artifact hash mismatch")
        selected_representation = SourceFittedTensor.from_source_artifact(
            json.loads((fold_dir / selected_record["path"]).read_text(encoding="utf-8"))
        )
        selected_representation_cost = selected_record["cost"]
        selected_tensor = selected_representation.transform(target.iq)[:, active_indices]

        for candidate_id in source_freeze["candidate_order"]:
            if candidate_id in result["candidate_bank"]:
                cached_metric = result["candidate_bank"][candidate_id].get("metrics", {})
                if {
                    "recording_order",
                    "recording_truth",
                    "recording_probabilities",
                    "probability_class_order",
                    "replication_unit",
                }.issubset(cached_metric):
                    continue
                result["candidate_bank"].pop(candidate_id)
            candidate_record = source_freeze["candidates"][candidate_id]
            artifact_path = fold_dir / candidate_record["path"]
            if sha256_file(artifact_path) != candidate_record["sha256"]:
                raise RuntimeError("H2 representation artifact hash mismatch")
            candidate_representation = SourceFittedTensor.from_source_artifact(
                json.loads(artifact_path.read_text(encoding="utf-8"))
            )
            candidate_tensor = candidate_representation.transform(target.iq)
            probe = RidgeProbe.from_source_artifact(candidate_record["probe"])
            probability = probe.predict_proba(candidate_tensor.reshape(len(candidate_tensor), -1))
            candidate_metrics = recording_level_metrics(
                    target.labels,
                    target.recording_ids,
                    probability,
                    probe.classes_,
                )
            candidate_metrics["probability_class_order"] = probe.classes_.astype(str).tolist()
            result["candidate_bank"][candidate_id] = {
                "metrics": candidate_metrics,
                "cost": candidate_record["cost"],
                "representation_similarity": representation_similarity(
                    selected_tensor, candidate_tensor
                ),
                "domain_probe_accuracy": candidate_record["domain_probe_accuracy"],
            }
            atomic_json(result_path, result)
        for h2_id in source_freeze["h2_configuration_order"]:
            h2_record = source_freeze["h2_candidates"][h2_id]
            h2_path = fold_dir / h2_record["path"]
            if sha256_file(h2_path) != h2_record["sha256"]:
                raise RuntimeError("Frozen H2 representation artifact hash mismatch")
            h2_representation = load_h2_representation(
                json.loads(h2_path.read_text(encoding="utf-8"))
            )
            h2_tensor = h2_representation.transform(target.iq)
            h2_probe = RidgeProbe.from_source_artifact(h2_record["probe"])
            h2_probability = h2_probe.predict_proba(h2_tensor.reshape(len(h2_tensor), -1))
            h2_metrics = recording_level_metrics(
                target.labels,
                target.recording_ids,
                h2_probability,
                h2_probe.classes_,
            )
            h2_metrics["probability_class_order"] = h2_probe.classes_.astype(str).tolist()
            result["h2_bank"][h2_id] = {
                "metrics": h2_metrics,
                "cost": h2_record["cost"],
                "effective_signature": h2_record["effective_signature"],
            }
            atomic_json(result_path, result)
        ablation_candidate_map = {
            "W": "W",
            "C": "C",
            "W+C": "W+C",
            "W+C+E": "W+C+E",
            "W+C+E+S": "W+C+E+S",
            "wst_J3Q1": "wst_J3Q1",
            "wst_J3Q2": "wst_J3Q2",
            "wst_J3Q4": "wst_J3Q4",
            "wst_J4Q1": "wst_J4Q1",
            "wst_J4Q2": "wst_J4Q2",
            "wst_J4Q4": "wst_J4Q4",
            "wst_J5Q1": "wst_J5Q1",
            "wst_J5Q2": "wst_J5Q2",
            "wst_J5Q4": "wst_J5Q4",
            "cyclic_A4": "cyclic_A4",
            "cyclic_A8": "cyclic_A8",
            "cyclic_A16": "cyclic_A16",
            "cyclic_permuted": "cyclic_permuted",
            "norm_percentile": "norm_percentile",
            "norm_zscore": "norm_zscore",
            "norm_none": "norm_none",
            "resolution_8": "resolution_8",
            "resolution_16": "resolution_16",
            "resolution_32": "resolution_32",
            "selection_scalar": source_freeze["scalar_selected_stable_id"],
            "selection_pareto": source_freeze["selected_stable_id"],
        }
        result["ablation_bank"] = {
            condition: {
                "candidate_id": candidate,
                **result["candidate_bank"][candidate],
            }
            for condition, candidate in ablation_candidate_map.items()
        }
        atomic_json(result_path, result)

        for condition in ("ordinary_gate", "shuffled_pcrd", "pcrd"):
            condition_result = result["conditions"].setdefault(condition, {"seeds": []})
            if _condition_complete(
                condition_result, len(model_freeze["models"][condition])
            ):
                continue
            condition_result["seeds"] = []
            seed_costs = []
            for record in model_freeze["models"][condition]:
                checkpoint = fold_dir / record["path"]
                if sha256_file(checkpoint) != record["sha256"]:
                    raise RuntimeError("SCARS-Net checkpoint hash mismatch")
                model, _ = _load_torch(checkpoint, condition, classes, active)
                model.to(resolved)
                probability, gate_weights = predict_scars(
                    model, selected_tensor, device=resolved, batch_size=record["report"]["batch_size"]
                )
                model_cost = _with_representation_cost(
                    _torch_cost(model, selected_tensor, resolved),
                    selected_representation_cost,
                )
                seed_costs.append(model_cost)
                condition_result["seeds"].append(
                    {
                        "seed": record["report"]["seed"],
                        "metrics": _record_metrics(
                            target, probability, classes, record.get("source_detection_threshold")
                        ),
                        "mean_gate_weights": np.mean(gate_weights, axis=0).tolist(),
                        "cost": model_cost,
                    }
                )
                del model
                if resolved.type == "cuda":
                    torch.cuda.empty_cache()
                atomic_json(result_path, result)
            condition_result["cost"] = _aggregate_seed_costs(seed_costs)

        for condition, ablation in model_freeze.get("deletion_ablations", {}).items():
            if _condition_complete(
                result["conditions"].get(condition), len(ablation.get("models", []))
            ):
                continue
            if ablation["status"] != "complete":
                result["conditions"][condition] = {
                    "status": ablation["status"], "seeds": []
                }
                continue
            retained = tuple(ablation["active_families"])
            input_array = selected_tensor[:, ablation["retained_indices"]]
            seed_rows = []
            ablation_costs = []
            for record in ablation["models"]:
                checkpoint = fold_dir / record["path"]
                if sha256_file(checkpoint) != record["sha256"]:
                    raise RuntimeError("Deletion-ablation checkpoint hash mismatch")
                model, _ = _load_torch(checkpoint, condition, classes, retained)
                model.to(resolved)
                probability, gate_weights = predict_scars(
                    model,
                    input_array,
                    device=resolved,
                    batch_size=record["report"]["batch_size"],
                )
                model_cost = _with_representation_cost(
                    _torch_cost(model, input_array, resolved),
                    selected_representation_cost,
                )
                ablation_costs.append(model_cost)
                seed_rows.append(
                    {
                        "seed": record["report"]["seed"],
                        "metrics": _record_metrics(target, probability, classes, None),
                        "mean_gate_weights": np.mean(gate_weights, axis=0).tolist(),
                        "cost": model_cost,
                    }
                )
                del model
                if resolved.type == "cuda":
                    torch.cuda.empty_cache()
            result["conditions"][condition] = {
                "status": "complete",
                "deleted_family": ablation["deleted_family"],
                "seeds": seed_rows,
                "cost": _aggregate_seed_costs(ablation_costs),
                "representation_similarity": representation_similarity(
                    selected_tensor,
                    np.where(
                        np.arange(selected_tensor.shape[1])[None, :, None, None]
                        == active.index(ablation["deleted_family"]),
                        0.0,
                        selected_tensor,
                    ),
                ),
            }
            atomic_json(result_path, result)

        for condition, baseline in model_freeze["baselines"].items():
            expected_baseline_seeds = 1 if baseline["trainer"] == "ridge" else len(
                baseline["models"]
            )
            if _condition_complete(
                result["conditions"].get(condition), expected_baseline_seeds
            ):
                continue
            if baseline["trainer"] == "ridge":
                if sha256_file(fold_dir / baseline["path"]) != baseline["sha256"]:
                    raise RuntimeError("Fixed-wavelet probe hash mismatch")
                probe = RidgeProbe.from_source_artifact(
                    json.loads((fold_dir / baseline["path"]).read_text(encoding="utf-8"))
                )
                probability = probe.predict_proba(fixed_wavelet_subband_features(target.iq))
                preprocessing_cost = _measured_preprocessing_cost(
                    fixed_wavelet_subband_features,
                    target.iq[:1],
                    int(8 * target.iq.shape[1]),
                )
                result["conditions"][condition] = {
                    "seeds": [{"seed": None, "metrics": _record_metrics(target, probability, probe.classes_, baseline.get("source_detection_threshold"))}],
                    "cost": _with_representation_cost(
                        _cpu_predict_cost(
                            probe.predict_proba,
                            fixed_wavelet_subband_features(target.iq[:1]),
                            parameters=int(probe.weights_.size),
                            macs=int(probe.weights_.shape[0] * probe.weights_.shape[1]),
                        ),
                        preprocessing_cost,
                    ),
                }
            elif baseline["trainer"] == "xgboost":
                seed_rows = []
                xgb_costs = []
                features = stft_dct_features(target.iq)
                xgb_preprocessing_cost = _measured_preprocessing_cost(
                    stft_dct_features,
                    target.iq[:1],
                    int(5 * target.iq.shape[1] * np.log2(max(target.iq.shape[1], 2)) + 64 * 16 * 16),
                )
                for seed, record in zip((11, 23, 37, 53, 71), baseline["models"]):
                    if sha256_file(fold_dir / record["path"]) != record["sha256"]:
                        raise RuntimeError("XGBoost model hash mismatch")
                    classes_path = fold_dir / record["classes_path"]
                    if sha256_file(classes_path) != record["classes_sha256"]:
                        raise RuntimeError("XGBoost class-order hash mismatch")
                    probe = XGBoostProbe.load_model(fold_dir / record["path"], seed)
                    probability = probe.predict_proba(features)
                    xgb_counted = probe.counted_cost()
                    model_cost = _with_representation_cost(
                        _cpu_predict_cost(
                            probe.predict_proba,
                            features[:1],
                            parameters=xgb_counted["parameters"],
                            macs=xgb_counted["macs"],
                        ),
                        xgb_preprocessing_cost,
                    )
                    xgb_costs.append(model_cost)
                    seed_rows.append(
                        {"seed": seed, "metrics": _record_metrics(target, probability, probe.classes_, record.get("source_detection_threshold")), "cost": model_cost}
                    )
                result["conditions"][condition] = {
                    "seeds": seed_rows,
                    "cost": _aggregate_seed_costs(xgb_costs),
                }
            else:
                input_array = (
                    selected_tensor
                    if condition in {"early_fusion_resnet", "early_fusion_resnet_mac_matched", "uniform_late_fusion"}
                    else _baseline_input(condition, target, baseline["preprocessing"], fold_dir)
                )
                seed_rows = []
                baseline_costs = []
                preprocessing_cost = None
                if condition not in {
                    "early_fusion_resnet",
                    "early_fusion_resnet_mac_matched",
                    "uniform_late_fusion",
                    "wst_only",
                    "cyclic_only",
                }:
                    preprocessing_cost = _baseline_preprocessing_cost(
                        condition, target, baseline["preprocessing"], fold_dir
                    )
                for record in baseline["models"]:
                    checkpoint = fold_dir / record["path"]
                    if sha256_file(checkpoint) != record["sha256"]:
                        raise RuntimeError("Learned-baseline checkpoint hash mismatch")
                    model, payload = _load_torch(checkpoint, condition, classes, active)
                    model.to(resolved)
                    probability = predict_classifier(
                        model, input_array, device=resolved, batch_size=record["report"]["batch_size"]
                    )
                    model_cost = _torch_cost(model, input_array, resolved)
                    if condition in {
                        "early_fusion_resnet",
                        "early_fusion_resnet_mac_matched",
                        "uniform_late_fusion",
                    }:
                        model_cost = _with_representation_cost(
                            model_cost, selected_representation_cost
                        )
                    elif condition in {"wst_only", "cyclic_only"}:
                        candidate_id = "W" if condition == "wst_only" else "C"
                        model_cost = _with_representation_cost(
                            model_cost, source_freeze["candidates"][candidate_id]["cost"]
                        )
                    elif preprocessing_cost is not None:
                        model_cost = _with_representation_cost(
                            model_cost, preprocessing_cost
                        )
                    baseline_costs.append(model_cost)
                    seed_rows.append(
                        {"seed": record["report"]["seed"], "metrics": _record_metrics(target, probability, classes, record.get("source_detection_threshold")), "cost": model_cost}
                    )
                    del model
                    if resolved.type == "cuda":
                        torch.cuda.empty_cache()
                result["conditions"][condition] = {"seeds": seed_rows, "cost": _aggregate_seed_costs(baseline_costs)}
            atomic_json(result_path, result)
        if "robustness" not in result:
            robustness = {}
            grids = {"snr": ([20.0, 10.0, 0.0, -10.0], awgn), "sir": ([20.0, 10.0, 0.0, -10.0], ism_mixing)}
            for condition in ("pcrd", "W", "C", "STFT"):
                condition_curves = {}
                for axis, (grid, transform) in grids.items():
                    values = []
                    for severity_index, severity in enumerate(grid):
                        perturbed_iq = np.asarray(
                            [
                                transform(
                                    window,
                                    np.random.default_rng(
                                        26000 + 101 * severity_index + window_index
                                    ),
                                    severity,
                                )
                                for window_index, window in enumerate(target.iq)
                            ],
                            dtype=np.complex64,
                        )
                        if condition == "pcrd":
                            perturbed_tensor = selected_representation.transform(perturbed_iq)[:, active_indices]
                            seed_metrics = []
                            for record in model_freeze["models"]["pcrd"]:
                                if sha256_file(fold_dir / record["path"]) != record["sha256"]:
                                    raise RuntimeError("PCRD robustness checkpoint hash mismatch")
                                model, _ = _load_torch(
                                    fold_dir / record["path"], "pcrd", classes, active
                                )
                                model.to(resolved)
                                probability, _ = predict_scars(
                                    model,
                                    perturbed_tensor,
                                    device=resolved,
                                    batch_size=record["report"]["batch_size"],
                                )
                                robustness_metric = recording_level_metrics(
                                        target.labels,
                                        target.recording_ids,
                                        probability,
                                        classes,
                                    )
                                robustness_metric["probability_class_order"] = classes.astype(str).tolist()
                                seed_metrics.append(robustness_metric)
                                del model
                            values.append(_ensemble_recording_macro_f1(seed_metrics))
                        else:
                            candidate_record = source_freeze["candidates"][condition]
                            candidate_representation = SourceFittedTensor.from_source_artifact(
                                json.loads(
                                    (fold_dir / candidate_record["path"]).read_text(
                                        encoding="utf-8"
                                    )
                                )
                            )
                            candidate_tensor = candidate_representation.transform(perturbed_iq)
                            probe = RidgeProbe.from_source_artifact(candidate_record["probe"])
                            probability = probe.predict_proba(
                                candidate_tensor.reshape(len(candidate_tensor), -1)
                            )
                            robustness_metric = recording_level_metrics(
                                    target.labels, target.recording_ids, probability, probe.classes_
                                )
                            robustness_metric["probability_class_order"] = probe.classes_.astype(str).tolist()
                            seed_metrics = [robustness_metric]
                            values.append(seed_metrics[0]["macro_f1"])
                        condition_curves.setdefault(axis + "_recording_metrics", []).append(seed_metrics)
                    condition_curves[axis] = {
                        "grid_db": grid,
                        "macro_f1": values,
                        "normalized_auc": normalized_curve_area(grid, values),
                        "recording_metrics": condition_curves.pop(axis + "_recording_metrics"),
                    }
                nuisance_records = []
                for case_index, case in enumerate(registered_nuisance_cases()):
                    perturbed_iq = np.asarray(
                        [
                            case.transform(
                                window,
                                np.random.default_rng(36000 + 1009 * case_index + window_index),
                                case.severity,
                            )
                            for window_index, window in enumerate(target.iq)
                        ],
                        dtype=np.complex64,
                    )
                    if condition == "pcrd":
                        perturbed_tensor = selected_representation.transform(perturbed_iq)[:, active_indices]
                        seed_metrics = []
                        for record in model_freeze["models"]["pcrd"]:
                            model, _ = _load_torch(
                                fold_dir / record["path"], "pcrd", classes, active
                            )
                            model.to(resolved)
                            probability, _ = predict_scars(
                                model,
                                perturbed_tensor,
                                device=resolved,
                                batch_size=record["report"]["batch_size"],
                            )
                            metric = recording_level_metrics(
                                target.labels, target.recording_ids, probability, classes
                            )
                            metric["probability_class_order"] = classes.astype(str).tolist()
                            seed_metrics.append(metric)
                            del model
                    else:
                        candidate_record = source_freeze["candidates"][condition]
                        candidate_representation = SourceFittedTensor.from_source_artifact(
                            json.loads((fold_dir / candidate_record["path"]).read_text(encoding="utf-8"))
                        )
                        tensor = candidate_representation.transform(perturbed_iq)
                        probe = RidgeProbe.from_source_artifact(candidate_record["probe"])
                        probability = probe.predict_proba(tensor.reshape(len(tensor), -1))
                        metric = recording_level_metrics(
                            target.labels, target.recording_ids, probability, probe.classes_
                        )
                        metric["probability_class_order"] = probe.classes_.astype(str).tolist()
                        seed_metrics = [metric]
                    nuisance_records.append(
                        {
                            "case_id": case.id,
                            "nuisance": case.id.split(":", 1)[0],
                            "severity": float(case.severity),
                            "seed_metrics": seed_metrics,
                        }
                    )
                condition_curves["registered_nuisances"] = nuisance_records
                robustness[condition] = condition_curves
            result["robustness"] = robustness
            atomic_json(result_path, result)
        result["status"] = "complete"
        result["active_families"] = list(active)
        result["selected_stable_id"] = model_freeze["selected_stable_id"]
        atomic_json(result_path, result)
        if state.phase == RunPhase.TARGET_UNLOCKED:
            state.transition(RunPhase.TARGET_EVALUATED)
        all_fold_results.append(result)
    target_campaign = {
        "schema_version": "scars-target-campaign-2.0",
        "status": "complete",
        "target_access_authorized": True,
        "folds": [
            {
                "fold_id": item["fold_id"],
                "path": f"fold-{index:02d}/target_metrics.json",
                "sha256": sha256_file(source_campaign_dir / f"fold-{index:02d}/target_metrics.json"),
            }
            for index, item in enumerate(all_fold_results)
        ],
    }
    atomic_json(source_campaign_dir / "target_campaign.json", target_campaign)
    campaign["status"] = "target_evaluated"
    campaign["target_access_authorized"] = True
    atomic_json(campaign_path, campaign)
    return source_campaign_dir / "target_campaign.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-shot, checkpointed target evaluation")
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--source-campaign-dir", type=Path, required=True)
    parser.add_argument("--freeze-package", type=Path)
    parser.add_argument("--authorize-target", action="store_true")
    parser.add_argument("--pilot-three-dataset", action="store_true")
    parser.add_argument(
        "--pilot-two-dataset",
        action="store_true",
        help="Evaluate a non-confirmatory DroneRFa/DroneRFb-DIR pilot without a three-dataset freeze package.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--window-samples", type=int, default=4096)
    parser.add_argument("--hop-samples", type=int, default=2048)
    parser.add_argument("--max-windows-per-recording", type=int, default=64)
    return parser


def main() -> int:
    print(run(**vars(build_parser().parse_args())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
