from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
import pickle
import warnings
from pathlib import Path
import shutil
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from scars.baselines import (
    cwt_images,
    fixed_wavelet_subband_features,
    log_psd_sobel_images,
    log_stft_images,
    magnitude_phase,
    real_imag,
    stft_dct_features,
    torch_model_factory,
)
from scars.data.manifest import load_manifest
from scars.data.windowing import window_recordings
from scars.experiment.common import atomic_json, assert_frozen_environment, load_split_plan, one_window_per_recording
from scars.experiment.sampling import frozen_sampling_contract
from scars.models.scars_net import FamilyTeacher, SCARSNet
from scars.models.baselines import capacity_match_variants
from scars.probes.ridge import RidgeProbe
from scars.probes.xgboost_probe import XGBoostProbe
from scars.evaluation.detection import recording_binary_scores, source_threshold_by_domain
from scars.representations.tensor import SourceFittedTensor
from scars.results.provenance import environment_manifest, sha256_file
from scars.selection.nuisance import registered_nuisance_cases
from scars.training.engine import (
    TrainingSpec,
    fit_scars_with_oom_backoff,
    predict_scars,
    resolve_device,
    train_family_teacher,
)
from scars.training.baselines import fit_paired_classifier_with_oom_backoff
from scars.training.baselines import predict_classifier
from scars.state import RunState
from scars.training.pcrd import (
    build_pareto_relations,
    fit_recording_balanced_temperature,
    shuffle_relations_within_strata,
    write_relation_cache,
)
from scars.training.disk_cache import (
    ChannelSubsetArray,
    RelationTensorCache,
    build_disk_feature_pair,
    build_disk_representation_pair,
    build_relation_tensor_cache,
    build_teacher_family_training_array,
)


SEEDS = (11, 23, 37, 53, 71)
LAMBDAS = (0.1, 0.5, 1.0)
RELATION_CACHE_CHUNK_SIZE = 64
TEACHER_RESUME_SCHEMA = "scars-teacher-resume-1.0"


def _validate_resumable_model_freeze(path: Path, fold_id: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "scars-model-freeze-2.0" or payload.get("fold_id") != fold_id:
        raise RuntimeError("Existing model-freeze has an incompatible semantic identity")
    if payload.get("status") != "complete":
        return payload
    records = list(payload.get("teachers", {}).values())
    for values in payload.get("models", {}).values():
        records.extend(values)
    for value in payload.get("baselines", {}).values():
        records.extend(value.get("models", []))
        if value.get("trainer") == "ridge":
            records.append(value)
    for value in payload.get("deletion_ablations", {}).values():
        records.extend(value.get("models", []))
    for record in records:
        artifact = path.parent / record["path"]
        if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
            raise RuntimeError(f"Resume refused: checkpoint drift at {artifact.name}")
        if record.get("classes_path"):
            classes = path.parent / record["classes_path"]
            if not classes.is_file() or sha256_file(classes) != record["classes_sha256"]:
                raise RuntimeError("Resume refused: XGBoost class-order artifact drift")
    relation = payload.get("relation_cache")
    if relation:
        artifact = path.parent / relation["path"]
        if not artifact.is_file() or sha256_file(artifact) != relation["sha256"]:
            raise RuntimeError("Resume refused: PCRD relation cache drift")
    return payload


def _teacher_logits(
    model: FamilyTeacher, tensor: np.ndarray, device: torch.device, batch_size: int = 32
) -> torch.Tensor:
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(tensor), batch_size):
            batch = torch.as_tensor(
                tensor[start : start + batch_size], dtype=torch.float32, device=device
            )
            output.append(model(batch).cpu())
    return torch.cat(output)


def _save_model(path: Path, model: torch.nn.Module, payload: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            **payload,
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
        },
        temporary,
    )
    temporary.replace(path)
    return sha256_file(path)


def _torch_load_checkpoint(
    path: Path, *, trusted_campaign_checkpoint: bool = False,
) -> dict[str, object]:
    """Prefer restricted loading; legacy float metadata needs explicit trust.

    Older weights-only readers reject pickle BINFLOAT (opcode 71) used by
    temperature and training-report floats. Never retry arbitrary unpickling
    errors with unrestricted loading.
    """
    digest = sha256_file(path)
    manifest_path = path.parent / "teacher_resume.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest.get("teachers", {}).values():
            if record.get("path") == path.name and record.get("sha256") != digest:
                raise RuntimeError(f"Teacher checkpoint hash mismatch before loading: {path.name}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch releases before ``weights_only``.
        if not trusted_campaign_checkpoint:
            raise RuntimeError("Legacy checkpoint loading requires explicit campaign trust")
        payload = torch.load(path, map_location="cpu")
    except pickle.UnpicklingError as error:
        if not trusted_campaign_checkpoint or "Unsupported operand 71" not in str(error):
            raise
        warnings.warn(
            f"Loading trusted campaign checkpoint {path.name} with weights_only=False "
            "because this PyTorch restricted reader does not support float metadata. "
            "Use this option only for checkpoints created by your own campaign.",
            RuntimeWarning,
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if sha256_file(path) != digest:
        raise RuntimeError(f"Teacher checkpoint changed during loading: {path.name}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Teacher checkpoint is not a mapping: {path.name}")
    return payload


def _teacher_resume_manifest(
    fold_dir: Path,
    *,
    fold_id: str,
    source_freeze_sha256: str,
    representation_sha256: str,
    active_families: Sequence[str],
    classes: np.ndarray,
    records: dict[str, dict[str, object]],
    legacy_adopted: bool,
) -> None:
    """Bind individually saved teachers to frozen source artifacts before reuse."""
    atomic_json(
        fold_dir / "teacher_resume.json",
        {
            "schema_version": TEACHER_RESUME_SCHEMA,
            "fold_id": fold_id,
            "source_freeze_sha256": source_freeze_sha256,
            "representation_sha256": representation_sha256,
            "active_families": list(active_families),
            "classes": classes.tolist(),
            "legacy_checkpoint_adopted_after_structural_verification": bool(legacy_adopted),
            "teachers": records,
        },
    )


def _teacher_checkpoint_resumable(
    *,
    fold_dir: Path,
    fold_id: str,
    source_freeze_sha256: str,
    representation_sha256: str,
    family: str,
    family_index: int,
    active_families: Sequence[str],
    classes: np.ndarray,
    fold_index: int,
    allow_legacy_adoption: bool,
) -> bool:
    """Check checkpoint and provenance before allocating a teacher bank."""
    checkpoint = fold_dir / f"teacher-{family}.pt"
    if not checkpoint.is_file():
        return False
    payload = _torch_load_checkpoint(checkpoint, trusted_campaign_checkpoint=allow_legacy_adoption)
    report_payload = payload.get("training_report")
    expected_seed = SEEDS[0] + 100 * fold_index + family_index
    compatible = (
        payload.get("family") == family
        and payload.get("classes") == classes.tolist()
        and isinstance(report_payload, dict)
        and report_payload.get("seed") == expected_seed
        and isinstance(payload.get("state_dict"), dict)
        and np.isfinite(float(payload.get("temperature", np.nan)))
    )
    manifest_path = fold_dir / "teacher_resume.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        compatible = compatible and all(
            manifest.get(key) == expected
            for key, expected in (
                ("schema_version", TEACHER_RESUME_SCHEMA),
                ("fold_id", fold_id),
                ("source_freeze_sha256", source_freeze_sha256),
                ("representation_sha256", representation_sha256),
                ("active_families", list(active_families)),
                ("classes", classes.tolist()),
            )
        )
        record = manifest.get("teachers", {}).get(family)
        if record is None:
            # An older run can contain several valid teachers but no aggregate
            # resume manifest.  The loader adopts them sequentially into that manifest
            # only under the explicit amendment flag.
            compatible = compatible and allow_legacy_adoption
        else:
            compatible = compatible and record.get("sha256") == sha256_file(checkpoint)
    else:
        compatible = compatible and allow_legacy_adoption
    if not compatible:
        raise RuntimeError(
            f"Refusing incompatible teacher checkpoint {checkpoint.name}; create a new campaign "
            "or explicitly adopt a structurally verified legacy checkpoint"
        )
    return True


def _load_or_train_teacher(
    *,
    fold_dir: Path,
    fold_id: str,
    source_freeze_sha256: str,
    representation_sha256: str,
    family: str,
    family_index: int,
    active_families: Sequence[str],
    classes: np.ndarray,
    family_train: np.ndarray,
    family_labels: np.ndarray,
    selection_tensor: np.ndarray,
    selection_labels: np.ndarray,
    selection_recording_ids: np.ndarray,
    calibration_tensor: np.ndarray,
    calibration_labels: np.ndarray,
    calibration_recording_ids: np.ndarray,
    fold_index: int,
    spec: TrainingSpec,
    device_name: str,
    allow_temperature_boundary: bool,
    allow_legacy_adoption: bool,
    accumulated_records: dict[str, dict[str, object]],
) -> tuple[FamilyTeacher, float, dict[str, object], bool]:
    """Resume a compatible family teacher or train and immediately bind it.

    Older checkpoints lack a standalone resume manifest.  They are
    accepted only under the explicit implementation-amendment flag and only if
    family, class order, deterministic seed, finite temperature and strict
    state-dict shape all match the frozen fold.
    """
    checkpoint = fold_dir / f"teacher-{family}.pt"
    expected_seed = SEEDS[0] + 100 * fold_index + family_index
    adopted_legacy = False
    if checkpoint.is_file():
        payload = _torch_load_checkpoint(checkpoint, trusted_campaign_checkpoint=allow_legacy_adoption)
        report_payload = payload.get("training_report")
        compatible = (
            payload.get("family") == family
            and payload.get("classes") == classes.tolist()
            and isinstance(report_payload, dict)
            and report_payload.get("seed") == expected_seed
            and isinstance(payload.get("state_dict"), dict)
            and np.isfinite(float(payload.get("temperature", np.nan)))
        )
        manifest_path = fold_dir / "teacher_resume.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            compatible = compatible and all(
                manifest.get(key) == expected
                for key, expected in (
                    ("schema_version", TEACHER_RESUME_SCHEMA),
                    ("fold_id", fold_id),
                    ("source_freeze_sha256", source_freeze_sha256),
                    ("representation_sha256", representation_sha256),
                    ("active_families", list(active_families)),
                    ("classes", classes.tolist()),
                )
            )
            record = manifest.get("teachers", {}).get(family)
            if record is None:
                compatible = compatible and allow_legacy_adoption
                adopted_legacy = bool(allow_legacy_adoption)
            else:
                compatible = compatible and record.get("sha256") == sha256_file(checkpoint)
        elif allow_legacy_adoption:
            adopted_legacy = True
        else:
            compatible = False
        if compatible:
            teacher = FamilyTeacher(len(classes))
            try:
                teacher.load_state_dict(payload["state_dict"], strict=True)
            except RuntimeError as error:
                raise RuntimeError(f"Teacher checkpoint architecture drift: {checkpoint.name}") from error
            temperature = float(payload["temperature"])
            record = {
                "path": checkpoint.name,
                "sha256": sha256_file(checkpoint),
                "temperature": temperature,
                "training_report": dict(report_payload),
                "temperature_at_boundary": temperature <= 0.050001 or temperature >= 19.999999,
                "checkpoint_role": "source_selection",
                "calibration_role": "source_calibration",
                "resumed": True,
            }
            accumulated_records[family] = record
            _teacher_resume_manifest(
                fold_dir,
                fold_id=fold_id,
                source_freeze_sha256=source_freeze_sha256,
                representation_sha256=representation_sha256,
                active_families=active_families,
                classes=classes,
                records=accumulated_records,
                legacy_adopted=adopted_legacy,
            )
            return teacher, temperature, record, adopted_legacy
        raise RuntimeError(
            f"Refusing incompatible teacher checkpoint {checkpoint.name}; create a new campaign "
            "or explicitly adopt a structurally verified legacy checkpoint"
        )

    teacher, report = train_family_teacher(
        family_train,
        family_labels,
        selection_tensor,
        selection_labels,
        selection_recording_ids,
        class_count=len(classes),
        seed=expected_seed,
        spec=spec,
        device_name=device_name,
    )
    calibration_logits = _teacher_logits(teacher, calibration_tensor, resolve_device(device_name))
    temperature = fit_recording_balanced_temperature(
        calibration_logits,
        torch.as_tensor(calibration_labels),
        calibration_recording_ids,
        allow_boundary=allow_temperature_boundary,
    )
    digest = _save_model(
        checkpoint,
        teacher,
        {
            "family": family,
            "classes": classes.tolist(),
            "temperature": temperature,
            "training_report": report.to_dict(),
        },
    )
    record = {
        "path": checkpoint.name,
        "sha256": digest,
        "temperature": temperature,
        "training_report": report.to_dict(),
        "temperature_at_boundary": temperature <= 0.050001 or temperature >= 19.999999,
        "checkpoint_role": "source_selection",
        "calibration_role": "source_calibration",
        "resumed": False,
    }
    accumulated_records[family] = record
    _teacher_resume_manifest(
        fold_dir,
        fold_id=fold_id,
        source_freeze_sha256=source_freeze_sha256,
        representation_sha256=representation_sha256,
        active_families=active_families,
        classes=classes,
        records=accumulated_records,
        legacy_adopted=False,
    )
    return teacher, temperature, record, False


def _encode(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(classes)}
    try:
        return np.asarray([lookup[label] for label in labels], dtype=np.int64)
    except KeyError as error:
        raise ValueError("A source role contains a class absent from source_fit") from error


def _source_detection_threshold(batch, probability: np.ndarray, classes: np.ndarray):
    if "background" not in classes.astype(str).tolist():
        return None
    y, scores, recording_ids = recording_binary_scores(
        batch.labels, batch.recording_ids, probability, classes
    )
    domains = []
    for recording_id in recording_ids:
        values = np.unique(batch.domains[batch.recording_ids == recording_id])
        if len(values) != 1:
            raise ValueError("A source-validation recording crosses domains")
        domains.append(values[0])
    return source_threshold_by_domain(y, scores, np.asarray(domains, dtype=object))


def _fit_sensitivity_scales_from_cache(
    cache: RelationTensorCache,
    *,
    teachers: Sequence[FamilyTeacher],
    temperatures: Sequence[float],
    active_families: Sequence[str],
    device: torch.device,
    chunk_size: int,
) -> dict[str, dict[str, float]]:
    """Fit the same 95th-percentile source-calibration kappa, chunk by chunk."""
    grouped: dict[str, dict[str, dict[str, list[float]]]] = {
        family: {} for family in active_families
    }
    clean_map, perturbed_map = cache.clean, cache.perturbed
    for start in range(0, cache.sample_count, chunk_size):
        stop = min(cache.sample_count, start + chunk_size)
        clean = np.asarray(clean_map[start:stop], dtype=np.float32)
        perturbed = np.asarray(perturbed_map[start:stop], dtype=np.float32)
        recording_ids, _, nuisances, severities = cache.metadata(start, stop)
        keys = [f"{name}:{float(value):.12g}" for name, value in zip(nuisances, severities)]
        for family_index, (family, teacher, temperature) in enumerate(
            zip(active_families, teachers, temperatures)
        ):
            clean_logp = F.log_softmax(
                _teacher_logits(teacher, clean[:, family_index : family_index + 1], device)
                / temperature,
                dim=1,
            ).numpy()
            perturbed_logp = F.log_softmax(
                _teacher_logits(teacher, perturbed[:, family_index : family_index + 1], device)
                / temperature,
                dim=1,
            ).numpy()
            numerator = np.linalg.norm(perturbed_logp - clean_logp, axis=1)
            denominator = np.mean(
                np.abs(
                    np.asarray(perturbed[:, family_index : family_index + 1], dtype=float)
                    - np.asarray(clean[:, family_index : family_index + 1], dtype=float)
                ),
                axis=(1, 2, 3),
            )
            ratios = numerator / (denominator + 1.0e-8)
            if not np.all(np.isfinite(ratios)):
                raise RuntimeError("Sensitivity ratio is nonfinite")
            for recording_id, key, ratio in zip(recording_ids, keys, ratios):
                grouped[family].setdefault(key, {}).setdefault(str(recording_id), []).append(float(ratio))
    values: dict[str, dict[str, float]] = {family: {} for family in active_families}
    for family, cases in grouped.items():
        for key, by_recording in cases.items():
            medians = np.asarray(
                [np.median(samples) for _, samples in sorted(by_recording.items())], dtype=float
            )
            if len(medians) == 0 or not np.all(np.isfinite(medians)):
                raise RuntimeError(f"Sensitivity scale kappa is undefined for {family}/{key}")
            values[family][key] = float(np.quantile(medians, 0.95, method="higher"))
    return values


def _build_relations_from_cache(
    cache: RelationTensorCache,
    *,
    teachers: Sequence[FamilyTeacher],
    temperatures: Sequence[float],
    active_families: Sequence[str],
    sensitivity_scales: dict[str, dict[str, float]],
    classes: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> tuple[list, int, int, list[list[object]]]:
    """Construct Pareto relations with bounded representation working sets."""
    relations = []
    has_relation = np.zeros(cache.sample_count, dtype=bool)
    clean_map, perturbed_map = cache.clean, cache.perturbed
    for start in range(0, cache.sample_count, chunk_size):
        stop = min(cache.sample_count, start + chunk_size)
        clean = np.asarray(clean_map[start:stop], dtype=np.float32)
        perturbed = np.asarray(perturbed_map[start:stop], dtype=np.float32)
        recording_ids, labels, nuisances, severities = cache.metadata(start, stop)
        risk = np.zeros((stop - start, len(active_families)), dtype=np.float64)
        margin = np.zeros_like(risk)
        keys = [f"{name}:{float(value):.12g}" for name, value in zip(nuisances, severities)]
        true_index = _encode(labels, classes)
        for family_index, (family, teacher, temperature) in enumerate(
            zip(active_families, teachers, temperatures)
        ):
            perturbed_logits = _teacher_logits(
                teacher, perturbed[:, family_index : family_index + 1], device
            )
            perturbed_logp = F.log_softmax(perturbed_logits / temperature, dim=1).numpy()
            displacement = np.mean(
                np.abs(
                    np.asarray(perturbed[:, family_index : family_index + 1], dtype=float)
                    - np.asarray(clean[:, family_index : family_index + 1], dtype=float)
                ),
                axis=(1, 2, 3),
            )
            scales = sensitivity_scales[family]
            try:
                risk[:, family_index] = np.asarray(
                    [scales[key] for key in keys], dtype=np.float64
                ) * displacement
            except KeyError as error:
                raise RuntimeError(f"Missing frozen sensitivity scale: {family}/{error.args[0]}") from error
            correct = perturbed_logp[np.arange(len(perturbed_logp)), true_index]
            masked = perturbed_logp.copy()
            masked[np.arange(len(masked)), true_index] = -np.inf
            margin[:, family_index] = correct - np.max(masked, axis=1)
        local = build_pareto_relations(
            risk,
            margin,
            active_families,
            recording_ids,
            labels,
            nuisances,
            severities,
            tolerance=1.0e-6,
        )
        for relation in local:
            has_relation[start + relation.sample_index] = True
            relations.append(replace(relation, sample_index=start + relation.sample_index))
    missing = []
    for index in np.flatnonzero(~has_relation):
        recording_ids, labels, nuisances, severities = cache.metadata(int(index), int(index) + 1)
        missing.append(
            [str(recording_ids[0]), str(nuisances[0]), float(severities[0])]
        )
    return relations, int(cache.sample_count), int(has_relation.sum()), missing


def _release_memmaps(*arrays: object) -> None:
    """Close mapped files before removing a temporary baseline cache."""
    for array in arrays:
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()
    gc.collect()


def run(
    *,
    preflight_dir: Path,
    source_campaign_dir: Path,
    device: str,
    window_samples: int,
    hop_samples: int,
    max_windows_per_recording: int,
    max_epochs: int,
    patience: int,
    allow_memory_implementation_amendment: bool = False,
) -> Path:
    campaign_path = source_campaign_dir / "source_campaign.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if campaign.get("status") not in {"all_source_folds_frozen", "all_source_models_frozen"}:
        raise PermissionError("All source representation folds must be frozen first")
    project_root = Path(__file__).resolve().parents[3]
    try:
        assert_frozen_environment(campaign, project_root)
    except RuntimeError as error:
        # The memory amendment changes only how the already frozen source computations are
        # materialized: bounded chunks and disk-backed arrays replace the previous
        # all-in-RAM relation arrays.  Reuse is opt-in and is recorded without
        # mutating the original source-freeze provenance.
        if not allow_memory_implementation_amendment or "source_tree_sha256" not in str(error):
            raise
        config_paths = sorted((project_root / "configs").glob("*.yaml")) + sorted(
            (project_root / "configs").glob("*.json")
        )
        current = environment_manifest(project_root, config_paths)
        frozen = campaign["provenance"]
        unchanged = ("combined_config_hash", "packages", "platform", "hardware")
        changed = [key for key in unchanged if current.get(key) != frozen.get(key)]
        if changed:
            raise RuntimeError(
                "Memory implementation amendment may not cross execution-environment drift: "
                + ", ".join(changed)
            ) from error
        if campaign.get("target_access_count", 0) != 0:
            raise PermissionError("Implementation amendment is prohibited after target access")
        atomic_json(
            source_campaign_dir / "memory_implementation_amendment.json",
            {
                "schema_version": "scars-memory-amendment-1.0",
                "scope": "memory_bounded_relation_and_baseline_materialization_only",
                "frozen_source_tree_sha256": frozen["source_tree_sha256"],
                "current_source_tree_sha256": current["source_tree_sha256"],
                "verified_unchanged": list(unchanged),
                "target_access_count": 0,
                "scientific_contract": "No change to windows, source roles, nuisance grid, representation, model, loss, hypotheses, or decision rules.",
            },
        )
    preflight = json.loads((preflight_dir / "preflight.json").read_text(encoding="utf-8"))
    from scars.experiment.pilot_policy import is_pilot, policy_for
    pilot = is_pilot(preflight)
    POLICY = policy_for(preflight) if pilot else None
    if preflight.get("status") != "ready":
        raise PermissionError("Preflight is not ready")
    if sha256_file(preflight_dir / "preflight.json") != campaign["input_hashes"]["preflight"]:
        raise RuntimeError("Training preflight differs from the source-freeze preflight")
    requested_windowing = {
        "window_samples": window_samples,
        "hop_samples": hop_samples,
        "max_windows_per_recording": max_windows_per_recording,
    }
    if requested_windowing != campaign.get("windowing"):
        raise ValueError("Training windowing differs from the frozen source campaign")
    if max_epochs != 100 or patience != 10:
        raise ValueError("Confirmatory training fixes max_epochs=100 and patience=10")
    recordings = load_manifest(
        preflight_dir / preflight["artifacts"]["recordings_manifest"]
    )
    folds = load_split_plan(
        preflight_dir / preflight["artifacts"]["split_manifest"], recordings
    )
    spec = TrainingSpec(
        max_epochs=max_epochs,
        patience=patience,
        batch_size=32,
        effective_batch_size=64,
        minimum_batch_size=4,
        amp=True,
    )
    resolved_device = resolve_device(device)
    fold_records = []
    for fold_index, (fold, campaign_fold) in enumerate(zip(folds, campaign["folds"])):
        fold_dir = source_campaign_dir / campaign_fold["directory"]
        model_freeze_path = fold_dir / "model_freeze.json"
        if model_freeze_path.is_file():
            _validate_resumable_model_freeze(model_freeze_path, fold["fold_id"])
            fold_records.append(
                {
                    "fold_id": fold["fold_id"],
                    "path": str(model_freeze_path.relative_to(source_campaign_dir)),
                    "sha256": sha256_file(model_freeze_path),
                    "resumed": True,
                }
            )
            continue
        freeze = json.loads((fold_dir / "source_freeze.json").read_text(encoding="utf-8"))
        if pilot and freeze.get("pilot_policy") != POLICY:
            raise RuntimeError("Create a new amended pilot or import the old source artifacts with prepare_pilot_amendment")
        selected_id = freeze["selected_stable_id"]
        selected_record = freeze["canonical_tensor_artifact"]
        representation_artifact = json.loads(
            (fold_dir / selected_record["path"]).read_text(encoding="utf-8")
        )
        if sha256_file(fold_dir / selected_record["path"]) != selected_record["sha256"]:
            raise RuntimeError("Selected representation artifact hash mismatch")
        representation = SourceFittedTensor.from_source_artifact(representation_artifact)
        base_families = representation.config.active_families()
        if base_families != ("W", "C", "E", "S"):
            raise RuntimeError("Canonical representation is not ordered W,C,E,S")
        active_families = tuple(freeze["active_families"])
        active_indices = list(freeze["active_indices"])
        if active_families != tuple(base_families[index] for index in active_indices):
            raise RuntimeError("Frozen active-family selector does not match the representation artifact")
        if len(active_families) < 2:
            payload = {
                "schema_version": "scars-model-freeze-2.0",
                "fold_id": fold["fold_id"],
                "status": "scars_net_not_applicable_after_registered_shrink",
                "selected_stable_id": selected_id,
                "active_families": list(active_families),
                "active_indices": active_indices,
                "channel_decisions": freeze["channel_decisions"],
                "negative_controls": freeze["negative_controls"],
                "reason": "PCRD requires at least one comparable family pair",
            }
            digest = atomic_json(model_freeze_path, payload)
            state = RunState(fold_dir / "run_state.json")
            state.source_artifact_hash = digest
            state.save()
            fold_records.append(
                {"fold_id": fold["fold_id"], "path": str(model_freeze_path.relative_to(source_campaign_dir)), "sha256": digest}
            )
            continue
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
        # source_fit can be much larger than the three source control roles.
        # Its teacher tensors are materialized one family at a time below, so
        # do not retain a four-family all-window source-fit tensor in RAM.
        tensors = {
            role: representation.transform(batches[role].iq)[:, active_indices]
            for role in ("source_calibration", "source_selection", "source_validation")
        }
        classes = np.unique(batches["source_fit"].labels)
        encoded = {
            role: _encode(batches[role].labels, classes)
            for role in ("source_calibration", "source_selection", "source_validation")
        }
        # This one-window bank is the unchanged PCRD/one-window-neural sample
        # population.  Keep it before releasing the large all-window tensor.
        relation_source = one_window_per_recording(batches["source_fit"])
        teacher_records = {}
        teachers: list[FamilyTeacher] = []
        temperatures = []
        legacy_teacher_adopted = False
        for family_index, family in enumerate(active_families):
            resumable = _teacher_checkpoint_resumable(
                fold_dir=fold_dir,
                fold_id=fold["fold_id"],
                source_freeze_sha256=campaign_fold["source_freeze_sha256"],
                representation_sha256=selected_record["sha256"],
                family=family,
                family_index=family_index,
                active_families=active_families,
                classes=classes,
                fold_index=fold_index,
                allow_legacy_adoption=allow_memory_implementation_amendment,
            ) if (fold_dir / f"teacher-{family}.pt").is_file() else False
            teacher_scratch = fold_dir / "teacher_tensor_scratch_v2" / family
            if resumable:
                # _load_or_train_teacher returns before examining these empty
                # sentinels.  Avoid materializing a disk bank for a frozen,
                # provenance-compatible teacher.
                family_train = np.empty((0, 1, 1, 1), dtype=np.float32)
                family_labels = np.empty(0, dtype=np.int64)
                fit_labels = None
            else:
                family_train = build_teacher_family_training_array(
                    teacher_scratch,
                    batches["source_fit"],
                    representation,
                    family_index=active_indices[family_index],
                    seed=70_000 + fold_index,
                    chunk_size=RELATION_CACHE_CHUNK_SIZE,
                )
                fit_labels = _encode(batches["source_fit"].labels, classes)
                family_labels = np.concatenate([fit_labels, fit_labels])
            teacher, temperature, record, adopted = _load_or_train_teacher(
                fold_dir=fold_dir,
                fold_id=fold["fold_id"],
                source_freeze_sha256=campaign_fold["source_freeze_sha256"],
                representation_sha256=selected_record["sha256"],
                family=family,
                family_index=family_index,
                active_families=active_families,
                classes=classes,
                family_train=family_train,
                family_labels=family_labels,
                selection_tensor=tensors["source_selection"][:, family_index : family_index + 1],
                selection_labels=encoded["source_selection"],
                selection_recording_ids=batches["source_selection"].recording_ids,
                calibration_tensor=tensors["source_calibration"][:, family_index : family_index + 1],
                calibration_labels=encoded["source_calibration"],
                calibration_recording_ids=batches["source_calibration"].recording_ids,
                fold_index=fold_index,
                spec=spec,
                device_name=device,
                allow_temperature_boundary=pilot,
                allow_legacy_adoption=allow_memory_implementation_amendment,
                accumulated_records=teacher_records,
            )
            legacy_teacher_adopted = legacy_teacher_adopted or adopted
            teacher.to(resolved_device).eval()
            teachers.append(teacher)
            temperatures.append(temperature)
            _release_memmaps(family_train)
            del family_train, family_labels, fit_labels
            if not resumable:
                shutil.rmtree(teacher_scratch, ignore_errors=True)
            gc.collect()

        # Relation construction below receives only a one-window physical
        # recording bank.  Full all-window perturbation tensors are
        # never created in host RAM.
        gc.collect()

        # Fit empirical sensitivity scales exclusively on source_calibration,
        # then freeze them before constructing source_fit PCRD relations.
        calibration_relation_source = one_window_per_recording(
            batches["source_calibration"]
        )
        calibration_tensor_cache = build_relation_tensor_cache(
            fold_dir / "calibration_relation_tensors_v2",
            calibration_relation_source,
            representation,
            seed=75_000 + fold_index,
            active_indices=active_indices,
            representation_sha256=selected_record["sha256"],
            chunk_size=RELATION_CACHE_CHUNK_SIZE,
        )
        frozen_sensitivity_scales = _fit_sensitivity_scales_from_cache(
            calibration_tensor_cache,
            teachers=teachers,
            temperatures=temperatures,
            active_families=active_families,
            device=resolved_device,
            chunk_size=RELATION_CACHE_CHUNK_SIZE,
        )
        del calibration_relation_source
        _release_memmaps(calibration_tensor_cache.clean, calibration_tensor_cache.perturbed)
        gc.collect()

        relation_tensor_cache = build_relation_tensor_cache(
            fold_dir / "relation_tensors_v2",
            relation_source,
            representation,
            seed=80_000 + fold_index,
            active_indices=active_indices,
            representation_sha256=selected_record["sha256"],
            chunk_size=RELATION_CACHE_CHUNK_SIZE,
        )
        relations, expected_cell_count, observed_cell_count, missing_cells = _build_relations_from_cache(
            relation_tensor_cache,
            teachers=teachers,
            temperatures=temperatures,
            active_families=active_families,
            sensitivity_scales=frozen_sensitivity_scales,
            classes=classes,
            device=resolved_device,
            chunk_size=RELATION_CACHE_CHUNK_SIZE,
        )
        if missing_cells and not pilot:
            for teacher in teachers:
                teacher.cpu()
            del teachers
            if resolved_device.type == "cuda":
                torch.cuda.empty_cache()
            payload = {
                "schema_version": "scars-model-freeze-2.0",
                "fold_id": fold["fold_id"],
                "status": "source_decision_rejected_incomplete_relation_cells",
                "selected_stable_id": selected_id,
                "selected_representation_sha256": selected_record["sha256"],
                "active_families": list(active_families),
                "active_indices": active_indices,
                "channel_decisions": freeze["channel_decisions"],
                "negative_controls": freeze["negative_controls"],
                "relation_coverage": {
                    "expected_cell_count": expected_cell_count,
                    "observed_cell_count": observed_cell_count,
                    "complete_cells": False,
                    "missing_cells": [list(cell) for cell in missing_cells],
                    "tolerance": 1.0e-6,
                },
                "reason": "Every registered recording-by-nuisance-by-severity cell must yield a frozen Pareto relation",
            }
            digest = atomic_json(model_freeze_path, payload)
            fold_records.append(
                {"fold_id": fold["fold_id"], "path": str(model_freeze_path.relative_to(source_campaign_dir)), "sha256": digest}
            )
            continue
        for teacher in teachers:
            teacher.cpu()
        del teachers
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()
        relation_cache = write_relation_cache(fold_dir / "relations.json", relations)
        clean = relation_tensor_cache.clean
        perturbed = relation_tensor_cache.perturbed
        relation_recording_set = sorted({relation.recording_id for relation in relations})
        coverage_by_nuisance = {
            nuisance: len({relation.recording_id for relation in relations if relation.nuisance == nuisance})
            / len(set(map(str, relation_source.recording_ids.tolist())))
            for nuisance in sorted({case.id.split(":", 1)[0] for case in registered_nuisance_cases()})
        }
        relation_labels = np.repeat(
            np.asarray(relation_source.labels, dtype=object), len(registered_nuisance_cases())
        )
        validation_tensor = tensors["source_validation"]
        validation_labels = encoded["source_validation"]
        validation_recordings = batches["source_validation"].recording_ids
        model_records = {}

        def train_condition(condition: str, lambda_value: float, relation_set: Sequence, seed_value: int):
            model, report = fit_scars_with_oom_backoff(
                lambda: SCARSNet(active_families, len(classes)),
                clean,
                perturbed,
                _encode(relation_labels, classes),
                relation_set,
                validation_tensor,
                validation_labels,
                validation_recordings,
                seed=seed_value,
                lambda_pcrd=lambda_value,
                spec=spec,
                device_name=device,
            )
            checkpoint = fold_dir / f"{condition}-seed{seed_value}.pt"
            validation_probability, _ = predict_scars(
                model,
                validation_tensor,
                device=resolved_device,
                batch_size=report.batch_size,
            )
            threshold = _source_detection_threshold(
                batches["source_validation"], validation_probability, classes
            )
            digest = _save_model(
                checkpoint,
                model,
                {
                    "condition_id": condition,
                    "active_families": list(active_families),
                    "classes": classes.tolist(),
                    "lambda_pcrd": lambda_value,
                    "training_report": report.to_dict(),
                },
            )
            result = {
                "path": checkpoint.name,
                "sha256": digest,
                "report": report.to_dict(),
                "source_detection_threshold": threshold,
            }
            model.cpu()
            del model
            if resolved_device.type == "cuda":
                torch.cuda.empty_cache()
            return result

        lambda_records = {}
        effective_lambdas = LAMBDAS if relations else (0.0,)
        for lambda_value in effective_lambdas:
            key = f"lambda={lambda_value:g}"
            condition_name = f"pcrd_lambda_{str(lambda_value).replace('.', 'p')}"
            lambda_records[key] = [
                train_condition(condition_name, lambda_value, relations, seed_value)
                for seed_value in SEEDS
            ]
        selected_lambda = max(
            effective_lambdas,
            key=lambda value: (
                np.mean(
                    [
                        item["report"]["best_source_validation_macro_f1"]
                        for item in lambda_records[f"lambda={value:g}"]
                    ]
                ),
                -value,
            ),
        )
        model_records["pcrd"] = lambda_records[f"lambda={selected_lambda:g}"]
        shuffle_distinct = True
        try:
            shuffled = shuffle_relations_within_strata(relations, seed=24025 + fold_index)
        except RuntimeError:
            if not pilot:
                raise
            shuffled = list(relations)
            shuffle_distinct = False
        model_records["ordinary_gate"] = [
            train_condition("ordinary_gate", 0.0, (), seed_value) for seed_value in SEEDS
        ]
        model_records["shuffled_pcrd"] = [
            train_condition("shuffled_pcrd", selected_lambda, shuffled, seed_value)
            for seed_value in SEEDS
        ]

        deletion_records = {}
        for deleted_index, deleted_family in enumerate(active_families):
            retained = tuple(
                family for family in active_families if family != deleted_family
            )
            retained_indices = [
                index for index, family in enumerate(active_families) if family in retained
            ]
            index_map = {old: new for new, old in enumerate(retained_indices)}
            deletion_relations = [
                replace(
                    relation,
                    winner=index_map[relation.winner],
                    loser=index_map[relation.loser],
                )
                for relation in relations
                if relation.winner in index_map and relation.loser in index_map
            ]
            condition_id = f"active_minus_{deleted_family}"
            deletion_pcrd_applicable = len(retained) >= 2 and (bool(deletion_relations) or not pilot)
            if deletion_pcrd_applicable and not deletion_relations:
                raise RuntimeError(
                    f"{condition_id} retains multiple families but has no valid PCRD relation"
                )
            deletion_lambda = selected_lambda if deletion_pcrd_applicable else 0.0
            records = []
            for seed_value in SEEDS:
                model, report = fit_scars_with_oom_backoff(
                    lambda retained=retained: SCARSNet(retained, len(classes)),
                    ChannelSubsetArray(clean, retained_indices),
                    ChannelSubsetArray(perturbed, retained_indices),
                    _encode(relation_labels, classes),
                    deletion_relations,
                    validation_tensor[:, retained_indices],
                    validation_labels,
                    validation_recordings,
                    seed=seed_value,
                    lambda_pcrd=deletion_lambda,
                    spec=spec,
                    device_name=device,
                )
                checkpoint = fold_dir / f"{condition_id}-seed{seed_value}.pt"
                digest = _save_model(
                    checkpoint,
                    model,
                    {
                        "condition_id": condition_id,
                        "active_families": list(retained),
                        "classes": classes.tolist(),
                        "lambda_pcrd": deletion_lambda,
                        "training_report": report.to_dict(),
                    },
                )
                records.append(
                    {"path": checkpoint.name, "sha256": digest, "report": report.to_dict()}
                )
                model.cpu()
                del model
                if resolved_device.type == "cuda":
                    torch.cuda.empty_cache()
            deletion_records[condition_id] = {
                "status": "complete",
                "deleted_family": deleted_family,
                "active_families": list(retained),
                "retained_indices": retained_indices,
                "models": records,
                "lambda_pcrd": deletion_lambda,
                "pcrd_applicable": deletion_pcrd_applicable,
                "training_objective": (
                    "paired_two_CE_plus_PCRD"
                    if deletion_pcrd_applicable
                    else (
                        "paired_two_CE_no_observed_PCRD_relations_pilot"
                        if len(retained) >= 2
                        else "paired_two_CE_single_family_PCRD_structurally_not_applicable"
                    )
                ),
            }

        # Registered comparison matrix. Every learned control sees the same
        # expanded clean/perturbed examples and source-validation checkpoint rule.
        baseline_records = {}
        baseline_labels = _encode(relation_labels, classes)

        def train_learned_baseline(
            condition_id: str,
            train_clean: np.ndarray,
            train_perturbed: np.ndarray,
            validation: np.ndarray,
            preprocessing: dict[str, object],
            *,
            early_width: int = 32,
            architecture_id: str | None = None,
        ):
            records = []
            for seed_value in SEEDS:
                factory = torch_model_factory(
                    architecture_id or condition_id,
                    class_count=len(classes),
                    active_families=active_families,
                    early_fusion_width=early_width,
                )
                artifact = fit_paired_classifier_with_oom_backoff(
                    factory,
                    train_clean,
                    train_perturbed,
                    baseline_labels,
                    validation,
                    encoded["source_validation"],
                    batches["source_validation"].recording_ids,
                    seed=seed_value,
                    spec=spec,
                    device_name=device,
                )
                checkpoint = fold_dir / f"{condition_id}-seed{seed_value}.pt"
                validation_probability = predict_classifier(
                    artifact.model,
                    validation,
                    device=resolved_device,
                    batch_size=artifact.report.batch_size,
                )
                threshold = _source_detection_threshold(
                    batches["source_validation"], validation_probability, classes
                )
                digest = _save_model(
                    checkpoint,
                    artifact.model,
                    {
                        "condition_id": condition_id,
                        "classes": classes.tolist(),
                        "active_families": list(active_families),
                        "early_fusion_width": early_width,
                        "preprocessing": preprocessing,
                        "training_report": artifact.report.to_dict(),
                    },
                )
                records.append(
                    {
                        "path": checkpoint.name,
                        "sha256": digest,
                        "report": artifact.report.to_dict(),
                        "source_detection_threshold": threshold,
                    }
                )
                artifact.model.cpu()
                del artifact
                if resolved_device.type == "cuda":
                    torch.cuda.empty_cache()
            baseline_records[condition_id] = {
                "trainer": "torch",
                "preprocessing": preprocessing,
                "models": records,
            }

        # Each baseline gets a short-lived disk pair to avoid retaining
        # expanded clean/perturbed IQ arrays for multiple representations.
        baseline_scratch = fold_dir / "baseline_tensor_scratch_v2"

        def train_disk_feature_baselines(
            identifiers: Sequence[str], transform, cache_name: str,
        ) -> None:
            cache_dir = baseline_scratch / cache_name
            paired_clean, paired_perturbed, validation, preprocessing = build_disk_feature_pair(
                cache_dir,
                relation_tensor_cache,
                relation_source,
                transform,
                source_fit_iq=batches["source_fit"].iq,
                validation_iq=batches["source_validation"].iq,
                chunk_size=RELATION_CACHE_CHUNK_SIZE,
            )
            try:
                for identifier in identifiers:
                    train_learned_baseline(
                        identifier, paired_clean, paired_perturbed, validation, preprocessing
                    )
            finally:
                _release_memmaps(paired_clean, paired_perturbed)
                shutil.rmtree(cache_dir, ignore_errors=True)
                gc.collect()

        train_disk_feature_baselines(("raw_iq_cnn",), real_imag, "raw_iq")
        train_disk_feature_baselines(("magnitude_phase_cnn",), magnitude_phase, "magnitude_phase")
        train_disk_feature_baselines(
            ("log_stft_cnn", "patch_transformer"),
            lambda iq: log_stft_images(iq, 16),
            "log_stft",
        )
        train_disk_feature_baselines(
            ("cwt_cnn",), lambda iq: cwt_images(iq, 16), "cwt"
        )
        train_disk_feature_baselines(
            ("log_psd_sobel",), lambda iq: log_psd_sobel_images(iq, 16), "log_psd_sobel"
        )

        for condition_id, candidate_id in (("wst_only", "W"), ("cyclic_only", "C")):
            record = freeze["candidates"][candidate_id]
            candidate_artifact = json.loads(
                (fold_dir / record["path"]).read_text(encoding="utf-8")
            )
            candidate_representation = SourceFittedTensor.from_source_artifact(candidate_artifact)
            cache_dir = baseline_scratch / condition_id
            paired_clean, paired_perturbed, validation, preprocessing = build_disk_representation_pair(
                cache_dir,
                relation_tensor_cache,
                relation_source,
                candidate_representation,
                batches["source_validation"].iq,
                chunk_size=RELATION_CACHE_CHUNK_SIZE,
            )
            preprocessing.update(
                {
                    "representation_artifact": record["path"],
                    "representation_sha256": record["sha256"],
                }
            )
            try:
                train_learned_baseline(
                    condition_id, paired_clean, paired_perturbed, validation, preprocessing
                )
            finally:
                _release_memmaps(paired_clean, paired_perturbed)
                shutil.rmtree(cache_dir, ignore_errors=True)
                gc.collect()

        reference = SCARSNet(active_families, len(classes))
        matching = capacity_match_variants(
            reference,
            len(active_families),
            len(classes),
            height=representation.config.output_bins,
            width=representation.config.output_bins,
        )
        parameter_width = int(matching["parameter_matched"]["base_width"])
        mac_width = int(matching["mac_matched"]["base_width"])
        train_learned_baseline(
            "early_fusion_resnet",
            clean,
            perturbed,
            validation_tensor,
            {"selected_representation_sha256": selected_record["sha256"], "capacity_matching": matching},
            early_width=parameter_width,
        )
        if mac_width != parameter_width or not matching["parameter_matched"]["mac_gate"]:
            train_learned_baseline(
                "early_fusion_resnet_mac_matched",
                clean,
                perturbed,
                validation_tensor,
                {
                    "selected_representation_sha256": selected_record["sha256"],
                    "capacity_matching": matching,
                    "reporting_role": "mandatory_mac_matched_sensitivity",
                },
                early_width=mac_width,
                architecture_id="early_fusion_resnet",
            )
        train_learned_baseline(
            "uniform_late_fusion",
            clean,
            perturbed,
            validation_tensor,
            {"selected_representation_sha256": selected_record["sha256"]},
        )

        fixed_probe = RidgeProbe().fit(
            fixed_wavelet_subband_features(batches["source_fit"].iq),
            batches["source_fit"].labels,
        )
        fixed_path = fold_dir / "fixed_wavelet_subbands.json"
        fixed_digest = atomic_json(fixed_path, fixed_probe.source_artifact())
        fixed_validation_probability = fixed_probe.predict_proba(
            fixed_wavelet_subband_features(batches["source_validation"].iq)
        )
        baseline_records["fixed_wavelet_subbands"] = {
            "trainer": "ridge",
            "path": fixed_path.name,
            "sha256": fixed_digest,
            "preprocessing": "level3_db4_packet_log_energy_8",
            "source_detection_threshold": _source_detection_threshold(
                batches["source_validation"], fixed_validation_probability, fixed_probe.classes_
            ),
        }

        dct_fit = stft_dct_features(batches["source_fit"].iq)
        dct_selection = stft_dct_features(batches["source_selection"].iq)
        xgb_models = []
        for seed_value in SEEDS:
            probe = XGBoostProbe(seed_value).fit(
                dct_fit,
                batches["source_fit"].labels,
                dct_selection,
                batches["source_selection"].labels,
            )
            model_path = fold_dir / f"stft_dct_xgboost-seed{seed_value}.json"
            probe.save_model(model_path)
            xgb_validation_probability = probe.predict_proba(
                stft_dct_features(batches["source_validation"].iq)
            )
            xgb_models.append(
                {
                    "path": model_path.name,
                    "sha256": sha256_file(model_path),
                    "classes_path": model_path.with_suffix(model_path.suffix + ".classes.npy").name,
                    "classes_sha256": sha256_file(model_path.with_suffix(model_path.suffix + ".classes.npy")),
                    "source_detection_threshold": _source_detection_threshold(
                        batches["source_validation"], xgb_validation_probability, probe.classes_
                    ),
                }
            )
        baseline_records["stft_dct_xgboost"] = {
            "trainer": "xgboost",
            "models": xgb_models,
            "preprocessing": "hann128_hop32_log_stft_dct2_zigzag64",
        }

        payload = {
            "schema_version": "scars-model-freeze-2.0",
            "fold_id": fold["fold_id"],
            "status": "complete",
            "selected_stable_id": selected_id,
            "selected_representation_sha256": selected_record["sha256"],
            "active_families": list(active_families),
            "active_indices": active_indices,
            "channel_decisions": freeze["channel_decisions"],
            "classes": classes.tolist(),
            "teachers": teacher_records,
            "relation_cache": relation_cache,
            "sensitivity_scales": {
                "fit_role": "source_calibration",
                "frozen_before_relation_role": "source_fit",
                "values": frozen_sensitivity_scales,
            },
            "relation_coverage": {
                "recording_count": len(relation_recording_set),
                "by_nuisance": coverage_by_nuisance,
                "expected_cell_count": expected_cell_count,
                "observed_cell_count": observed_cell_count,
                "complete_cells": not missing_cells,
                "missing_cells": [list(cell) for cell in missing_cells],
                "tolerance": 1.0e-6,
            },
            "lambda_candidates": list(effective_lambdas),
            "selected_lambda": selected_lambda,
            "selection_role": "source_validation",
            "models": model_records,
            "baselines": baseline_records,
            "deletion_ablations": deletion_records,
            "paired_seeds": list(SEEDS),
            "training_spec": spec.__dict__,
            "sampling_contract": frozen_sampling_contract(),
            "windowing": requested_windowing,
            "target_reads": 0,
            "memory_bounded_storage": {
                "relation_tensor_cache": "relation_tensors_v2/manifest.json",
                "calibration_tensor_cache": "calibration_relation_tensors_v2/manifest.json",
                "relation_chunk_size": RELATION_CACHE_CHUNK_SIZE,
                "teacher_resume_manifest": "teacher_resume.json",
                "legacy_teacher_adopted_after_verification": legacy_teacher_adopted,
                "temporary_baseline_caches_removed_after_checkpoint": True,
            },
        }
        if pilot:
            payload["pilot_policy"] = dict(POLICY)
            payload["pilot_mechanism_status"] = {
                "pcrd_available": bool(relations),
                "shuffle_distinct": shuffle_distinct,
                "m1_confirmatory": False,
                "empty_relation_fallback": "CE_only" if not relations else None,
            }
        digest = atomic_json(model_freeze_path, payload)
        state = RunState(fold_dir / "run_state.json")
        state.source_artifact_hash = digest
        state.save()
        fold_records.append(
            {"fold_id": fold["fold_id"], "path": str(model_freeze_path.relative_to(source_campaign_dir)), "sha256": digest}
        )
        _release_memmaps(clean, perturbed)
        del clean, perturbed, relation_tensor_cache, calibration_tensor_cache
        gc.collect()
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()
    model_statuses = [
        json.loads((source_campaign_dir / record["path"]).read_text(encoding="utf-8"))["status"]
        for record in fold_records
    ]
    campaign["status"] = (
        "all_source_models_frozen"
        if all(status == "complete" for status in model_statuses)
        else "source_decision_rejected_scars_net"
    )
    campaign["model_freezes"] = fold_records
    campaign["model_freeze_statuses"] = model_statuses
    campaign["target_access_authorized"] = False
    atomic_json(campaign_path, campaign)
    return campaign_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train source-only teachers and SCARS-Net/PCRD controls")
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--source-campaign-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--window-samples", type=int, default=4096)
    parser.add_argument("--hop-samples", type=int, default=2048)
    parser.add_argument("--max-windows-per-recording", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--allow-memory-implementation-amendment",
        action="store_true",
        help="Allow bounded-memory execution to resume a compatible pre-target source campaign after provenance verification.",
    )
    return parser


def main() -> int:
    print(run(**vars(build_parser().parse_args())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
