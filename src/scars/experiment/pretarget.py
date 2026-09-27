from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scars.data.manifest import load_manifest
from scars.data.splits import assert_disjoint
from scars.experiment.common import assert_frozen_environment, load_split_plan
from scars.results.provenance import sha256_file


FROZEN_DATASETS = {"DroneRFa", "DroneRFb-DIR", "DRFF-R2"}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_record(root: Path, record: dict[str, Any], role: str) -> str:
    if not record.get("path") or not record.get("sha256"):
        raise RuntimeError(f"Missing {role} path/hash")
    path = root / str(record["path"])
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"Frozen {role} artifact drift: {path}")
    return str(path)


def _model_artifact_records(model_freeze: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    records: list[tuple[dict[str, Any], str]] = []
    records.extend((item, f"teacher-{name}") for name, item in model_freeze.get("teachers", {}).items())
    for condition, values in model_freeze.get("models", {}).items():
        records.extend((item, f"model-{condition}") for item in values)
    for condition, value in model_freeze.get("baselines", {}).items():
        if value.get("trainer") == "ridge":
            records.append((value, f"baseline-{condition}"))
        records.extend((item, f"baseline-{condition}") for item in value.get("models", []))
    for condition, value in model_freeze.get("deletion_ablations", {}).items():
        records.extend((item, f"deletion-{condition}") for item in value.get("models", []))
    relation = model_freeze.get("relation_cache")
    if relation:
        records.append((relation, "relation-cache"))
    return records


def verify_source_campaign_pre_target(
    source_campaign_dir: Path,
    *,
    expected_label_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify every source artifact before a target state transition or waveform read."""
    root = Path(source_campaign_dir).resolve()
    campaign_path = root / "source_campaign.json"
    campaign = _read(campaign_path)
    if campaign.get("status") != "all_source_models_frozen":
        raise PermissionError("Every source fold/model must be frozen before target authorization")
    if campaign.get("target_access_authorized") is not False:
        raise PermissionError("Source campaign target authorization is not pristine")
    if (root / "target_campaign.json").exists():
        raise PermissionError("A target campaign artifact already exists")
    project_root = Path(__file__).resolve().parents[3]
    assert_frozen_environment(campaign, project_root)

    preflight_path = root / "preflight.json"
    recordings_path = root / "recordings_recognition.json"
    splits_path = root / "splits.json"
    preflight = _read(preflight_path)
    if preflight.get("status") != "ready" or not preflight.get("content_hashes_complete"):
        raise PermissionError("Eligible-data preflight and complete content hashes are required")
    if set(preflight.get("datasets", [])) != FROZEN_DATASETS or preflight.get("fold_count") != 3:
        raise PermissionError("The campaign requires exactly the three frozen datasets/folds")
    if expected_label_contract_sha256 is not None and preflight.get("label_contract_sha256") != expected_label_contract_sha256:
        raise RuntimeError("Freeze ontology hash differs from source preflight ontology")
    for key, path in (("preflight", preflight_path), ("recordings", recordings_path), ("splits", splits_path)):
        if sha256_file(path) != campaign.get("input_hashes", {}).get(key):
            raise RuntimeError(f"Source campaign input hash drift: {key}")

    recordings = load_manifest(recordings_path)
    folds = load_split_plan(splits_path, recordings)
    if len(folds) != 3 or {record.dataset for record in recordings} != FROZEN_DATASETS:
        raise RuntimeError("Frozen manifest does not contain exactly three eligible datasets")
    for fold in folds:
        assert_disjoint(
            fold["source_fit"], fold["source_calibration"], fold["source_selection"],
            fold["source_validation"], fold["held_target"],
        )
        if not all(
            record.metadata.get("temporal_adjacency_subsumed_by_split_group_verified", False)
            for role in ("source_fit", "source_calibration", "source_selection", "source_validation", "held_target")
            for record in fold[role]
        ):
            raise RuntimeError("Temporal adjacency is not verified by physical split group")

    if len(campaign.get("folds", [])) != 3 or len(campaign.get("model_freezes", [])) != 3:
        raise RuntimeError("Source campaign does not contain exactly three complete folds")
    verified_paths: list[str] = [str(preflight_path), str(recordings_path), str(splits_path)]
    fold_lookup = {fold["fold_id"]: fold for fold in folds}
    for campaign_fold, model_record in zip(campaign["folds"], campaign["model_freezes"]):
        fold_id = campaign_fold["fold_id"]
        if fold_id not in fold_lookup or model_record.get("fold_id") != fold_id:
            raise RuntimeError("Campaign/source/model fold identities are misaligned")
        fold_dir = root / campaign_fold["directory"]
        source_path = fold_dir / "source_freeze.json"
        if not source_path.is_file() or sha256_file(source_path) != campaign_fold.get("source_freeze_sha256"):
            raise RuntimeError("Source-freeze hash drift before target authorization")
        source = _read(source_path)
        if source.get("target_reads") != 0 or source.get("fold_id") != fold_id:
            raise PermissionError("Source-freeze target ledger is not pristine")
        expected_roles = {
            role: [record.recording_id for record in fold_lookup[fold_id][role]]
            for role in ("source_fit", "source_calibration", "source_selection", "source_validation")
        }
        if source.get("source_roles") != expected_roles:
            raise RuntimeError("Source-freeze roles differ from the split manifest")
        if source.get("held_target_recordings") != [
            record.recording_id for record in fold_lookup[fold_id]["held_target"]
        ]:
            raise RuntimeError("Source-freeze held IDs differ from the split manifest")
        representation_records = (
            list(source.get("candidates", {}).values())
            + list(source.get("h2_candidates", {}).values())
            + [source.get("canonical_tensor_artifact", {})]
        )
        for item in representation_records:
            verified_paths.append(_verify_record(fold_dir, item, "representation"))
        if set(source.get("h2_candidates", {})) != set(source.get("h2_configuration_order", [])):
            raise RuntimeError("Frozen H2 artifact bank is incomplete")
        if set(source.get("h2_source_validation_records", {})) != set(source.get("h2_configuration_order", [])):
            raise RuntimeError("Frozen H2 source-record bank is incomplete")

        model_path = root / str(model_record.get("path", ""))
        if not model_path.is_file() or sha256_file(model_path) != model_record.get("sha256"):
            raise RuntimeError("Model-freeze hash drift before target authorization")
        model = _read(model_path)
        if model.get("status") != "complete" or model.get("target_reads") != 0:
            raise PermissionError("Model freeze is incomplete or its target ledger is not pristine")
        if model.get("fold_id") != fold_id or model.get("paired_seeds") != [11, 23, 37, 53, 71]:
            raise RuntimeError("Model-freeze identity or five-seed contract drifted")
        if model.get("selected_representation_sha256") != source.get("canonical_tensor_artifact", {}).get("sha256"):
            raise RuntimeError("Model freeze is not bound to the selected source representation")
        if not model.get("sampling_contract") or not model.get("sensitivity_scales"):
            raise RuntimeError("Model-freeze sampling/calibration provenance is incomplete")
        for item, role in _model_artifact_records(model):
            verified_paths.append(_verify_record(fold_dir, item, role))
            if item.get("classes_path"):
                class_record = {
                    "path": item["classes_path"],
                    "sha256": item.get("classes_sha256"),
                }
                verified_paths.append(_verify_record(fold_dir, class_record, f"{role}-classes"))
        finite_probe = np.asarray(
            [teacher.get("temperature") for teacher in model.get("teachers", {}).values()], dtype=float
        )
        if finite_probe.size == 0 or not np.all(np.isfinite(finite_probe)):
            raise RuntimeError("Teacher calibration provenance is missing or nonfinite")
        state = _read(fold_dir / "run_state.json")
        if state.get("phase") != "PARETO_FROZEN" or state.get("target_reads", 0) != 0 or state.get("target_unlocks", 0) != 0:
            raise PermissionError("Fold state is not pristine at the target firewall")
        if state.get("source_artifact_hash") != model_record.get("sha256"):
            raise RuntimeError("Run state is not bound to the verified model freeze")
        if (fold_dir / "target_metrics.json").exists() or (fold_dir / "target-cache").exists():
            raise PermissionError("Target-derived fold artifacts already exist")
        verified_paths.extend([str(source_path), str(model_path), str(fold_dir / "run_state.json")])

    return {
        "campaign_path": str(campaign_path),
        "campaign_sha256": sha256_file(campaign_path),
        "label_contract_sha256": preflight.get("label_contract_sha256"),
        "fold_count": 3,
        "datasets": sorted(FROZEN_DATASETS),
        "verified_artifact_count": len(set(verified_paths)),
        "group_leakage_audit": "passed",
        "temporal_grouping_audit": "passed",
        "target_access_count": 0,
    }


def verify_freeze_package(package: Path) -> dict[str, Any]:
    root = Path(package).resolve()
    manifest_path = root / "freeze-manifest.json"
    manifest = _read(manifest_path)
    lines = (root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    expected: dict[str, str] = {}
    for line in lines:
        digest, relative = line.split("  ", 1)
        expected[relative] = digest
    for relative, digest in expected.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError("Freeze manifest path escapes package") from error
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"Freeze package hash drift: {relative}")
    required = {"freeze-manifest.json", "target-access-ledger.json"}
    if not required.issubset(expected):
        raise RuntimeError("Freeze package manifest omits required authorization artifacts")
    return manifest
