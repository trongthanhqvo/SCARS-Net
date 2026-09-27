from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Any

from scars.data.base_adapter import Recording
from scars.data.mat_recordings import canonical_dataset_names, discover_mat_iq_streams
from scars.data.splits import leave_one_dataset_out
from scars.results.provenance import sha256_file
from scars.selection.nuisance import registered_nuisance_cases


FROZEN_CONFIRMATORY_DATASETS = {"DroneRFa", "DroneRFb-DIR", "DRFF-R2"}


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _hardware() -> dict[str, object]:
    ram_bytes = None
    try:
        import psutil

        ram_bytes = int(psutil.virtual_memory().total)
    except ImportError:
        if hasattr(os, "sysconf"):
            ram_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    gpu: dict[str, object] | None = None
    try:
        import torch

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            gpu = {
                "name": properties.name,
                "vram_bytes": int(properties.total_memory),
                "cuda_runtime": torch.version.cuda,
                "torch": torch.__version__,
            }
    except ImportError:
        pass
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "ram_bytes": ram_bytes,
        "gpu": gpu,
    }


def _audit_hardware(
    hardware: dict[str, object],
    peak_host_estimate: int,
    *,
    pilot_two_dataset: bool,
) -> tuple[list[str], list[str]]:
    """Return blocking issues and provenance warnings for the execution host.

    The frozen three-dataset campaign remains bound to the registered RTX 3060
    Ti / 16 GiB profile.  The explicitly non-confirmatory two-dataset pilot may
    run on a compatible host, while recording profile differences as warnings.
    CUDA availability and the 7 GiB VRAM safety floor remain blocking in both
    modes.
    """

    issues: list[str] = []
    warnings: list[str] = []

    def profile_difference(message: str) -> None:
        (warnings if pilot_two_dataset else issues).append(message)

    ram_bytes = hardware.get("ram_bytes")
    if ram_bytes is not None and int(ram_bytes) < 16 * 1024**3:
        profile_difference(
            "System RAM is below the registered 16 GiB execution profile"
        )
    if ram_bytes is not None and peak_host_estimate > int(0.60 * int(ram_bytes)):
        profile_difference(
            "Estimated peak host memory exceeds 60% of installed RAM; "
            "monitor host memory and reduce max-windows-per-recording if needed"
        )

    gpu = hardware.get("gpu")
    if gpu is None:
        issues.append("CUDA GPU is unavailable")
    elif int(gpu["vram_bytes"]) < 7 * 1024**3:
        issues.append("CUDA VRAM is below the 7 GiB safe envelope")
    elif "3060ti" not in str(gpu["name"]).lower().replace(" ", ""):
        profile_difference(
            f"Registered hardware is RTX 3060 Ti; detected {gpu['name']}"
        )
    return issues, warnings


def _load_label_contract(path: Path | None) -> tuple[dict[str, Any], str | None]:
    if path is None:
        return {}, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, sha256_file(path)


def _canonical_label(stream, contract: dict[str, Any]) -> str | None:
    if stream.label == "background" or stream.model_label == "background":
        return "background"
    dataset = contract.get("datasets", {}).get(stream.dataset, {})
    if stream.dataset == "DRFF-R2" and stream.task_type == "single_label":
        subset = stream.metadata.get("top_level_subset", "")
        if subset in dataset.get("binary_uav_subsets", []):
            return "uav"
    mapping = dataset.get("labels", {})
    return mapping.get(stream.label, mapping.get(stream.model_label))


def _portable_record(
    stream,
    canonical_label: str,
    checksum: str | None,
    temporal_grouping_verified: bool,
) -> dict[str, Any]:
    directory = {
        "DroneRFa": "DroneRFa_2024",
        "DroneRFb-DIR": "DroneRFb-DIR_2025",
        "DRFF-R2": "DRFF-R2_2026",
    }[stream.dataset]
    return {
        "recording_id": stream.recording_id,
        "dataset": stream.dataset,
        "path": str(Path(directory) / stream.relative_path),
        "label": canonical_label,
        "original_label": stream.label,
        "model_label": stream.model_label,
        "is_background": canonical_label == "background",
        "sample_rate_hz": stream.sample_rate_hz,
        "center_frequency_hz": stream.center_frequency_hz,
        "split_group": stream.group_id,
        "array_key": None,
        "i_key": stream.i_key,
        "q_key": stream.q_key,
        "sample_count": stream.sample_count,
        "continuity_block_samples": stream.continuity_block_samples,
        "sha256": checksum,
        "metadata": {
            "split_group": stream.group_id,
            "original_split": stream.split,
            "task_type": stream.task_type,
            "input_storage": stream.metadata.get("input_storage", "mat_hdf5"),
            "original_window_starts": stream.metadata.get("original_window_starts", []),
            "source_file_sha256": stream.metadata.get("source_file_sha256"),
            "temporal_adjacency_subsumed_by_split_group_verified": temporal_grouping_verified,
        },
    }


def _split_payload(folds) -> dict[str, object]:
    output = []
    for fold in folds:
        output.append(
            {
                "fold_id": fold.fold_id,
                **{
                    name: [record.recording_id for record in records]
                    for name, records in (*fold.source_roles.items(), ("held_target", fold.held_target))
                },
                "coverage": fold.coverage,
            }
        )
    return {"schema_version": "scars-splits-2.0", "folds": output}


def run_preflight(
    *,
    dataset_root: Path,
    output_dir: Path,
    label_contract: Path | None,
    datasets: list[str],
    hash_files: bool,
    window_samples: int,
    hop_samples: int,
    max_windows_per_recording: int,
    exclusion_manifest: Path | None = None,
    pilot_two_dataset: bool = False,
    pilot_three_dataset: bool = False,
) -> Path:
    if pilot_two_dataset and pilot_three_dataset:
        raise ValueError("Select only one pilot mode")
    if output_dir.exists() and any(output_dir.iterdir()) and not pilot_two_dataset:
        raise FileExistsError("Preflight output directory must be new and empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    names = canonical_dataset_names(datasets)
    contract, contract_sha256 = _load_label_contract(label_contract)
    exclusion_payload = (
        json.loads(exclusion_manifest.read_text(encoding="utf-8"))
        if exclusion_manifest is not None
        else {"recording_ids": []}
    )
    excluded_ids = {str(item) for item in exclusion_payload.get("recording_ids", [])}
    excluded_drff = tuple(exclusion_payload.get("drff_relative_paths", []))
    if excluded_drff and not pilot_three_dataset:
        raise ValueError("DRFF path exclusions are registered only for the three-dataset pilot")
    for path in excluded_drff:
        if not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts or not path.endswith(".mat"):
            raise ValueError("Invalid DRFF relative exclusion path")
    streams = discover_mat_iq_streams(dataset_root, names, excluded_drff_paths=excluded_drff)
    exclusion_sha256 = (
        sha256_file(exclusion_manifest) if exclusion_manifest is not None else None
    )
    issues: list[str] = []
    campaign_mode = "pilot_three_dataset" if pilot_three_dataset else ("pilot_two_dataset" if pilot_two_dataset else "real_confirmatory")
    if pilot_two_dataset:
        if set(names) != {"DroneRFa", "DroneRFb-DIR"} or len(names) != 2:
            issues.append("Two-dataset pilot requires exactly DroneRFa and DroneRFb-DIR")
    elif set(names) != FROZEN_CONFIRMATORY_DATASETS or len(names) != 3:
        issues.append("Confirmatory campaign requires exactly DroneRFa, DroneRFb-DIR, and DRFF-R2")
    warnings: list[str] = []
    contract_datasets = contract.get("datasets", {})
    for dataset in names:
        record = contract_datasets.get(dataset, {})
        if not record.get("license_verified"):
            issues.append(f"{dataset}: license_verified is not true")
        if not record.get("derived_tensor_release_reviewed"):
            warnings.append(
                f"{dataset}: derived tensors must remain private until release rights are reviewed"
            )
        if not record.get("temporal_adjacency_subsumed_by_split_group_verified"):
            issues.append(
                f"{dataset}: temporal adjacency is not verified as subsumed by split_group"
            )

    file_hashes: dict[Path, str | None] = {}
    records_by_dataset: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    missing_labels: set[str] = set()
    for stream in streams:
        if stream.recording_id in excluded_ids:
            continue
        if stream.task_type != "single_label":
            continue
        canonical = _canonical_label(stream, contract)
        if canonical is None:
            missing_labels.add(f"{stream.dataset}:{stream.label}")
            continue
        if stream.path not in file_hashes:
            file_hashes[stream.path] = sha256_file(stream.path) if hash_files else None
        records_by_dataset[stream.dataset].append(
            _portable_record(
                stream,
                canonical,
                file_hashes[stream.path],
                bool(
                    contract_datasets.get(stream.dataset, {}).get(
                        "temporal_adjacency_subsumed_by_split_group_verified"
                    )
                ),
            )
        )
    if missing_labels:
        issues.append(
            "Missing exact ontology mappings: " + ", ".join(sorted(missing_labels)[:25])
        )
    label_sets = [
        {record["label"] for record in records}
        for records in records_by_dataset.values()
        if records
    ]
    shared = set.intersection(*label_sets) if len(label_sets) == len(names) else set()
    if len(shared) < 2:
        issues.append(
            "Multiclass recognition needs at least two exact canonical labels shared by all datasets"
        )
    if "background" not in shared:
        warnings.append(
            "Detection arm ineligible: canonical background is not shared by all datasets; recognition may proceed and detection manuscript fields remain TBD"
        )
    if not hash_files:
        issues.append("Confirmatory execution requires complete SHA-256 content hashes")
    recognition_rows = [
        record
        for records in records_by_dataset.values()
        for record in records
        if record["label"] in shared
    ]
    folds = []
    if len(shared) >= 2:
        metadata_records = [
            Recording(
                recording_id=row["recording_id"],
                dataset=row["dataset"],
                path=row["path"],
                label=row["label"],
                is_background=row["is_background"],
                sample_rate_hz=row["sample_rate_hz"],
                center_frequency_hz=row["center_frequency_hz"],
                metadata={"split_group": row["split_group"]},
            )
            for row in recognition_rows
        ]
        try:
            folds = leave_one_dataset_out(metadata_records, seed=24021,
                                         sparse_drff_background=pilot_three_dataset)
        except ValueError as error:
            issues.append(str(error))

    total_windows = 0
    for row in recognition_rows:
        available = max(0, 1 + (int(row["sample_count"]) - window_samples) // hop_samples)
        total_windows += min(available, max_windows_per_recording)
    per_window_bytes = window_samples * 8 + 4 * 16 * 16 * 4
    nuisance_count = len(registered_nuisance_cases())
    relation_window_count = len(recognition_rows) * nuisance_count
    relation_bytes = relation_window_count * (
        2 * window_samples * 8 + 2 * 4 * 16 * 16 * 4
    )
    cached_upper_bound = total_windows * per_window_bytes
    peak_host_estimate = cached_upper_bound + relation_bytes
    hardware = _hardware()
    hardware_issues, hardware_warnings = _audit_hardware(
        hardware,
        peak_host_estimate,
        pilot_two_dataset=pilot_two_dataset or pilot_three_dataset,
    )
    issues.extend(hardware_issues)
    warnings.extend(hardware_warnings)

    manifest_payload = {
        "schema_version": "scars-recordings-2.0",
        "campaign_mode": campaign_mode,
        "dataset_root_env": "RAW_IQ_DATASET_PATH",
        "label_contract_sha256": contract_sha256,
        "confirmatory_exclusion_manifest_sha256": exclusion_sha256,
        "temporal_grouping_proof": {
            dataset: bool(
                contract_datasets.get(dataset, {}).get(
                    "temporal_adjacency_subsumed_by_split_group_verified"
                )
            )
            for dataset in names
        },
        "excluded_recording_ids": sorted(excluded_ids),
        "excluded_drff_relative_paths": list(excluded_drff),
        "path_exclusion_reason": exclusion_payload.get("reason"),
        "recordings": recognition_rows,
    }
    _write_json(output_dir / "recordings_recognition.json", manifest_payload)
    split_payload = _split_payload(folds)
    _write_json(output_dir / "splits.json", split_payload)
    preflight = {
        "schema_version": "scars-preflight-2.0",
        "campaign_mode": campaign_mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready" if not issues else "blocked",
        "confirmatory_target_access_authorized": False,
        "datasets": list(names),
        "discovered_stream_count": len(streams),
        "eligible_recognition_recordings": len(recognition_rows),
        "shared_canonical_labels": sorted(shared),
        "recognition_eligible": len(shared) >= 2,
        "detection_eligible": "background" in shared,
        "fold_count": len(folds),
        "content_hashes_complete": hash_files,
        "label_contract_sha256": contract_sha256,
        "confirmatory_exclusion_manifest_sha256": exclusion_sha256,
        "excluded_discovered_stream_count": sum(
            stream.recording_id in excluded_ids for stream in streams
        ),
        "excluded_drff_relative_paths": list(excluded_drff),
        "excluded_drff_existing_file_count": sum(
            (dataset_root / "DRFF-R2_2026" / "dataset" / path).is_file() for path in excluded_drff
        ),
        "issues": issues,
        "warnings": warnings,
        "hardware": hardware,
        "hardware_gate_policy": (
            "compatible_pilot_host_with_profile_differences_as_warnings"
            if pilot_two_dataset or pilot_three_dataset
            else "strict_registered_confirmatory_profile"
        ),
        "memory_profile": {
            "window_samples": window_samples,
            "hop_samples": hop_samples,
            "max_windows_per_recording": max_windows_per_recording,
            "estimated_cached_window_count": total_windows,
            "estimated_raw_plus_tensor_bytes_per_fold_upper_bound": cached_upper_bound,
            "registered_nuisance_case_count": nuisance_count,
            "estimated_relation_expansion_bytes_upper_bound": relation_bytes,
            "estimated_peak_host_bytes_upper_bound": peak_host_estimate,
            "host_ram_gate_fraction": 0.60,
            "training_micro_batch": 32,
            "effective_batch": 64,
            "minimum_oom_backoff_batch": 4,
            "amp": True,
            "folds_and_models_sequential": True,
        },
        "artifacts": {
            "recordings_manifest": "recordings_recognition.json",
            "split_manifest": "splits.json",
        },
    }
    _write_json(output_dir / "preflight.json", preflight)
    return output_dir / "preflight.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed preflight for the three local MATLAB UAV-RF corpora")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-contract", type=Path)
    parser.add_argument(
        "--exclusion-manifest",
        type=Path,
        default=Path("configs/confirmatory_exclusions.json"),
        help="Recording IDs exposed during development and forbidden as confirmatory targets",
    )
    parser.add_argument("--datasets", nargs="+", default=["DroneRFa", "DroneRFb-DIR", "DRFF-R2"])
    parser.add_argument("--pilot-three-dataset", action="store_true",
                        help="Explicit three-dataset exploratory amendment with sparse DRFF background roles")
    parser.add_argument(
        "--pilot-two-dataset",
        action="store_true",
        help="Run a non-confirmatory DroneRFa/DroneRFb-DIR pilot instead of the frozen three-dataset campaign.",
    )
    parser.add_argument("--no-hash-files", dest="hash_files", action="store_false")
    parser.set_defaults(hash_files=True)
    parser.add_argument("--window-samples", type=int, default=4096)
    parser.add_argument("--hop-samples", type=int, default=2048)
    parser.add_argument("--max-windows-per-recording", type=int, default=64)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    path = run_preflight(**vars(arguments))
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(path)
    print(json.dumps({"status": payload["status"], "issues": payload["issues"]}, indent=2))
    return 0 if payload["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
