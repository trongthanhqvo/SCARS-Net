from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import numpy as np
from PIL import Image

from scars.data.mat_recordings import (
    MatIQStream,
    canonical_dataset_names,
    discover_mat_iq_streams,
    window_starts,
)
from scars.representations.tensor import RepresentationConfig, SourceFittedTensor
from scars.results.provenance import environment_manifest, sha256_file


FAMILY_ORDER = ("W", "C", "E", "S")


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-.")
    return cleaned or "unknown"


def _recording_output_component(stream: MatIQStream) -> str:
    short_id = hashlib.sha256(stream.recording_id.encode("utf-8")).hexdigest()[:12]
    return _safe_component(f"{stream.path.stem}-{stream.stream_id}-{short_id}")


def _sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _scale_for_png(values: np.ndarray, low: float, high: float, bits: int) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError("Representation contains NaN or Inf")
    span = max(float(high - low), 1.0e-12)
    unit = np.clip((np.asarray(values, dtype=np.float64) - low) / span, 0.0, 1.0)
    maximum = (1 << bits) - 1
    dtype = np.uint16 if bits == 16 else np.uint8
    return np.rint(unit * maximum).astype(dtype)


def _save_family_png(path: Path, values: np.ndarray, low: float, high: float) -> None:
    image = Image.fromarray(_scale_for_png(values, low, high, bits=16), mode="I;16")
    image.save(path, format="PNG")


def _save_rgba_preview(
    path: Path,
    family_maps: dict[str, np.ndarray],
    display_ranges: dict[str, tuple[float, float]],
    preview_scale: int,
) -> None:
    channels = []
    for family in FAMILY_ORDER:
        low, high = display_ranges[family]
        channels.append(_scale_for_png(family_maps[family], low, high, bits=8))
    rgba = np.stack(channels, axis=-1)
    image = Image.fromarray(rgba, mode="RGBA")
    if preview_scale > 1:
        image = image.resize(
            (image.width * preview_scale, image.height * preview_scale),
            resample=Image.Resampling.NEAREST,
        )
    image.save(path, format="PNG")


def _save_montage_preview(
    path: Path,
    family_maps: dict[str, np.ndarray],
    display_ranges: dict[str, tuple[float, float]],
    preview_scale: int,
) -> None:
    tiles = []
    for family in FAMILY_ORDER:
        low, high = display_ranges[family]
        tile = Image.fromarray(_scale_for_png(family_maps[family], low, high, bits=8), mode="L")
        tile = tile.resize(
            (tile.width * preview_scale, tile.height * preview_scale),
            resample=Image.Resampling.NEAREST,
        )
        tiles.append(tile)
    gap = preview_scale
    canvas = Image.new("L", (2 * tiles[0].width + gap, 2 * tiles[0].height + gap), color=0)
    positions = (
        (0, 0),
        (tiles[0].width + gap, 0),
        (0, tiles[0].height + gap),
        (tiles[0].width + gap, tiles[0].height + gap),
    )
    for tile, position in zip(tiles, positions):
        canvas.paste(tile, position)
    canvas.save(path, format="PNG")


def _fit_starts(stream: MatIQStream, window_samples: int, count: int) -> np.ndarray:
    return window_starts(
        stream,
        window_samples=window_samples,
        hop_samples=window_samples,
        max_windows=count,
    )


def _collect_fit_windows(
    streams: Iterable[MatIQStream],
    source_recording_ids: set[str],
    window_samples: int,
    fit_windows_per_stream: int,
    max_fit_windows: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    candidates: list[tuple[MatIQStream, int]] = []
    for stream in streams:
        if stream.recording_id not in source_recording_ids:
            continue
        candidates.extend(
            (stream, int(start))
            for start in _fit_starts(stream, window_samples, fit_windows_per_stream)
        )
    if max_fit_windows <= 0:
        raise ValueError("max_fit_windows must be positive")
    if len(candidates) > max_fit_windows:
        chosen = np.linspace(0, len(candidates) - 1, max_fit_windows, dtype=np.int64)
        candidates = [candidates[int(index)] for index in chosen]

    windows: list[np.ndarray] = []
    recording_ids: list[str] = []
    provenance: list[dict[str, Any]] = []
    for stream, start in candidates:
        windows.append(stream.read_iq(start, window_samples))
        recording_ids.append(stream.recording_id)
        provenance.append(
            {
                "recording_id": stream.recording_id,
                "group_id": stream.group_id,
                "start_sample": start,
                "stop_sample": int(start + window_samples),
                "dataset": stream.dataset,
                "split": stream.split,
            }
        )
    if not windows:
        raise ValueError(
            "No source-fit windows matched --source-datasets and --source-splits; "
            "the representation cannot be fitted on target data implicitly."
        )
    return (
        np.stack(windows).astype(np.complex64),
        np.asarray(recording_ids, dtype=object),
        provenance,
    )


def _resolve_source_contract(
    streams: list[MatIQStream],
    source_datasets: set[str],
    source_splits: set[str],
    source_recording_manifest: Path | None,
    exclude_group_ids: set[str],
    enforce_group_firewall: bool = True,
) -> tuple[set[str], dict[str, Any]]:
    by_id = {stream.recording_id: stream for stream in streams}
    manifest_payload: dict[str, Any] | None = None
    manifest_hash: str | None = None
    if source_recording_manifest is not None:
        source_recording_manifest = source_recording_manifest.resolve()
        manifest_payload = json.loads(source_recording_manifest.read_text(encoding="utf-8"))
        if not isinstance(manifest_payload, dict):
            raise ValueError("Source recording manifest must be a JSON object")
        if not isinstance(manifest_payload.get("source_recording_ids"), list):
            raise ValueError("Source recording manifest requires source_recording_ids[]")
        source_ids = set(map(str, manifest_payload["source_recording_ids"]))
        manifest_hash = sha256_file(source_recording_manifest)
    else:
        source_ids = {
            stream.recording_id
            for stream in streams
            if stream.dataset in source_datasets
            and stream.split in source_splits
            and stream.group_id not in exclude_group_ids
        }
    unknown = source_ids - set(by_id)
    if unknown:
        raise ValueError(f"Unknown source recording IDs: {sorted(unknown)}")
    if not source_ids:
        raise ValueError("Source recording selection is empty")

    disallowed_sources = {
        recording_id
        for recording_id in source_ids
        if by_id[recording_id].dataset not in source_datasets
        or by_id[recording_id].split not in source_splits
        or by_id[recording_id].group_id in exclude_group_ids
    }
    if disallowed_sources:
        raise ValueError(
            "Source recording manifest bypasses dataset/split/group allowlist: "
            f"{sorted(disallowed_sources)}"
        )

    source_groups = {by_id[recording_id].group_id for recording_id in source_ids}
    # Closed-world rule: every exported recording that is not explicitly a
    # source is a target. This prevents an omitted same-group stream from
    # silently escaping the source/target leakage check.
    target_ids = set(by_id) - source_ids
    declared_target_groups = set(exclude_group_ids)
    if manifest_payload is not None:
        explicitly_declared_targets = set(
            map(str, manifest_payload.get("target_recording_ids", []))
        )
        unknown_targets = explicitly_declared_targets - set(by_id)
        if unknown_targets:
            raise ValueError(f"Unknown target recording IDs: {sorted(unknown_targets)}")
        target_ids.update(explicitly_declared_targets)
        declared_target_groups.update(map(str, manifest_payload.get("target_group_ids", [])))
        declared_source_groups = set(map(str, manifest_payload.get("source_group_ids", [])))
        if declared_source_groups and declared_source_groups != source_groups:
            raise ValueError(
                "source_group_ids do not exactly match groups implied by source_recording_ids"
            )
    target_groups = declared_target_groups | {by_id[item].group_id for item in target_ids}
    overlap = source_groups & target_groups
    if enforce_group_firewall and overlap:
        raise ValueError(f"Source/target group leakage in source contract: {sorted(overlap)}")
    if source_ids & target_ids:
        raise ValueError("A recording cannot be both source and target")
    return source_ids, {
        "selection_mode": "recording_manifest" if manifest_payload is not None else "dataset_split",
        "source_recording_manifest": None
        if source_recording_manifest is None
        else str(source_recording_manifest),
        "source_recording_manifest_sha256": manifest_hash,
        "source_recording_ids": sorted(source_ids),
        "source_group_ids": sorted(source_groups),
        "target_recording_ids": sorted(target_ids),
        "target_group_ids": sorted(target_groups),
        "exclude_group_ids": sorted(exclude_group_ids),
        "group_firewall_enforced": bool(enforce_group_firewall),
        "source_target_group_overlap": sorted(overlap),
    }


def _display_ranges(
    representation: SourceFittedTensor, fit_windows: np.ndarray
) -> dict[str, tuple[float, float]]:
    transformed = representation.transform_by_family(fit_windows)
    ranges: dict[str, tuple[float, float]] = {}
    for family in FAMILY_ORDER:
        values = transformed[family]
        if family in {"W", "C", "S"} and representation.config.normalization == "percentile":
            low, high = 0.0, 1.0
        else:
            low, high = np.percentile(values, [1.0, 99.0])
            if high <= low:
                high = low + 1.0
        ranges[family] = (float(low), float(high))
    return ranges


def _ensure_new_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use a new directory to keep provenance immutable."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def run_export(
    *,
    dataset_root: Path,
    output_dir: Path,
    datasets: Iterable[str],
    source_datasets: Iterable[str],
    source_splits: Iterable[str] = ("train", "unspecified"),
    source_recording_manifest: Path | None = None,
    exclude_group_ids: Iterable[str] = (),
    window_samples: int = 4096,
    hop_samples: int = 4096,
    fit_windows_per_stream: int = 4,
    max_fit_windows: int = 64,
    max_windows_per_stream: int = 0,
    output_bins: int = 16,
    wst_j: int = 4,
    wst_q: int = 2,
    cyclic_count: int = 8,
    frame_samples: int = 128,
    frame_hop_samples: int = 32,
    normalization: str = "percentile",
    preview_scale: int = 16,
    save_npy: bool = True,
    hash_inputs: bool = True,
    enforce_group_firewall: bool = True,
) -> Path:
    if fit_windows_per_stream <= 0:
        raise ValueError("fit_windows_per_stream must be positive")
    if preview_scale <= 0:
        raise ValueError("preview_scale must be positive")
    output_dir = output_dir.resolve()
    _ensure_new_output_dir(output_dir)

    selected = canonical_dataset_names(datasets)
    sources = set(canonical_dataset_names(source_datasets))
    unknown_sources = sources - set(selected)
    if unknown_sources:
        raise ValueError(f"Source datasets are not selected for export: {sorted(unknown_sources)}")
    source_split_set = set(source_splits)
    streams = discover_mat_iq_streams(dataset_root, selected)
    source_recording_ids, source_contract = _resolve_source_contract(
        streams,
        sources,
        source_split_set,
        source_recording_manifest,
        set(exclude_group_ids),
        enforce_group_firewall,
    )

    fit_windows, fit_recording_ids, fit_provenance = _collect_fit_windows(
        streams,
        source_recording_ids,
        window_samples,
        fit_windows_per_stream,
        max_fit_windows,
    )
    config = RepresentationConfig(
        stable_id="W+C+E+S_mat_image_export",
        use_w=True,
        use_c=True,
        use_e=True,
        use_s=True,
        output_bins=output_bins,
        wst_j=wst_j,
        wst_q=wst_q,
        cyclic_count=cyclic_count,
        frame_samples=frame_samples,
        hop_samples=frame_hop_samples,
        normalization=normalization,
    )
    representation = SourceFittedTensor(config).fit(
        fit_windows,
        source_fold="mat-image-export-source",
        fit_split_kind="source_fit",
        recording_ids=fit_recording_ids,
    )
    display_ranges = _display_ranges(representation, fit_windows)
    source_fit_payload = {
        "artifact_type": "source_fitted_scars_representation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_datasets": sorted(sources),
        "source_splits": sorted(source_split_set),
        "source_contract": source_contract,
        "fit_windows": fit_provenance,
        "fit_window_sampling_rule": (
            "up to fit_windows_per_stream evenly spaced windows per selected stream, then "
            "up to max_fit_windows evenly spaced candidates in sorted stream order"
        ),
        "representation": representation.source_artifact(),
        "display_ranges_for_png_only": {
            family: {"low": bounds[0], "high": bounds[1]}
            for family, bounds in display_ranges.items()
        },
        "warning": (
            "PNG values are display encodings. tensor.npy is the canonical float32 SCARS tensor; "
            "the E channel remains unclipped there."
        ),
    }
    _write_json(output_dir / "source_fit.json", source_fit_payload)

    project_root = Path(__file__).resolve().parents[3]
    run_provenance = environment_manifest(project_root, [])
    run_provenance.update(
        {
            "argv": list(sys.argv),
            "source_fit_sha256": sha256_file(output_dir / "source_fit.json"),
        }
    )
    _write_json(output_dir / "run_provenance.json", run_provenance)

    input_hashes: dict[Path, str | None] = {}
    manifest_path = output_dir / "manifest.jsonl"
    exported = 0
    stream_summaries = []
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for stream in streams:
            starts = window_starts(
                stream,
                window_samples=window_samples,
                hop_samples=hop_samples,
                max_windows=max_windows_per_stream,
            )
            if hash_inputs and stream.path not in input_hashes:
                input_hashes[stream.path] = _sha256(stream.path)
            elif stream.path not in input_hashes:
                input_hashes[stream.path] = None
            stream_count = 0
            for window_index, start in enumerate(starts):
                iq = stream.read_iq(int(start), window_samples)
                family_maps = representation.transform_one_by_family(iq)
                tensor = np.stack([family_maps[family] for family in FAMILY_ORDER]).astype(
                    np.float32
                )
                relative_dir = Path(
                    _safe_component(stream.dataset),
                    _safe_component(stream.label),
                    _recording_output_component(stream),
                    f"w{window_index:07d}-s{int(start):012d}",
                )
                sample_dir = output_dir / relative_dir
                sample_dir.mkdir(parents=True, exist_ok=False)
                family_files: dict[str, str] = {}
                for family in FAMILY_ORDER:
                    filename = f"{family}.png"
                    low, high = display_ranges[family]
                    _save_family_png(sample_dir / filename, family_maps[family], low, high)
                    family_files[family] = str(relative_dir / filename)
                _save_rgba_preview(
                    sample_dir / "SCARS_RGBA.png",
                    family_maps,
                    display_ranges,
                    preview_scale,
                )
                _save_montage_preview(
                    sample_dir / "SCARS_MONTAGE.png",
                    family_maps,
                    display_ranges,
                    preview_scale,
                )
                tensor_path: str | None = None
                if save_npy:
                    np.save(sample_dir / "tensor.npy", tensor, allow_pickle=False)
                    tensor_path = str(relative_dir / "tensor.npy")
                record = {
                    "schema_version": "1.0",
                    "dataset": stream.dataset,
                    "recording_id": stream.recording_id,
                    "source_path": str(stream.path),
                    "source_relative_path": stream.relative_path,
                    "source_file_sha256": input_hashes[stream.path],
                    "stream_id": stream.stream_id,
                    "split": stream.split,
                    "label": stream.label,
                    "model_label": stream.model_label,
                    "task_type": stream.task_type,
                    "emitter_labels": list(stream.emitter_labels),
                    "group_id": stream.group_id,
                    "window_index": int(window_index),
                    "start_sample": int(start),
                    "stop_sample": int(start + window_samples),
                    "sample_rate_hz": stream.sample_rate_hz,
                    "center_frequency_hz": stream.center_frequency_hz,
                    "duration_sec": float(window_samples / stream.sample_rate_hz),
                    "tensor_shape": list(tensor.shape),
                    "tensor_dtype": str(tensor.dtype),
                    "family_order": list(FAMILY_ORDER),
                    "family_png": family_files,
                    "rgba_preview": str(relative_dir / "SCARS_RGBA.png"),
                    "montage_preview": str(relative_dir / "SCARS_MONTAGE.png"),
                    "tensor_npy": tensor_path,
                    "channel_min_max": {
                        family: {
                            "min": float(np.min(family_maps[family])),
                            "max": float(np.max(family_maps[family])),
                        }
                        for family in FAMILY_ORDER
                    },
                    "metadata": stream.metadata,
                }
                manifest.write(
                    json.dumps(record, sort_keys=True, allow_nan=False, default=_json_default)
                    + "\n"
                )
                exported += 1
                stream_count += 1
            stream_summaries.append(
                {
                    "recording_id": stream.recording_id,
                    "available_samples": stream.sample_count,
                    "exported_windows": stream_count,
                }
            )
            print(f"exported {stream_count} windows from {stream.recording_id}", flush=True)

    summary = {
        "artifact_type": "scars_mat_image_export",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root.resolve()),
        "output_dir": str(output_dir),
        "datasets": list(selected),
        "source_datasets": sorted(sources),
        "source_splits": sorted(source_split_set),
        "source_contract": source_contract,
        "all_selected_datasets_used_for_fit": source_recording_ids
        == {stream.recording_id for stream in streams},
        "confirmatory_evidence": False,
        "warning": (
            "This is a deterministic representation export, not a held-domain result. "
            "For confirmatory evaluation, fit only on pre-registered source recordings and "
            "keep held-target images inaccessible until the target gate opens."
        ),
        "configuration": {
            **asdict(config),
            "window_samples": window_samples,
            "window_hop_samples": hop_samples,
            "fit_windows_per_stream": fit_windows_per_stream,
            "max_fit_windows": max_fit_windows,
            "max_windows_per_stream": max_windows_per_stream,
            "preview_scale": preview_scale,
            "save_npy": save_npy,
            "hash_inputs": hash_inputs,
        },
        "stream_count": len(streams),
        "exported_window_count": exported,
        "streams": stream_summaries,
        "files": {
            "manifest": "manifest.jsonl",
            "source_fit": "source_fit.json",
            "run_provenance": "run_provenance.json",
        },
    }
    _write_json(output_dir / "summary.json", summary)
    completed = {
        "status": "complete",
        "exported_window_count": exported,
        "artifact_sha256": {
            "manifest.jsonl": sha256_file(manifest_path),
            "source_fit.json": sha256_file(output_dir / "source_fit.json"),
            "run_provenance.json": sha256_file(output_dir / "run_provenance.json"),
            "summary.json": sha256_file(output_dir / "summary.json"),
        },
    }
    _write_json(output_dir / "COMPLETED.json", completed)
    return output_dir / "summary.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read DroneRFa, DroneRFb-DIR, and DRFF-R2 MATLAB IQ files lazily and "
            "export the source-fitted four-channel SCARS representation as images."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["DroneRFa", "DroneRFb-DIR", "DRFF-R2"],
    )
    parser.add_argument(
        "--source-datasets",
        nargs="+",
        required=True,
        help="Datasets allowed to fit cyclic bins and normalizers; held targets must be omitted.",
    )
    parser.add_argument(
        "--source-splits",
        nargs="+",
        default=["train", "unspecified"],
        help="DroneRFb-DIR test is excluded from fitting by default.",
    )
    parser.add_argument(
        "--source-recording-manifest",
        type=Path,
        help=(
            "Optional frozen JSON with source_recording_ids and optional source_group_ids, "
            "target_recording_ids, and target_group_ids."
        ),
    )
    parser.add_argument(
        "--exclude-group-ids",
        nargs="*",
        default=[],
        help="Groups forbidden from source fitting; use for held session/receiver/band folds.",
    )
    parser.add_argument("--window-samples", type=int, default=4096)
    parser.add_argument("--hop-samples", type=int, default=4096)
    parser.add_argument("--fit-windows-per-stream", type=int, default=4)
    parser.add_argument("--max-fit-windows", type=int, default=64)
    parser.add_argument(
        "--max-windows-per-stream",
        type=int,
        default=0,
        help="0 exports every valid window; use a small positive value for a bounded preview.",
    )
    parser.add_argument("--output-bins", type=int, default=16)
    parser.add_argument("--wst-j", type=int, default=4)
    parser.add_argument("--wst-q", type=int, default=2)
    parser.add_argument("--cyclic-count", type=int, default=8)
    parser.add_argument("--frame-samples", type=int, default=128)
    parser.add_argument("--frame-hop-samples", type=int, default=32)
    parser.add_argument(
        "--normalization",
        choices=["percentile", "zscore", "none"],
        default="percentile",
    )
    parser.add_argument("--preview-scale", type=int, default=16)
    parser.add_argument("--no-save-npy", dest="save_npy", action="store_false")
    parser.set_defaults(hash_inputs=True)
    parser.add_argument("--hash-inputs", dest="hash_inputs", action="store_true")
    parser.add_argument("--no-hash-inputs", dest="hash_inputs", action="store_false")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    print(run_export(**vars(arguments)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
