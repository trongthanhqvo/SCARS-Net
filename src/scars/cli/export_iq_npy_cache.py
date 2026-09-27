from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from scars.data.mat_recordings import (
    canonical_dataset_names,
    discover_mat_iq_streams,
    window_starts,
)
from scars.results.provenance import sha256_file


def _safe(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-.")
    return cleaned or "unknown"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_iq_cache_export(
    *,
    dataset_root: Path,
    output_dir: Path,
    datasets: Iterable[str],
    window_samples: int = 4096,
    hop_samples: int = 2048,
    max_windows_per_stream: int = 16,
    hash_inputs: bool = True,
) -> Path:
    """Read each MAT hyperslab once and persist lossless complex64 IQ windows."""
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"IQ-cache output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    names = canonical_dataset_names(datasets)
    streams = discover_mat_iq_streams(dataset_root, names)
    if any(stream.path.suffix.lower() == ".npy" for stream in streams):
        raise ValueError("Refusing to build an IQ cache from an existing IQ cache")

    rows: list[dict[str, object]] = []
    input_hashes: dict[Path, str | None] = {}
    for stream in streams:
        starts = window_starts(
            stream,
            window_samples=window_samples,
            hop_samples=hop_samples,
            max_windows=max_windows_per_stream,
        )
        if starts.size == 0:
            continue
        source_hash = input_hashes.get(stream.path)
        if stream.path not in input_hashes:
            source_hash = sha256_file(stream.path) if hash_inputs else None
            input_hashes[stream.path] = source_hash
        short_id = hashlib.sha256(stream.recording_id.encode("utf-8")).hexdigest()[:12]
        relative = Path(
            "iq_cache",
            _safe(stream.dataset),
            _safe(stream.label),
            f"{_safe(stream.path.stem)}-{_safe(stream.stream_id)}-{short_id}.npy",
        )
        cache_path = output_dir / relative
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache = np.lib.format.open_memmap(
            cache_path,
            mode="w+",
            dtype=np.complex64,
            shape=(len(starts), window_samples),
        )
        for index, start in enumerate(starts):
            cache[index] = stream.read_iq(int(start), window_samples)
        cache.flush()
        del cache
        rows.append(
            {
                "recording_id": stream.recording_id,
                "dataset": stream.dataset,
                "path": str(relative),
                "sha256": sha256_file(cache_path),
                "source_file_sha256": source_hash,
                "source_relative_path": stream.relative_path,
                "stream_id": stream.stream_id,
                "split": stream.split,
                "label": stream.label,
                "model_label": stream.model_label,
                "task_type": stream.task_type,
                "emitter_labels": list(stream.emitter_labels),
                "group_id": stream.group_id,
                "sample_rate_hz": stream.sample_rate_hz,
                "center_frequency_hz": stream.center_frequency_hz,
                "sample_count": int(len(starts) * window_samples),
                "continuity_block_samples": int(window_samples),
                "cached_window_count": int(len(starts)),
                "original_window_starts": [int(value) for value in starts],
                "metadata": stream.metadata,
            }
        )
        print(f"cached {len(starts)} IQ windows from {stream.recording_id}", flush=True)

    if not rows:
        raise ValueError("No complete IQ windows were cached")
    datasets_present = sorted({str(row["dataset"]) for row in rows})
    if len(datasets_present) != 1:
        raise ValueError("Each converter cache must contain exactly one dataset")
    manifest = {
        "schema_version": "scars-iq-npy-cache-1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": datasets_present[0],
        "storage": "lossless_complex64_iq_windows",
        "family_order_note": "W/C/E/S are fitted from this IQ cache source-only per fold",
        "window_samples": window_samples,
        "hop_samples_used_for_selection": hop_samples,
        "max_windows_per_stream": max_windows_per_stream,
        "recordings": rows,
    }
    manifest_path = output_dir / "iq_recordings.json"
    _write_json(manifest_path, manifest)
    _write_json(
        output_dir / "COMPLETED.json",
        {
            "status": "complete",
            "dataset": datasets_present[0],
            "recording_count": len(rows),
            "window_count": sum(int(row["cached_window_count"]) for row in rows),
            "iq_recordings_sha256": sha256_file(manifest_path),
        },
    )
    return manifest_path
