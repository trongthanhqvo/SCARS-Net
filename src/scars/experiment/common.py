from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scars.data.base_adapter import Recording
from scars.data.windowing import WindowBatch
from scars.results.provenance import environment_manifest


def atomic_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return hashlib.sha256(encoded).hexdigest()


def load_split_plan(path: Path, recordings: list[Recording]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    lookup = {record.recording_id: record for record in recordings}
    output = []
    for fold in payload["folds"]:
        materialized = {"fold_id": fold["fold_id"], "coverage": fold.get("coverage", {})}
        for role in (
            "source_fit",
            "source_calibration",
            "source_selection",
            "source_validation",
            "held_target",
        ):
            missing = [value for value in fold[role] if value not in lookup]
            if missing:
                raise ValueError(f"Split references unknown recordings: {missing[:3]}")
            materialized[role] = [lookup[value] for value in fold[role]]
        output.append(materialized)
    return output


def one_window_per_recording(batch: WindowBatch) -> WindowBatch:
    indices = np.asarray(
        [np.flatnonzero(batch.recording_ids == recording)[0] for recording in sorted(set(batch.recording_ids), key=str)]
    )
    return WindowBatch(
        iq=batch.iq[indices],
        labels=batch.labels[indices],
        recording_ids=batch.recording_ids[indices],
        domains=batch.domains[indices],
        starts=batch.starts[indices],
        ends=batch.ends[indices],
    )


def flatten(tensor: np.ndarray) -> np.ndarray:
    return np.asarray(tensor).reshape(len(tensor), -1)


def assert_frozen_environment(campaign: dict[str, Any], project_root: Path) -> None:
    config_paths = sorted((project_root / "configs").glob("*.yaml")) + sorted(
        (project_root / "configs").glob("*.json")
    )
    current = environment_manifest(project_root, config_paths)
    frozen = campaign["provenance"]
    for key in ("source_tree_sha256", "combined_config_hash", "packages", "platform", "hardware"):
        if current.get(key) != frozen.get(key):
            raise RuntimeError(f"Frozen execution environment drifted at {key}")
