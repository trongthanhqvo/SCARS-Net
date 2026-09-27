from __future__ import annotations

import json

import numpy as np
import pytest

from scars.data.manifest import load_manifest, materialize_recording_iq
from scars.data.windowing import window_recordings


def test_mat_manifest_uses_bounded_hyperslabs_and_window_cap(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    mat = dataset_root / "one.mat"
    with h5py.File(mat, "w") as handle:
        handle.create_dataset("I", data=np.arange(1024, dtype=np.float32))
        handle.create_dataset("Q", data=-np.arange(1024, dtype=np.float32))
    manifest = tmp_path / "recordings.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_root_env": "RAW_IQ_DATASET_PATH",
                "recordings": [
                    {
                        "recording_id": "r0",
                        "dataset": "d0",
                        "path": "one.mat",
                        "label": "class0",
                        "is_background": False,
                        "sample_rate_hz": 1.0,
                        "i_key": "I",
                        "q_key": "Q",
                        "sample_count": 1024,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RAW_IQ_DATASET_PATH", str(dataset_root))
    recording = load_manifest(manifest)[0]
    with pytest.raises(MemoryError):
        materialize_recording_iq(recording)
    batch = window_recordings(
        [recording], 128, 64, max_windows_per_recording=3
    )
    assert batch.iq.shape == (3, 128)
    assert batch.starts.tolist() == [0, 448, 896]
