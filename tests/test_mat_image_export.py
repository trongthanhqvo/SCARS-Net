from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import pytest

from scars.cli.export_mat_images import run_export
from scars.cli.export_iq_npy_cache import run_iq_cache_export
from scars.data.base_adapter import Recording
from scars.data.windowing import window_recordings
from scars.data.mat_recordings import discover_mat_iq_streams


def _write_iq(path: Path, pairs: tuple[tuple[str, str], ...], samples: int = 512) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    phase = np.linspace(0.0, 12.0 * np.pi, samples, dtype=np.float32)
    with h5py.File(path, "w") as handle:
        for index, (i_key, q_key) in enumerate(pairs, start=1):
            handle.create_dataset(i_key, data=np.cos(index * phase)[None, :])
            handle.create_dataset(q_key, data=np.sin(index * phase)[None, :])


def _write_drff(path: Path, samples: int = 512) -> None:
    _write_iq(path, (("RF0_I", "RF0_Q"),), samples=samples)
    with h5py.File(path, "a") as handle:
        handle.create_dataset("Fs", data=np.asarray([[100e6]], dtype=np.float64))
        handle.create_dataset("CenterFrequence", data=np.asarray([[5.745e9]], dtype=np.float64))
        for key, value in {"TD": "mavic3C_1", "State": "Ascend", "D": "d2", "U": "u1"}.items():
            dataset = handle.create_dataset(
                key, data=np.asarray([ord(character) for character in value], dtype=np.uint16)[:, None]
            )
            dataset.attrs["MATLAB_class"] = np.bytes_("char")


def _make_dataset_root(root: Path) -> Path:
    _write_iq(
        root / "DroneRFa_2024" / "dataset" / "T0000_D00_S0000.mat",
        (("RF0_I", "RF0_Q"), ("RF1_I", "RF1_Q")),
    )
    _write_iq(
        root / "DroneRFa_2024" / "dataset" / "T10000_S0000.mat",
        (("RF0_I", "RF0_Q"), ("RF1_I", "RF1_Q")),
    )
    _write_iq(
        root / "DroneRFb-DIR_2025" / "dataset" / "train" / "A1_IN_S0_slice_1.mat",
        (("I", "Q"),),
    )
    _write_iq(
        root / "DroneRFb-DIR_2025" / "dataset" / "test" / "0.mat",
        (("I", "Q"),),
    )
    labels = root / "DroneRFb-DIR_2025" / "dataset" / "test_labels.txt"
    labels.write_text("A1_IN_S0_slice_47.mat 0\n", encoding="utf-8")
    _write_iq(
        root / "DroneRFb-DIR_2025" / "dataset" / "train" / "background_slice_24.mat",
        (("I", "Q"),),
    )
    _write_drff(
        root
        / "DRFF-R2_2026"
        / "dataset"
        / "dataset1-single_drone_states"
        / "mavic3C_1_Ascend_c17_u1_d2.mat"
    )
    _write_drff(
        root
        / "DRFF-R2_2026"
        / "dataset"
        / "dataset2-drone mixed"
        / "mavic air2s 2 & mavic air2s 3"
        / "mixed_c1.mat"
    )
    _write_drff(
        root
        / "DRFF-R2_2026"
        / "dataset"
        / "dataset7-environment"
        / "environment_c1.mat"
    )
    for subset in ("dataset3-single drone hover", "dataset4-single drone dual frequency"):
        _write_drff(
            root
            / "DRFF-R2_2026"
            / "dataset"
            / subset
            / "mavic3C_1_repeated.mat"
        )
    return root


def test_discovers_dataset_specific_iq_and_test_label(tmp_path: Path):
    root = _make_dataset_root(tmp_path / "datasets")
    streams = discover_mat_iq_streams(root, ["DroneRFa", "DroneRFb-DIR", "DRFF-R2"])
    assert len(streams) == 12
    shortened = next(
        stream
        for stream in streams
        if stream.dataset == "DroneRFa"
        and stream.relative_path.endswith("T10000_S0000.mat")
    )
    assert shortened.label == "DJI_Matrice_600_Pro"
    assert shortened.metadata["distance_code"] is None
    assert shortened.metadata["distance_in_filename"] is False
    test_stream = next(stream for stream in streams if stream.split == "test")
    assert test_stream.label == "A1"
    assert test_stream.model_label == "DJI_Mavic_3_Pro"
    assert test_stream.metadata["original_filename"] == "A1_IN_S0_slice_47.mat"
    assert test_stream.read_iq(10, 64).shape == (64,)
    assert test_stream.read_iq(10, 64).dtype == np.complex64
    background = next(
        stream for stream in streams if stream.relative_path.endswith("background_slice_24.mat")
    )
    assert background.label == "B"
    assert background.model_label == "background"
    mixed = next(stream for stream in streams if stream.task_type == "multilabel")
    assert mixed.emitter_labels == ("mavic_air2s_2", "mavic_air2s_3")
    environment = next(
        stream
        for stream in streams
        if stream.metadata.get("top_level_subset") == "dataset7-environment"
    )
    assert environment.label == "background"


def test_converter_iq_cache_replaces_mat_reads_without_losing_physical_identity(
    tmp_path: Path,
):
    raw_root = _make_dataset_root(tmp_path / "raw")
    original = discover_mat_iq_streams(raw_root, ["DroneRFa"])[0]
    cache_root = tmp_path / "dronerfa-cache"
    manifest = run_iq_cache_export(
        dataset_root=raw_root,
        output_dir=cache_root,
        datasets=["DroneRFa"],
        window_samples=256,
        hop_samples=128,
        max_windows_per_stream=3,
        hash_inputs=False,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "scars-iq-npy-cache-1.0"
    assert payload["recordings"][0]["recording_id"] == original.recording_id

    layout = tmp_path / "cache-layout"
    (layout / "DroneRFa_2024").mkdir(parents=True)
    (layout / "DroneRFa_2024" / "dataset").symlink_to(
        cache_root, target_is_directory=True
    )
    cached = discover_mat_iq_streams(layout, ["DroneRFa"])[0]
    assert cached.recording_id == original.recording_id
    assert cached.path.suffix == ".npy"
    assert cached.metadata["input_storage"] == "npy_complex_iq_window_cache"
    np.testing.assert_array_equal(cached.read_iq(0, 256), original.read_iq(0, 256))

    recording = Recording(
        recording_id=cached.recording_id,
        dataset=cached.dataset,
        path=str(cached.path),
        label="uav",
        is_background=False,
        sample_rate_hz=cached.sample_rate_hz,
        metadata={
            "split_group": cached.group_id,
            "_loader_entry": {
                "sample_count": cached.sample_count,
                "continuity_block_samples": cached.continuity_block_samples,
            },
        },
    )
    batch = window_recordings(
        [recording], window_samples=256, hop_samples=128
    )
    assert batch.iq.shape == (3, 256)
    assert batch.starts.tolist() == [0, 256, 512]


def test_exports_four_images_and_exact_tensor_without_fitting_on_test(tmp_path: Path):
    root = _make_dataset_root(tmp_path / "datasets")
    output = tmp_path / "images"
    summary_path = run_export(
        dataset_root=root,
        output_dir=output,
        datasets=["DroneRFa", "DroneRFb-DIR", "DRFF-R2"],
        source_datasets=["DroneRFa", "DroneRFb-DIR"],
        window_samples=256,
        hop_samples=256,
        fit_windows_per_stream=1,
        max_windows_per_stream=1,
        output_bins=4,
        wst_j=2,
        wst_q=1,
        cyclic_count=2,
        frame_samples=32,
        frame_hop_samples=8,
        preview_scale=2,
        enforce_group_firewall=False,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["exported_window_count"] == 12
    source_fit = json.loads((output / "source_fit.json").read_text(encoding="utf-8"))
    assert all(item["split"] != "test" for item in source_fit["fit_windows"])
    assert len(source_fit["fit_windows"]) <= 64

    first = json.loads((output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    tensor = np.load(output / first["tensor_npy"], allow_pickle=False)
    assert tensor.shape == (4, 4, 4)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()
    for image_path in first["family_png"].values():
        with Image.open(output / image_path) as image:
            assert image.size == (4, 4)
    with Image.open(output / first["rgba_preview"]) as preview:
        assert preview.mode == "RGBA"
        assert preview.size == (8, 8)
    with Image.open(output / first["montage_preview"]) as montage:
        assert montage.mode == "L"
        assert montage.size == (18, 18)
    assert (output / "COMPLETED.json").is_file()


def test_source_recording_manifest_rejects_shared_target_group(tmp_path: Path):
    root = _make_dataset_root(tmp_path / "datasets")
    streams = discover_mat_iq_streams(root, ["DroneRFa", "DroneRFb-DIR", "DRFF-R2"])
    rf0 = next(
        stream for stream in streams if stream.dataset == "DroneRFa" and stream.stream_id == "RF0"
    )
    rf1 = next(
        stream for stream in streams if stream.dataset == "DroneRFa" and stream.stream_id == "RF1"
    )
    contract = tmp_path / "source_contract.json"
    contract.write_text(
        json.dumps(
            {
                "source_recording_ids": [rf0.recording_id],
                "target_recording_ids": [rf1.recording_id],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="group leakage"):
        run_export(
            dataset_root=root,
            output_dir=tmp_path / "rejected",
            datasets=["DroneRFa", "DroneRFb-DIR", "DRFF-R2"],
            source_datasets=["DroneRFa"],
            source_recording_manifest=contract,
            window_samples=256,
            hop_samples=256,
            fit_windows_per_stream=1,
            max_windows_per_stream=1,
            output_bins=4,
            wst_j=2,
            wst_q=1,
            cyclic_count=2,
            frame_samples=32,
            frame_hop_samples=8,
        )


def test_export_can_transform_overlapping_dronerfb_test_groups_when_firewall_disabled(
    tmp_path: Path,
):
    root = _make_dataset_root(tmp_path / "datasets")
    output = tmp_path / "images"
    summary_path = run_export(
        dataset_root=root,
        output_dir=output,
        datasets=["DroneRFb-DIR"],
        source_datasets=["DroneRFb-DIR"],
        source_splits=["train"],
        window_samples=256,
        hop_samples=256,
        fit_windows_per_stream=1,
        max_windows_per_stream=1,
        output_bins=4,
        wst_j=2,
        wst_q=1,
        cyclic_count=2,
        frame_samples=32,
        frame_hop_samples=8,
        enforce_group_firewall=False,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["source_contract"]["group_firewall_enforced"] is False
    assert "A1_S0_IN" in summary["source_contract"]["source_target_group_overlap"]
    source_fit = json.loads((output / "source_fit.json").read_text(encoding="utf-8"))
    assert all(item["split"] == "train" for item in source_fit["fit_windows"])
    assert summary["exported_window_count"] == 3


def test_source_manifest_cannot_bypass_dataset_allowlist(tmp_path: Path):
    root = _make_dataset_root(tmp_path / "datasets")
    streams = discover_mat_iq_streams(root, ["DroneRFa", "DroneRFb-DIR", "DRFF-R2"])
    drff = next(stream for stream in streams if stream.dataset == "DRFF-R2")
    contract = tmp_path / "source_contract.json"
    contract.write_text(
        json.dumps({"source_recording_ids": [drff.recording_id]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="bypasses dataset/split/group allowlist"):
        run_export(
            dataset_root=root,
            output_dir=tmp_path / "rejected_allowlist",
            datasets=["DroneRFa", "DroneRFb-DIR", "DRFF-R2"],
            source_datasets=["DroneRFa"],
            source_recording_manifest=contract,
            window_samples=256,
            max_windows_per_stream=1,
            output_bins=4,
            wst_j=2,
            wst_q=1,
            cyclic_count=2,
            frame_samples=32,
            frame_hop_samples=8,
        )
