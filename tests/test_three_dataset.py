import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scars.data.base_adapter import Recording
from scars.data.splits import leave_one_dataset_out, assert_disjoint
from scars.data.mat_recordings import _discover_drff_r2
from scars.cli.preflight_mat import _canonical_label
from scars.experiment.pilot_policy import policy_for, THREE_POLICY


def test_sparse_domain_probe_is_unavailable_not_passed():
    from scars.audits.domain_probe import domain_probe_accuracy, domain_probe_shuffle_threshold
    x = np.arange(6).reshape(3, 2)
    domains = np.array(["A", "A", "DRFF"], dtype=object)
    ids = np.array(["a", "b", "c"], dtype=object)
    with pytest.raises(ValueError, match="two physical recordings"):
        domain_probe_accuracy(x, domains, ids)
    assert np.isnan(domain_probe_accuracy(x, domains, ids, allow_insufficient=True))
    record = domain_probe_shuffle_threshold(x, domains, ids, allow_insufficient=True)
    assert record["executed_permutations"] == 0
    assert np.isnan(record["threshold"])


def test_sparse_background_roles_remain_disjoint_and_all_classes_present():
    records = []
    for dataset in ("DroneRFa", "DroneRFb-DIR", "DRFF-R2"):
        for label in ("uav", "background"):
            count = 2 if dataset == "DRFF-R2" and label == "background" else 4
            for group in range(count):
                identity = f"{dataset}:{label}:{group}"
                records.append(Recording(recording_id=identity, dataset=dataset,
                    path=Path("unused.mat"), label=label, is_background=label == "background",
                    sample_rate_hz=100e6, metadata={"split_group": identity}))
    with pytest.raises(ValueError, match="physical groups"):
        leave_one_dataset_out(records, seed=24021)
    folds = leave_one_dataset_out(records, seed=24021, sparse_drff_background=True)
    assert len(folds) == 3
    assert [
        {k: sorted(r.recording_id for r in v) for k, v in f.source_roles.items()} for f in folds
    ] == [
        {k: sorted(r.recording_id for r in v) for k, v in f.source_roles.items()}
        for f in leave_one_dataset_out(list(reversed(records)), seed=24021, sparse_drff_background=True)
    ]
    for fold in folds:
        assert_disjoint(*fold.source_roles.values(), fold.held_target)
        for role in fold.source_roles.values():
            assert {r.label for r in role} == {"uav", "background"}
        if fold.fold_id != "held_dataset=DRFF-R2":
            for role, expected in (("source_fit", 1), ("source_validation", 1),
                                   ("source_calibration", 0), ("source_selection", 0)):
                assert sum(r.dataset == "DRFF-R2" and r.is_background for r in fold.source_roles[role]) == expected


def make_mat(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f["RF0_I"] = np.zeros((32, 1), dtype=np.float32)
        f["RF0_Q"] = np.ones((32, 1), dtype=np.float32)
        f["Fs"] = [[100e6]]
        td = f.create_dataset("TD", data=np.array([ord(c) for c in "mini5PRO_1"], dtype=np.uint16)[:, None])
        td.attrs["MATLAB_class"] = b"char"


def test_background_subset_overrides_stale_td_and_keeps_provenance(tmp_path):
    make_mat(tmp_path / "dataset/dataset7-environment/indoor_environment.mat")
    stream = _discover_drff_r2(tmp_path)[0]
    assert stream.label == "background"
    assert stream.metadata["TD"] == "mini5PRO_1"
    assert stream.group_id == "background:indoor_environment"
    assert "before UAV activation" in stream.metadata["label_provenance"]


def test_partial_download_is_not_silently_used(tmp_path):
    path = tmp_path / "dataset/dataset3-single_drone_hover/mini5PRO_1_hover.mat"
    make_mat(path)
    path.with_name(path.name + ".aria2").touch()
    with pytest.raises(ValueError, match="download sidecar"):
        _discover_drff_r2(tmp_path)


def test_author_excluded_mat_is_skipped_before_opening_invalid_hdf5(tmp_path):
    excluded = "dataset6-wifi_mixed/mavic3S_1_c4.mat"
    bad = tmp_path / "dataset" / excluded
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"incomplete download - not HDF5")
    bad.with_name(bad.name + ".aria2").touch()
    make_mat(tmp_path / "dataset/dataset7-environment/outdoor_environment.mat")
    streams = _discover_drff_r2(tmp_path, (excluded,))
    assert len(streams) == 1 and streams[0].label == "background"
    assert bad.read_bytes() == b"incomplete download - not HDF5"


def test_three_dataset_exclusion_registry_is_exact():
    root = Path(__file__).resolve().parents[1]
    exclusions = json.loads((root / "configs/exclusions.three_dataset.json").read_text())
    assert exclusions["drff_relative_paths"] == [
        "dataset3-single_drone_hover/mavicAir2_5_hover_c2_u2_d1.mat",
        "dataset6-wifi_mixed/mavic3S_1_c4.mat",
    ]


def test_three_dataset_preflight_records_exclusions_and_three_fold_plan(tmp_path, monkeypatch):
    from scars.cli import preflight_mat
    project = Path(__file__).resolve().parents[1]
    raw = tmp_path / "raw"
    for label in range(2):
        for group in range(4):
            a = raw / "DroneRFa_2024/dataset" / f"T{'0000' if label == 0 else '0001'}_D00_S{group:04d}.mat"
            b = raw / "DroneRFb-DIR_2025/dataset/train" / (f"background_slice_{group}.mat" if label == 0 else f"A1_IN_S{group}_slice_1.mat")
            for path, pairs in ((a, [("RF0_I", "RF0_Q"), ("RF1_I", "RF1_Q")]), (b, [("I", "Q")])):
                path.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(path, "w") as handle:
                    for i, q in pairs:
                        handle[i] = np.arange(32, dtype=np.float32)[None, :]
                        handle[q] = np.ones((1, 32), dtype=np.float32)
    drff = raw / "DRFF-R2_2026"
    for name in ("indoor", "outdoor"):
        make_mat(drff / f"dataset/dataset7-environment/{name}_environment.mat")
    for day in range(4):
        make_mat(drff / f"dataset/dataset3-single_drone_hover/mini5PRO_1_hover_c1_u1_d{day}.mat")
    exclusion = project / "configs/exclusions.three_dataset.json"
    for relative in json.loads(exclusion.read_text())["drff_relative_paths"]:
        path = drff / "dataset" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"incomplete")
        path.with_name(path.name + ".aria2").touch()
    monkeypatch.setattr(preflight_mat, "_hardware", lambda: {
        "ram_bytes": 32 * 1024**3, "gpu": {"name": "fixture", "vram_bytes": 8 * 1024**3}})
    output = tmp_path / "preflight"
    result = preflight_mat.run_preflight(dataset_root=raw, output_dir=output,
        label_contract=project / "configs/label_contract.three_dataset_binary.json",
        datasets=["DroneRFa", "DroneRFb-DIR", "DRFF-R2"], hash_files=True,
        window_samples=16, hop_samples=16, max_windows_per_recording=1,
        exclusion_manifest=exclusion, pilot_three_dataset=True)
    report = json.loads(result.read_text())
    assert report["status"] == "ready", report["issues"]
    assert report["fold_count"] == 3 and report["excluded_drff_existing_file_count"] == 2
    manifest = json.loads((output / "recordings_recognition.json").read_text())
    assert manifest["excluded_drff_relative_paths"] == json.loads(exclusion.read_text())["drff_relative_paths"]


def test_mixed_emitters_are_not_coerced_into_single_emitter(tmp_path):
    make_mat(tmp_path / "dataset/dataset2-drone_mixed/mavicair2_mix_mini3pro_c1.mat")
    stream = _discover_drff_r2(tmp_path)[0]
    assert stream.task_type == "multilabel"
    assert stream.emitter_labels == ("mavicair2", "mini3pro")


def test_three_dataset_policy_and_binary_uav_mapping(tmp_path):
    make_mat(tmp_path / "dataset/dataset6-wifi_mixed/mini5PRO_1_c1.mat")
    stream = _discover_drff_r2(tmp_path)[0]
    contract = json.loads((Path(__file__).resolve().parents[1] /
        "configs/label_contract.three_dataset_binary.json").read_text())
    assert _canonical_label(stream, contract) == "uav"
    assert policy_for({"campaign_mode": "pilot_three_dataset"}) == THREE_POLICY
