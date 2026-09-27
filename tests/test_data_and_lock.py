from __future__ import annotations

import json

import pytest

from scars.data.base_adapter import Recording
from scars.data.manifest import load_manifest
from scars.data.splits import assert_disjoint, leave_one_dataset_out
from scars.data.windowing import window_recordings
from scars.cli.run_campaign import run as run_campaign
from scars.state import RunPhase, RunState


def test_group_split_precedes_windowing(recordings):
    folds = leave_one_dataset_out(recordings, seed=5)
    for fold in folds:
        assert_disjoint(
            fold.source_fit,
            fold.source_calibration,
            fold.source_selection,
            fold.source_validation,
            fold.held_target,
        )
        train = window_recordings(fold.source_fit, 256, 256)
        validation = window_recordings(fold.source_validation, 256, 256)
        target = window_recordings(fold.held_target, 256, 256)
        assert not (set(train.recording_ids) & set(validation.recording_ids))
        assert not (set(train.recording_ids) & set(target.recording_ids))
        expected = {
            (record.dataset, record.label)
            for record in (
                *fold.source_fit,
                *fold.source_calibration,
                *fold.source_selection,
                *fold.source_validation,
            )
        }
        for role in fold.source_roles.values():
            assert {(record.dataset, record.label) for record in role} == expected
        assert fold.coverage["stratification"] == "dataset_x_class_by_physical_group"


def test_group_safe_stratification_fails_closed_on_single_group_stratum():
    records = [
        Recording("a0", "a", None, "c0", False, 1.0, device_group="g0"),
        Recording("a1", "a", None, "c1", False, 1.0, device_group="g1"),
        Recording("b0", "b", None, "c0", False, 1.0, device_group="g0"),
        Recording("b1", "b", None, "c1", False, 1.0, device_group="g1"),
    ]
    with pytest.raises(ValueError, match="at least 4 physical groups"):
        leave_one_dataset_out(records, seed=5, minimum_target_recordings_per_class=1)


def test_four_source_roles_are_deterministic_and_group_disjoint(recordings):
    first = leave_one_dataset_out(recordings, seed=24021)
    second = leave_one_dataset_out(recordings, seed=24021)
    assert [
        {name: [record.group_id for record in members] for name, members in fold.source_roles.items()}
        for fold in first
    ] == [
        {name: [record.group_id for record in members] for name, members in fold.source_roles.items()}
        for fold in second
    ]


def test_repeated_device_recordings_share_one_group():
    first = Recording("r1", "d", None, "c", False, 1.0, device_group="device-a", iq=None)
    second = Recording("r2", "d", None, "c", False, 1.0, device_group="device-a", iq=None)
    assert first.group_id == second.group_id


def test_target_lock_requires_frozen_source_hash(tmp_path):
    state = RunState(tmp_path / "run_state.json")
    state.save()
    state.transition(RunPhase.SPLITS_FROZEN)
    state.transition(RunPhase.SOURCE_FITTING_COMPLETE)
    state.transition(RunPhase.PARETO_FROZEN)
    with pytest.raises(RuntimeError, match="source artifact"):
        state.transition(RunPhase.TARGET_UNLOCKED)


def test_target_unlock_occurs_once(tmp_path):
    state = RunState(tmp_path / "state.json")
    state.save()
    state.transition(RunPhase.SPLITS_FROZEN)
    state.transition(RunPhase.SOURCE_FITTING_COMPLETE)
    state.source_artifact_hash = "abc"
    state.transition(RunPhase.PARETO_FROZEN)
    state.transition(RunPhase.TARGET_UNLOCKED)
    state.record_target_load(["target-r1"])
    payload = json.loads(state.path.read_text())
    assert payload["target_reads"] == 1
    with pytest.raises(RuntimeError):
        state.transition(RunPhase.TARGET_UNLOCKED)


def test_target_ledger_survives_restart(tmp_path):
    path = tmp_path / "state.json"
    state = RunState(path)
    state.save()
    state.transition(RunPhase.SPLITS_FROZEN)
    state.transition(RunPhase.SOURCE_FITTING_COMPLETE)
    state.source_artifact_hash = "frozen"
    state.transition(RunPhase.PARETO_FROZEN)
    state.transition(RunPhase.TARGET_UNLOCKED)
    state.record_target_load(["target-r1"])
    resumed = RunState(path)
    assert resumed.phase == RunPhase.TARGET_UNLOCKED
    assert resumed.target_reads == 1
    assert resumed.target_unlocks == 1
    with pytest.raises(PermissionError, match="only once"):
        resumed.record_target_load(["target-r1"])


def test_target_stream_is_restart_safe_per_recording(tmp_path):
    path = tmp_path / "stream-state.json"
    state = RunState(path)
    state.save()
    state.transition(RunPhase.SPLITS_FROZEN)
    state.transition(RunPhase.SOURCE_FITTING_COMPLETE)
    state.source_artifact_hash = "frozen-models"
    state.transition(RunPhase.PARETO_FROZEN)
    state.transition(RunPhase.TARGET_UNLOCKED)
    state.begin_target_stream(["r1", "r2"])
    state.authorize_target_recording("r1")
    state.complete_target_recording("r1")
    resumed = RunState(path)
    resumed.begin_target_stream(["r1", "r2"])
    with pytest.raises(PermissionError, match="already checkpointed"):
        resumed.authorize_target_recording("r1")
    resumed.complete_target_recording("r2")
    resumed.commit_target_stream("cache-hash")
    final = RunState(path)
    assert final.target_reads == 1
    assert final.target_read_status == "complete"
    assert final.target_cache_manifest_sha256 == "cache-hash"


def test_manifest_target_iq_is_lazy_and_guarded(tmp_path):
    import numpy as np

    iq_path = tmp_path / "iq.npy"
    np.save(iq_path, np.ones(512, dtype=np.complex64))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "recordings": [
                    {
                        "recording_id": "target",
                        "dataset": "d",
                        "path": str(iq_path),
                        "label": "c",
                        "is_background": False,
                        "sample_rate_hz": 1.0
                    }
                ]
            }
        )
    )
    record = load_manifest(manifest)[0]
    assert record.iq is None
    state = RunState(tmp_path / "lazy-state.json")
    state.save()
    with pytest.raises(PermissionError):
        window_recordings([record], 256, 256, access_role="held_target", state=state)
    state.transition(RunPhase.SPLITS_FROZEN)
    state.transition(RunPhase.SOURCE_FITTING_COMPLETE)
    state.source_artifact_hash = "frozen"
    state.transition(RunPhase.PARETO_FROZEN)
    state.transition(RunPhase.TARGET_UNLOCKED)
    batch = window_recordings([record], 256, 256, access_role="held_target", state=state)
    assert len(batch.iq) == 2
    assert state.target_reads == 1


def test_raw_iq_checksum_is_verified_before_window_read(tmp_path):
    import hashlib
    import numpy as np

    iq_path = tmp_path / "sealed.npy"
    np.save(iq_path, np.ones(512, dtype=np.complex64))
    expected = hashlib.sha256(iq_path.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "recordings": [
                    {
                        "recording_id": "sealed",
                        "dataset": "d",
                        "path": str(iq_path),
                        "label": "c",
                        "sha256": expected,
                    }
                ]
            }
        )
    )
    np.save(iq_path, np.zeros(512, dtype=np.complex64))
    record = load_manifest(manifest)[0]
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        window_recordings([record], 256, 256)


def test_manifest_rejects_duplicate_recording_ids(tmp_path):
    manifest = tmp_path / "duplicate.json"
    manifest.write_text(
        json.dumps(
            {
                "recordings": [
                    {"recording_id": "same", "dataset": "d", "path": "a.npy", "label": "c"},
                    {"recording_id": "same", "dataset": "d", "path": "b.npy", "label": "c"},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="Duplicate recording_id"):
        load_manifest(manifest)


def test_legacy_real_campaign_is_retired_before_any_manifest_or_target_access(tmp_path):
    with pytest.raises(PermissionError, match="legacy monolithic runner is retired"):
        run_campaign(
            recordings_manifest=tmp_path / "does-not-exist-recordings.json",
            acquisition_manifest=tmp_path / "does-not-exist-acquisition.json",
            deployment_budget=tmp_path / "does-not-exist-budget.yaml",
            output_dir=tmp_path / "real-output",
            seed=24021,
            window_samples=4096,
            hop_samples=2048,
            authorize_target=True,
        )
    assert not (tmp_path / "real-output").exists()
