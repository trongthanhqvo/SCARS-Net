"""Opt-in CPU integration: generated IQ, one epoch; never publication evidence."""
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from scars.cli import preflight_mat, freeze_source, train_source_models, evaluate_target, finalize_results
from scars.training.engine import TrainingSpec


@pytest.mark.skipif(os.environ.get("SCARS_RUN_PILOT_INTEGRATION") != "1", reason="Opt-in full pilot CPU integration")
@pytest.mark.parametrize("three", [False, True], ids=["two_dataset", "three_dataset"])
def test_full_two_dataset_pilot_on_generated_iq(tmp_path, monkeypatch, three):
    torch.set_num_threads(1)
    raw = tmp_path / "synthetic-mat"
    rng = np.random.default_rng(291)
    for dataset in ("DroneRFa_2024", "DroneRFb-DIR_2025"):
        for label in range(2):
            for group in range(4):
                if dataset == "DroneRFa_2024":
                    name = f"T{'0000' if label == 0 else '0001'}_D00_S{group:04d}.mat"
                    path = raw / dataset / "dataset" / name
                    pairs = [("RF0_I", "RF0_Q"), ("RF1_I", "RF1_Q")]
                else:
                    name = f"background_slice_{group}.mat" if label == 0 else f"A1_IN_S{group}_slice_1.mat"
                    path = raw / dataset / "dataset/train" / name
                    pairs = [("I", "Q")]
                path.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(path, "w") as handle:
                    for i_key, q_key in pairs:
                        x = (rng.normal(size=4096) + 1j * rng.normal(size=4096)).astype(np.complex64)
                        if label:
                            x += 4 * np.exp(2j * np.pi * (0.08 + group * 0.001) * np.arange(4096))
                        handle[i_key] = x.real[None, :]
                        handle[q_key] = x.imag[None, :]
    if three:
        for label in range(2):
            for group in range(2 if label == 0 else 4):
                if label == 0:
                    relative = "dataset7-environment/" + ("indoor_environment.mat" if group == 0 else "outdoor_environment.mat")
                else:
                    relative = f"dataset3-single_drone_hover/mini4PRO_{group + 1}_hover_c1_u1_d1.mat"
                path = raw / "DRFF-R2_2026/dataset" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                x = (rng.normal(size=4096) + 1j * rng.normal(size=4096)).astype(np.complex64)
                if label:
                    x += 4 * np.exp(2j * np.pi * 0.07 * np.arange(4096))
                with h5py.File(path, "w") as handle:
                    handle["RF0_I"], handle["RF0_Q"] = x.real[:, None], x.imag[:, None]
                    handle["Fs"] = np.array([[100e6]])
    monkeypatch.setenv("RAW_IQ_DATASET_PATH", str(raw))
    monkeypatch.setattr(preflight_mat, "_hardware", lambda: {
        "ram_bytes": 32 * 1024**3, "gpu": {"name": "fixture", "vram_bytes": 8 * 1024**3}})
    # Reduce only test runtime. Production still uses 100 epochs/patience 10.
    monkeypatch.setattr(train_source_models, "TrainingSpec", lambda **kwargs: TrainingSpec(
        **{**kwargs, "max_epochs": 1, "patience": 1}))
    # Label-shuffle scientific decisions are not asserted by this engineering test.
    monkeypatch.setattr(freeze_source, "_source_negative_controls", lambda *args, **kwargs: {
        "label_shuffle": {"status": "not_run_in_integration_fixture"},
        "background_only": {"status": "not_run_in_integration_fixture"}})
    project = Path(__file__).resolve().parents[1]
    preflight = tmp_path / "preflight"
    campaign = tmp_path / "scars-pilot"
    windowing = {"window_samples": 4096, "hop_samples": 2048, "max_windows_per_recording": 1}
    preflight_mat.run_preflight(dataset_root=raw, output_dir=preflight,
        label_contract=project / ("configs/label_contract.three_dataset_binary.json" if three else "configs/label_contract.dronerfa_dronerfb_binary_pilot.json"),
        datasets=["DroneRFa", "DroneRFb-DIR"] + (["DRFF-R2"] if three else []),
        hash_files=True, pilot_two_dataset=not three, pilot_three_dataset=three, **windowing)
    freeze_source.run(preflight_dir=preflight, output_dir=campaign, seed=24021, **windowing)
    train_source_models.run(preflight_dir=preflight, source_campaign_dir=campaign,
        device="cpu", max_epochs=100, patience=10, **windowing)
    evaluate_target.run(preflight_dir=preflight, source_campaign_dir=campaign,
        authorize_target=True, pilot_two_dataset=not three, pilot_three_dataset=three,
        freeze_package=None, device="cpu", **windowing)
    result_path = finalize_results.run(campaign_dir=campaign, output=campaign / "results.json", resamples=10000)
    result = json.loads(result_path.read_text())
    assert result["evidence_type"] == ("pilot_three_dataset" if three else "pilot_two_dataset")
    assert result["run"]["status"] == "finalized"
    assert len(result["held_domain"]["folds"]) == (3 if three else 2)
    assert all(item["status"] == "open" for item in result["hypotheses"].values())
    assert result["primary_metric"]["value"] is not None
