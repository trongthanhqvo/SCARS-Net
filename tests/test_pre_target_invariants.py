from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from scars.baselines import (
    assert_unique_h2_effective_configurations,
    frozen_external_sota_registry,
)
from scars.cli.train_source_models import SEEDS
from scars.cli.evaluate_target import run as evaluate_target
from scars.data.windowing import WindowBatch
from scars.experiment.common import one_window_per_recording
from scars.experiment.sampling import frozen_sampling_contract
from scars.results.registry import CONFIRMATORY_H2_CONFIGURATION_ORDER, RECOGNITION_CONDITIONS
from scars.selection.nuisance import (
    registered_execution_seed_contract,
    registered_nuisance_cases,
    registered_nuisance_specs,
)
from scars.experiment.pretarget import verify_freeze_package


ROOT = Path(__file__).resolve().parents[1]


def _yaml(name: str):
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_nuisance_yaml_is_an_exact_mirror_of_implementation_contract():
    nuisance_yaml = _yaml("nuisance_grid.yaml")
    observed = nuisance_yaml["nuisances"]
    assert nuisance_yaml["execution_seed_contract"] == registered_execution_seed_contract()
    expected = [asdict(item) for item in registered_nuisance_specs()]
    exact_fields = (
        "id", "severities", "parameter", "units", "mathematical_transform",
        "random_variables", "normalization", "seed_policy", "failure_conditions",
    )
    assert len(observed) == len(expected)
    for config, code in zip(observed, expected):
        for field in exact_fields:
            left = tuple(config[field]) if field == "severities" else config[field]
            right = tuple(code[field]) if field == "severities" else code[field]
            assert left == right, f"nuisance contract drift: {code['id']}.{field}"


def test_registered_nuisances_are_deterministic_shape_preserving_and_hashable():
    rng = np.random.default_rng(19)
    signal = (rng.normal(size=512) + 1j * rng.normal(size=512)).astype(np.complex64)
    for index, case in enumerate(registered_nuisance_cases()):
        left = case.transform(signal, np.random.default_rng(24021 + index), case.severity)
        right = case.transform(signal, np.random.default_rng(24021 + index), case.severity)
        assert left.shape == signal.shape
        assert np.all(np.isfinite(left))
        assert np.array_equal(left, right)
    encoded = json.dumps([asdict(item) for item in registered_nuisance_specs()], sort_keys=True)
    assert len(hashlib.sha256(encoded.encode()).hexdigest()) == 64


def test_h2_bank_is_exact_unique_and_amendment_precedes_target_access():
    config = _yaml("h2_bank.yaml")
    assert tuple(config["configurations"]) == CONFIRMATORY_H2_CONFIGURATION_ORDER
    assert config["amendment"]["target_access_count_at_amendment"] == 0
    signatures = assert_unique_h2_effective_configurations()
    assert len(signatures) == len(CONFIRMATORY_H2_CONFIGURATION_ORDER) == 12
    assert len({json.dumps(value, sort_keys=True) for value in signatures.values()}) == 12
    assert config["h4_comparator_contract"] == {
        "lower_resolution_candidate": "resolution_8",
        "reference_candidate": "resolution_32",
        "substitution_after_target_access": "forbidden",
    }


def test_one_window_policy_chooses_earliest_window_and_covers_every_condition():
    batch = WindowBatch(
        iq=np.arange(24).reshape(4, 6).astype(np.complex64),
        labels=np.asarray(["b", "a", "b", "a"], dtype=object),
        recording_ids=np.asarray(["r2", "r1", "r2", "r1"], dtype=object),
        domains=np.asarray(["d", "d", "d", "d"], dtype=object),
        starts=np.asarray([0, 0, 6, 6]),
        ends=np.asarray([6, 6, 12, 12]),
    )
    chosen = one_window_per_recording(batch)
    assert chosen.recording_ids.tolist() == ["r1", "r2"]
    assert chosen.starts.tolist() == [0, 0]
    contract = frozen_sampling_contract()
    covered = set(contract["pcrd_student_and_one_window_neural_conditions"]["condition_ids"])
    covered |= set(contract["full_capped_pool_conditions"]["condition_ids"])
    assert covered == set(RECOGNITION_CONDITIONS)


def test_five_model_ensemble_and_external_sota_statuses_are_frozen():
    assert SEEDS == (11, 23, 37, 53, 71)
    registry = frozen_external_sota_registry()
    assert len(registry) == 8
    assert all(item["implementation_status"] in {"executable", "frozen_ineligible"} for item in registry)
    assert all(item["implementation_status"] != "adapter_required" for item in registry)
    config_methods = _yaml("baselines.yaml")["external_sota"]["methods"]
    by_id = {item["id"]: item for item in config_methods}
    assert set(by_id) == {item["condition_id"] for item in registry}
    for item in registry:
        configured = by_id[item["condition_id"]]
        assert configured["implementation_status"] == item["implementation_status"]
        assert configured["reason"] == item["reason"]


def test_target_evaluator_refuses_a_blocked_freeze_before_reading_campaign(tmp_path):
    freeze = tmp_path / "freeze"
    freeze.mkdir()
    (freeze / "freeze-manifest.json").write_text(
        json.dumps(
            {
                "status": "blocked",
                "gates": {"ontology_frozen": False},
                "target_access_ledger": "target-access-ledger.json",
            }
        ),
        encoding="utf-8",
    )
    (freeze / "target-access-ledger.json").write_text(
        json.dumps({"target_access_count": 0, "target_performance_inspected": False}),
        encoding="utf-8",
    )
    (freeze / "MANIFEST.sha256").write_text(
        "\n".join(
            f"{hashlib.sha256((freeze / name).read_bytes()).hexdigest()}  {name}"
            for name in ("freeze-manifest.json", "target-access-ledger.json")
        ) + "\n",
        encoding="utf-8",
    )
    with np.testing.assert_raises_regex(PermissionError, "not READY"):
        evaluate_target(
            preflight_dir=tmp_path / "missing-preflight",
            source_campaign_dir=tmp_path / "missing-campaign",
            freeze_package=freeze,
            authorize_target=True,
            device="cpu",
            window_samples=4096,
            hop_samples=2048,
            max_windows_per_recording=64,
        )


def test_freeze_package_verifier_fails_on_manifested_artifact_drift(tmp_path):
    freeze = tmp_path / "freeze"
    freeze.mkdir()
    manifest = freeze / "freeze-manifest.json"
    ledger = freeze / "target-access-ledger.json"
    manifest.write_text(json.dumps({"status": "ready"}), encoding="utf-8")
    ledger.write_text(json.dumps({"target_access_count": 0}), encoding="utf-8")
    entries = []
    for path in (manifest, ledger):
        entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (freeze / "MANIFEST.sha256").write_text("\n".join(entries) + "\n", encoding="utf-8")
    assert verify_freeze_package(freeze)["status"] == "ready"
    ledger.write_text(json.dumps({"target_access_count": 1}), encoding="utf-8")
    with np.testing.assert_raises_regex(RuntimeError, "hash drift"):
        verify_freeze_package(freeze)
