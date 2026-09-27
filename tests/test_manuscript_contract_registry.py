import json
from pathlib import Path

import yaml

from scars.cli.freeze_source import H2_CONFIGURATION_ORDER, _candidates
from scars.results.registry import (
    ABLATION_CONDITIONS,
    RECOGNITION_CONDITIONS,
    manuscript_registry_envelope,
)


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_paper_and_config_registries_are_identical():
    paper = json.loads((ROOT / "paper" / "results_placeholder.json").read_text())
    assert paper["registry"] == manuscript_registry_envelope()["registry"]
    h2 = yaml.safe_load((ROOT / "configs" / "h2_bank.yaml").read_text())
    assert h2["configurations"] == list(H2_CONFIGURATION_ORDER)


def test_full_registered_wst_factorial_is_reachable():
    ids = {candidate.stable_id for candidate in _candidates()}
    expected = {f"wst_J{j}Q{q}" for j in (3, 4, 5) for q in (1, 2, 4)}
    assert expected <= ids
    assert expected <= set(ABLATION_CONDITIONS)


def test_3060ti_budget_and_training_contract_are_frozen():
    budget = yaml.safe_load((ROOT / "configs" / "deployment_budget.yaml").read_text())
    assert budget["hard_constraints"] == {
        "max_bytes_per_sample": 4096,
        "max_batch1_latency_ms": 100,
        "max_estimated_macs": 100_000_000,
    }
    baseline = yaml.safe_load((ROOT / "configs" / "baselines.yaml").read_text())
    learned = baseline["learned_common"]
    assert learned["oom_backoff"] == [32, 16, 8, 4]
    assert learned["effective_batch_size"] == 64
    assert baseline["tier_b"]["stochastic_seeds"] == [11, 23, 37, 53, 71]
    assert len(RECOGNITION_CONDITIONS) == 15
