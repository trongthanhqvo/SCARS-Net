from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scars.probes.small_resnet import seed_everything
from scars.baselines import H2_BANK_ORDER
from scars.results.schema import REQUIRED_TOP_LEVEL, empty_results
from scars.results.validator import ResultValidationError, validate_results
from scars.results.writer import write_results
from scars.cli.finalize_results import run as finalize_results


def test_stochastic_seed_reproducibility():
    seed_everything(71)
    first = np.random.random(5)
    seed_everything(71)
    second = np.random.random(5)
    np.testing.assert_array_equal(first, second)


def test_result_schema_preserves_fold_records(tmp_path):
    payload = empty_results("synthetic_dev", "test")
    payload["held_domain"]["folds"] = [{"fold_id": "d0", "conditions": {"W": {"macro_f1": 0.5}}}]
    path = tmp_path / "results.json"
    write_results(path, payload)
    loaded = json.loads(path.read_text())
    assert tuple(key for key in REQUIRED_TOP_LEVEL if key in loaded) == REQUIRED_TOP_LEVEL
    assert loaded["held_domain"]["folds"][0]["conditions"]["W"]["macro_f1"] == 0.5


def test_synthetic_cannot_decide_hypotheses():
    payload = empty_results("synthetic_dev", "test")
    payload["hypotheses"]["H1"]["status"] = "supported"
    with pytest.raises(ResultValidationError):
        validate_results(payload)


def test_synthetic_cannot_populate_confirmatory_latex(tmp_path):
    payload = empty_results("synthetic_dev", "test")
    result_path = tmp_path / "results.json"
    write_results(result_path, payload)
    output = tmp_path / "macros.tex"
    script = Path(__file__).resolve().parents[1] / "paper" / "scripts" / "results_to_latex.py"
    process = subprocess.run(
        [sys.executable, str(script), str(result_path), "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    assert "refus" in process.stderr.lower()
    assert not output.exists()


def test_empty_real_result_fails_closed():
    payload = empty_results("real_confirmatory", "empty")
    with pytest.raises(ResultValidationError, match="finalized"):
        validate_results(payload)


def test_confirmatory_finalizer_refuses_nonregistered_resample_count(tmp_path):
    with pytest.raises(ValueError, match="exactly 10,000"):
        finalize_results(campaign_dir=tmp_path, output=tmp_path / "results.json", resamples=999)


def test_latex_rejects_real_label_without_confirmatory_gates(tmp_path):
    payload = empty_results("real_confirmatory", "pretend")
    result_path = tmp_path / "results.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "macros.tex"
    script = Path(__file__).resolve().parents[1] / "paper" / "scripts" / "results_to_latex.py"
    process = subprocess.run(
        [sys.executable, str(script), str(result_path), "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    assert "refus" in process.stderr.lower()
    assert not output.exists()


def _minimal_blocked_real_result(artifact_root: Path):
    payload = empty_results("real_confirmatory", "blocked-valid")
    payload["run"]["status"] = "finalized"
    domains = ["d0", "d1", "d2"]
    payload["datasets"] = {
        "domains": domains,
        "ontology": {"shared_label_intersection": ["a", "b"]},
        "acquisition_gate": {
            "domain_gate": {
                "gate_open": True,
                "eligible_domain_ids": domains,
                "complete_domain_count": 3,
            },
            "compute_gate": {"gate_open": True},
        },
    }
    payload["provenance"].update(
        {
            "recordings_manifest_sha256": "1" * 64,
            "acquisition_manifest_sha256": "2" * 64,
            "combined_config_hash": "3" * 64,
            "ontology_sha256": "4" * 64,
            "label_map_sha256": "5" * 64,
            "source_tree_sha256": "6" * 64,
            "dataset_manifest_sha256": {domain: "7" * 64 for domain in domains},
        }
    )
    payload["representations"]["configuration_order"] = list(H2_BANK_ORDER)
    payload["splits"]["manifest_hash"] = "split"
    for index, domain in enumerate(domains):
        fold_id = f"held_dataset={domain}"
        payload["splits"]["folds"].append(
            {
                "fold_id": fold_id,
                "source_train_recordings": [f"s{index}"],
                "source_validation_recordings": [f"v{index}"],
                "held_target_recordings": [f"t{index}"],
            }
        )
        freeze_path = artifact_root / "folds" / str(index) / "source_freeze.json"
        freeze_path.parent.mkdir(parents=True, exist_ok=True)
        freeze_path.write_text(json.dumps({"fold_id": fold_id}), encoding="utf-8")
        source_hash = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
        payload["source_fitting"]["folds"].append(
            {
                "fold_id": fold_id,
                "h2_bank_status": "complete",
                "configuration_order": list(H2_BANK_ORDER),
                "source_freeze_path": f"folds/{index}/source_freeze.json",
                "source_freeze_sha256": source_hash,
                "records": [{"stable_id": key} for key in H2_BANK_ORDER],
            }
        )
        payload["source_selection"]["folds"].append(
            {
                "fold_id": fold_id,
                "source_artifact_hash": source_hash,
                "selected_theta_hat": H2_BANK_ORDER[0],
            }
        )
        payload["held_domain"]["folds"].append(
            {
                "fold_id": fold_id,
                "configuration_order": list(H2_BANK_ORDER),
                "selected_theta_hat": H2_BANK_ORDER[0],
                "conditions": {
                    key: {
                        "macro_f1": 0.0,
                        "confusion_matrix": [[0, 1], [0, 0]],
                        "label_order": ["a", "b"],
                        "per_class_recall": {"a": 0.0, "b": 0.0},
                        "replication_unit": "recording",
                        "recording_count": 1,
                        "recording_order": [f"t{index}"],
                        "recording_truth": ["a"],
                        "recording_prediction": ["b"],
                    }
                    for key in reversed(H2_BANK_ORDER)
                },
                "target_reads": 1,
                "target_unlocks": 1,
                "target_recording_ids": [f"t{index}"],
            }
        )
    payload["statistics"]["h2"] = {
        "status": "blocked",
        "configuration_order": list(H2_BANK_ORDER),
        "configuration_count": 12,
        "no_deletion_gate": "passed",
    }
    payload["statistics"].update(
        {
            "h1": {"inferential_status": "blocked"},
            "h3": {
                "learned_probe_seed_status": {
                    "status": "complete",
                    "completed": [11, 23, 37, 53, 71],
                }
            },
            "h4": {"inferential_status": "blocked"},
        }
    )
    payload["ablations"] = {
        "channel_masks": [{}, {}, {}],
        "resolution": [{}, {}, {}],
        "controls": {"degeneracy": {}},
    }
    payload["costs"]["measurement_manifest"] = {"deployment_budget_gate_open": True}
    payload["costs"]["conditions"] = {
        key: [
            {
                "bytes_per_sample": 1.0,
                "batch1_latency_ms": 1.0,
                "estimated_macs": 1.0,
            }
            for _ in domains
        ]
        for key in H2_BANK_ORDER
    }
    payload["leakage_audits"].update(
        {
            "duplicates": {"cross_domain_hashes": []},
            "near_duplicates": {"cross_domain_pair_count": 0},
            "temporal": {"status": "passed", "cross_partition_recording_violations": []},
            "domain_probe": {"aggregate_accuracy": 0.0},
            "label_shuffle": {"folds": [{}, {}, {}]},
            "background_only": {"status": "not_applicable"},
        }
    )
    for key in ("H1", "H2", "H3", "H4"):
        payload["hypotheses"][key]["status"] = "blocked"
    payload["held_domain"]["average_macro_f1"] = 0.0
    payload["held_domain"]["worst_macro_f1"] = 0.0
    payload["held_domain"]["focal_average_macro_f1"] = 0.0
    payload["held_domain"]["focal_worst_macro_f1"] = 0.0
    payload["held_domain"]["selected_average_macro_f1"] = 0.0
    payload["held_domain"]["selected_worst_macro_f1"] = 0.0
    return payload


def test_real_result_remains_valid_after_sorted_json_round_trip(tmp_path):
    payload = _minimal_blocked_real_result(tmp_path)
    path = tmp_path / "results.json"
    write_results(path, payload)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    validate_results(loaded, artifact_root=tmp_path)


def test_blocked_real_result_cannot_populate_confirmatory_macros(tmp_path):
    payload = _minimal_blocked_real_result(tmp_path)
    path = tmp_path / "results.json"
    write_results(path, payload)
    output = tmp_path / "macros.tex"
    script = Path(__file__).resolve().parents[1] / "paper" / "scripts" / "results_to_latex.py"
    process = subprocess.run(
        [sys.executable, str(script), str(path), "--output", str(output)], capture_output=True, text=True
    )
    assert process.returncode != 0
    assert "refus" in process.stderr.lower()
    assert not output.exists()


def test_fabricated_supported_decisions_with_blocked_h2_fail_closed(tmp_path):
    payload = _minimal_blocked_real_result(tmp_path)
    payload["held_domain"]["average_macro_f1"] = 0.9999
    payload["held_domain"]["worst_macro_f1"] = 0.9999
    payload["statistics"]["holm"] = {key: 0.0 for key in ("H1", "H2", "H3", "H4")}
    for key in ("H1", "H2", "H3", "H4"):
        payload["hypotheses"][key]["status"] = "supported"
    with pytest.raises(ResultValidationError):
        validate_results(payload, artifact_root=tmp_path)


def test_stale_decided_real_artifacts_are_refused_by_versioned_engine(tmp_path):
    payload = _minimal_blocked_real_result(tmp_path)
    payload["statistics"]["holm"] = {key: 0.0 for key in ("H1", "H2", "H3", "H4")}
    for key in ("H1", "H2", "H3", "H4"):
        payload["hypotheses"][key]["status"] = "supported"
    with pytest.raises(ResultValidationError):
        validate_results(payload, artifact_root=tmp_path)
