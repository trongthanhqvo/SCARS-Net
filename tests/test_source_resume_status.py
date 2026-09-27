import importlib.util
import json
from pathlib import Path

from scars.cli.source_resume_status import inspect_source_status


def test_rejected_source_report_preserves_artifacts_and_explains_nonfinite_probe(tmp_path):
    root = tmp_path / "scars-pilot"
    fold = root / "fold-00"
    fold.mkdir(parents=True)
    payloads = {
        "source_campaign.json": {
            "status": "source_decision_rejected_scars_net",
            "folds": [{"fold_id": "f0", "directory": "fold-00"}],
            "model_freezes": [{"fold_id": "f0", "path": "fold-00/model_freeze.json"}],
        },
        "preflight.json": {}, "recordings_recognition.json": {}, "splits.json": {},
        "fold-00/source_freeze.json": {
            "active_families": [], "channel_decisions": {"records": [{
                "family": "W", "retain": False, "domain_probe_accuracy": float("nan"),
                "independent_value_delta": 0.0, "independent_value_pass": False,
                "domain_probe_shuffle_null": {"threshold": float("nan")},
            }]},
        },
        "fold-00/model_freeze.json": {"status": "scars_net_not_applicable_after_registered_shrink"},
        "fold-00/run_state.json": {"target_reads": 0, "target_unlocks": 0},
    }
    for name, payload in payloads.items():
        (root / name).write_text(json.dumps(payload))
    before = {name: (root / name).read_bytes() for name in payloads}
    report = inspect_source_status(root)
    assert not report["source_status_allows_evaluation"]
    assert "domain_probe_not_estimable" in report["folds"][0]["family_checks"][0]["failed_checks"]
    json.dumps(report, allow_nan=False)
    assert before == {name: (root / name).read_bytes() for name in payloads}


def test_runner_checks_rejected_decision_without_spawning_training_or_evaluation(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/run_dronerfa_dronerfb_pilot_pipeline.py"
    spec = importlib.util.spec_from_file_location("pilot_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    root = tmp_path / "scars-pilot"
    root.mkdir()
    (root / "source_campaign.json").write_text(json.dumps({"status": "source_decision_rejected_scars_net"}))
    monkeypatch.setattr("scars.cli.source_resume_status.write_report", lambda *a: {
        "source_status_allows_evaluation": False, "folds": [], "next_action": "Stop",
    })
    monkeypatch.setattr(runner, "_run", lambda *a: (_ for _ in ()).throw(AssertionError("No subprocess allowed")))
    monkeypatch.setattr("sys.argv", [str(path), "--dronerfa-dir", "/missing/raw-a", "--dronerfb-dir", "/missing/raw-b", "--output-dir", str(tmp_path), "--authorize-pilot-target"])
    assert runner.main() == 2
