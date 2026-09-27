"""Diagnose a stopped source campaign using JSON only; never unlock targets."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def inspect_source_status(campaign_dir: Path) -> dict:
    root = Path(campaign_dir).resolve()
    checked = []
    problems = []

    def read(relative, expected=None):
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError("Artifact path escapes campaign directory")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        checked.append({"path": str(path.relative_to(root)), "sha256": digest})
        if expected and digest != expected:
            problems.append("Artifact hash mismatch: " + relative)
        return json.loads(raw)

    def number(value):
        return value if isinstance(value, (float, int)) and math.isfinite(value) else None

    campaign = read("source_campaign.json")
    for key, relative in (("preflight", "preflight.json"), ("recordings", "recordings_recognition.json"), ("splits", "splits.json")):
        read(relative, campaign.get("input_hashes", {}).get(key))
    models = {entry["fold_id"]: entry for entry in campaign.get("model_freezes", [])}
    folds = []
    for entry in campaign.get("folds", []):
        directory = entry["directory"]
        freeze = read(directory + "/source_freeze.json", entry.get("source_freeze_sha256"))
        model_record = models.get(entry["fold_id"])
        model = read(model_record["path"], model_record.get("sha256")) if model_record else {}
        state = read(directory + "/run_state.json")
        records = []
        for row in freeze.get("channel_decisions", {}).get("records", []):
            failed = []
            for key in ("selected_by_global_pareto", "independent_value_pass", "shortcut_pass"):
                if not row.get(key):
                    failed.append(key)
            if row.get("similarity", {}).get("degenerate"):
                failed.append("representation_degenerate")
            probe = number(row.get("domain_probe_accuracy"))
            threshold = number(row.get("domain_probe_shuffle_null", {}).get("threshold"))
            if probe is None or threshold is None:
                failed.append("domain_probe_not_estimable")
            records.append({
                "family": row["family"], "retain": row.get("retain"),
                "independent_value_delta": number(row.get("independent_value_delta")),
                "domain_probe_accuracy": probe, "domain_probe_shuffle_q95": threshold,
                "failed_checks": failed,
            })
        folds.append({
            "fold_id": entry["fold_id"], "selected_stable_id": freeze.get("selected_stable_id"),
            "active_families": freeze.get("active_families"),
            "model_status": model.get("status", "not_created"),
            "reason": model.get("reason"), "family_checks": records,
            "saved_target_reads": state.get("target_reads"),
            "saved_target_unlocks": state.get("target_unlocks"),
        })
    status = campaign.get("status")
    ready = (status == "all_source_models_frozen" and bool(folds)
             and all(fold["model_status"] == "complete" for fold in folds)
             and not problems)
    return {
        "report_type": "source_resume_diagnostic_not_experimental_results",
        "campaign_status": status, "source_status_allows_evaluation": ready,
        "target_authorization_granted_by_this_report": False,
        "waveforms_read_by_this_report": 0, "folds": folds,
        "artifact_hash_issues": problems, "checked_json_artifacts": checked,
        "next_action": (
            "Use normal evaluator; all environment, model, and target guards still apply."
            if ready else
            "Stop before target access. Resolve source decisions before training/evaluation; "
            "do not replace empty active sets or edit frozen status to complete."
        ),
    }


def write_report(campaign_dir: Path, output: Path) -> dict:
    report = inspect_source_status(campaign_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reports are separate from the immutable scientific artifacts.
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = write_report(args.campaign_dir, args.output)
    print(json.dumps({"campaign_status": report["campaign_status"],
                      "source_status_allows_evaluation": report["source_status_allows_evaluation"],
                      "report": str(args.output), "next_action": report["next_action"]}, indent=2))
    return 0 if report["source_status_allows_evaluation"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
