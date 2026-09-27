"""Canonical measurement export for the amended two-dataset pilot."""
import json
from pathlib import Path

import numpy as np

from scars.experiment.pilot_policy import policy_for, finite_json
from scars.experiment.common import atomic_json
from scars.results.schema import empty_results
from scars.results.provenance import sha256_file


def build_pilot_results(campaign_dir, campaign, preflight, sources, models, targets, resamples):
    from scars.cli.finalize_results import _ensemble_seed_rows, _target_condition_draws, _target_contrast_draws
    from scars.evaluation.classification import classification_metrics, calibration_metrics
    from scars.results.validator import validate_results

    POLICY = policy_for(preflight)
    if len(targets) != len(POLICY["datasets"]) or set(preflight["datasets"]) != set(POLICY["datasets"]):
        raise ValueError("Pilot finalization requires all registered dataset folds")
    if any(model.get("pilot_policy") != POLICY for model in models):
        raise ValueError("Model freeze does not implement the amended pilot policy")
    payload = empty_results(preflight["campaign_mode"], Path(campaign_dir).name)
    payload["run"].update(status="finalized", seeds=POLICY["seeds"],
                          device=preflight.get("hardware"), command="finalize_pilot")
    payload["pilot_policy"] = dict(POLICY)
    payload["datasets"] = {"domains": preflight["datasets"],
                           "ontology": preflight["shared_canonical_labels"]}
    payload["source_fitting"]["folds"] = models
    payload["source_selection"] = {"folds": sources, "policy": POLICY["global_selection"]}
    payload["representations"]["configuration_order"] = sources[0]["h2_configuration_order"]
    payload["held_domain"]["folds"] = targets
    payload["splits"]["folds"] = [
        {"fold_id": source["fold_id"], **source["source_roles"],
         "held_target": source["held_target_recordings"]} for source in sources
    ]
    conditions = set(targets[0]["conditions"])
    if any(set(fold["conditions"]) != conditions for fold in targets):
        raise ValueError("Pilot folds have different condition sets")
    summary = {}
    for index, condition in enumerate(sorted(conditions)):
        domain_values, draws = _target_condition_draws(targets, condition, resamples, 24022 + index)
        metrics = []
        for fold in targets:
            truth, prediction, probability, classes = _ensemble_seed_rows(fold["conditions"][condition]["seeds"])
            metrics.append({"fold_id": fold["fold_id"],
                            **classification_metrics(truth, prediction),
                            "calibration": calibration_metrics(truth, probability, classes)})
        summary[condition] = {
            "mean_macro_f1": float(np.mean(domain_values)),
            "worst_macro_f1": float(np.min(domain_values)),
            "mean_macro_f1_ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "per_domain": metrics,
            "cost_by_fold": [fold["conditions"][condition]["cost"] for fold in targets],
        }
    payload["metrics"] = summary
    payload["costs"]["conditions"] = {key: row["cost_by_fold"] for key, row in summary.items()}
    payload["baselines"]["folds"] = [fold["conditions"] for fold in targets]
    payload["ablations"] = {"folds": [fold.get("ablation_bank", {}) for fold in targets]}
    payload["robustness"] = {"folds": [fold.get("robustness", {}) for fold in targets]}
    payload["detection"] = {"folds": [
        {condition: [row["metrics"].get("detection") for row in record["seeds"]]
         for condition, record in fold["conditions"].items()} for fold in targets
    ]}
    payload["leakage_audits"] = {"folds": [fold.get("integrity", {}) for fold in targets]}
    payload["external_sota_registry"] = campaign.get("external_sota_registry", {})
    contrasts = {}
    for control in ("ordinary_gate", "shuffled_pcrd"):
        delta, draws, worst = _target_contrast_draws(targets, "pcrd", control, resamples, 25022)
        contrasts[control] = {"estimate": float(np.mean(delta)),
                              "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
                              "worst_difference_ci95": np.quantile(worst, [0.025, 0.975]).tolist()}
    payload["mechanism"] = {"status": "exploratory_only", "comparisons": contrasts,
                            "fold_status": [model["pilot_mechanism_status"] for model in models]}
    payload["statistics"].update(registered_resamples=resamples,
        block_bootstrap={"unit": "recording", "stratification": "within_domain_class",
                         "domain_weight": "equal", "seeds": "probability_ensemble_before_metric"})
    for item in payload["hypotheses"].values():
        item.update(status="open", reason="Amended fixed-WCES pilot is not confirmatory evidence")
    payload["primary_metric"] = {"name": "equal_domain_recording_macro_f1", "value": summary["pcrd"]["mean_macro_f1"],
                                 "direction": "maximize", "best_condition": "pcrd"}
    payload["held_domain"].update(average_macro_f1=summary["pcrd"]["mean_macro_f1"],
                                  worst_macro_f1=summary["pcrd"]["worst_macro_f1"])
    payload["warnings"] = ["Fixed WCES pilot: domain-probe/retention diagnostics do not gate training.",
                            "Null diagnostics are unavailable; H1-H4/M1 remain unconfirmed."]
    for model in models:
        status = model["pilot_mechanism_status"]
        if not status["pcrd_available"] or not status["shuffle_distinct"]:
            payload["warnings"].append(model["fold_id"] + ": " + json.dumps(status))
    artifact_files = ["source_campaign.json", "target_campaign.json", "preflight.json", "splits.json", "recordings_recognition.json"]
    artifact_files += [f["directory"] + "/" + name for f in campaign["folds"]
                       for name in ("source_freeze.json", "model_freeze.json", "target_metrics.json")]
    payload["provenance"] = {"source_environment": campaign["provenance"],
        "artifacts": [{"path": path, "sha256": sha256_file(Path(campaign_dir) / path)} for path in artifact_files]}
    payload = finite_json(payload)
    validate_results(payload, Path(campaign_dir), independent_recompute=False)
    return payload
