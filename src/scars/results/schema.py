from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .registry import manuscript_registry_envelope


REQUIRED_TOP_LEVEL = (
    "schema_version",
    "evidence_type",
    "run",
    "provenance",
    "datasets",
    "splits",
    "source_fitting",
    "representations",
    "source_selection",
    "static_metadata",
    "held_domain",
    "baselines",
    "ablations",
    "detection",
    "robustness",
    "costs",
    "statistics",
    "leakage_audits",
    "hypotheses",
    "warnings",
)


def empty_results(evidence_type: str, run_id: str) -> dict[str, Any]:
    if evidence_type not in {"synthetic_dev", "real_confirmatory", "pilot_two_dataset", "pilot_three_dataset"}:
        raise ValueError("Unknown evidence_type")
    payload = {
        "schema_version": "scars-canonical-results-2.0",
        "evidence_type": evidence_type,
        "run": {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "engineering_smoke" if evidence_type == "synthetic_dev" else "in_progress",
            "elapsed_sec": None,
            "command": None,
            "device": None,
            "seeds": [],
        },
        "provenance": {},
        "datasets": {"domains": [], "ontology": None, "acquisition_gate": None},
        "splits": {"folds": [], "manifest_hash": None},
        "source_fitting": {"folds": []},
        "representations": {"configuration_order": [], "records": {}},
        "source_selection": {"folds": [], "policy": None},
        "static_metadata": None,
        "held_domain": {
            "folds": [],
            "focal_condition_id": "wst_cyclic_resolution_16",
            "focal_average_macro_f1": None,
            "focal_worst_macro_f1": None,
            "selected_average_macro_f1": None,
            "selected_worst_macro_f1": None,
            # Backward-compatible aliases for the operational source-selected estimand.
            "average_macro_f1": None,
            "worst_macro_f1": None,
        },
        "baselines": {"folds": []},
        "ablations": {"channel_masks": [], "resolution": [], "controls": []},
        "detection": {"folds": [], "summary": None},
        "robustness": {"snr": [], "sir": [], "normalized_curve_auc": None},
        "costs": {"conditions": {}, "measurement_manifest": None},
        "statistics": {
            "decision_engine_version": None,
            "h1": None,
            "h2": None,
            "h3": None,
            "h4": None,
            "block_bootstrap": {},
            "holm": None,
        },
        "leakage_audits": {
            "duplicates": None,
            "near_duplicates": None,
            "temporal": None,
            "domain_probe": None,
            "label_shuffle": None,
            "background_only": None,
        },
        "hypotheses": {
            key: {
                "estimate": None,
                "raw_p": None,
                "holm_p": None,
                "status": "open",
                "decision": None,
                "reason": "real confirmatory evidence absent",
            }
            for key in ("H1", "H2", "H3", "H4")
        },
        "warnings": [],
    }
    payload.update(manuscript_registry_envelope())
    return payload
