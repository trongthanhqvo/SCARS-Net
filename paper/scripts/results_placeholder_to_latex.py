#!/usr/bin/env python3
"""Render the registered null-valued result schema as visible LaTeX tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from results_to_latex import (
    CORE_ABLATION_IDS,
    MAP,
    ablation_rows,
    decision_rows,
    generic_rows,
    get,
    model_rows,
    primary_rows,
    tex,
)


REQUIRED_TABLES = {
    "tab_primary_results", "tab_m1", "tab_ablations", "tab_robustness",
    "tab_detection", "tab_per_class", "tab_routing_calibration",
    "tab_hypotheses", "tab_integrity",
}

IDENTIFIER_FIELDS = {
    "condition_id", "component", "class_domain", "diagnostic", "id",
    "estimand", "audit",
}


def validate_placeholder_rows(data: dict) -> None:
    """Reject any quantitative or interpretive evidence in the template file."""
    for table_id in REQUIRED_TABLES:
        for row_index, row in enumerate(data["tables"][table_id].get("rows", [])):
            for field, value in row.items():
                if field in IDENTIFIER_FIELDS:
                    continue
                if table_id == "tab_hypotheses" and field == "status" and value == "open":
                    continue
                if value is not None:
                    raise SystemExit(
                        "refusing placeholder rendering: non-null evidence at "
                        f"tables.{table_id}.rows[{row_index}].{field}"
                    )
    for path in MAP.values():
        try:
            value = get(data, path)
        except (KeyError, ValueError):
            continue
        if path == "hypotheses.H2.status" and value == "open":
            continue
        raise SystemExit(f"refusing placeholder rendering: non-null evidence at {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=Path("generated/results_macros.tex"))
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    if data.get("evidence_type") not in {"placeholder", "template", "synthetic_dev"}:
        raise SystemExit("refusing placeholder rendering for unsupported evidence type")
    missing = REQUIRED_TABLES - set(data.get("tables", {}))
    if missing:
        raise SystemExit(f"missing registered result tables: {sorted(missing)}")
    validate_placeholder_rows(data)

    lines = [
        "% Generated from results_placeholder.json; replace only through",
        "% scripts/results_to_latex.py after real-confirmatory validation.",
    ]
    for macro, path in MAP.items():
        try:
            value = get(data, path)
        except (KeyError, ValueError):
            value = None
        lines.append(rf"\newcommand{{\{macro}}}{{{tex(value)}}}")

    primary = get(data, "tables.tab_primary_results.rows")
    ablations = get(data, "tables.tab_ablations.rows")
    decisions = get(data, "tables.tab_hypotheses.rows")
    core = [row for row in ablations if row.get("condition_id") in CORE_ABLATION_IDS]
    lines.extend([
        r"\newcommand{\AllModelRows}{" + model_rows(primary) + "}",
        r"\newcommand{\AllAblationRows}{" + ablation_rows(ablations) + "}",
        r"\newcommand{\CoreAblationRows}{" + ablation_rows(core) + "}",
        r"\newcommand{\PrimaryResultRows}{" + primary_rows(primary) + "}",
        r"\newcommand{\MoneRows}{" + generic_rows(
            get(data, "tables.tab_m1.rows"),
            ["component", "estimate", "interval_or_threshold", "status"],
        ) + "}",
        r"\newcommand{\MainAblationRows}{" + ablation_rows(ablations) + "}",
        r"\newcommand{\DecisionRows}{" + decision_rows(decisions) + "}",
        r"\newcommand{\RobustnessRows}{" + generic_rows(
            get(data, "tables.tab_robustness.rows"),
            ["condition_id", "snr_auc", "sir_auc", "worst_nuisance", "worst_degradation"],
        ) + "}",
        r"\newcommand{\DetectionRows}{" + generic_rows(
            get(data, "tables.tab_detection.rows"),
            ["condition_id", "auroc", "auprc", "far", "miss_rate", "ci95"],
        ) + "}",
        r"\newcommand{\PerClassRows}{" + generic_rows(
            get(data, "tables.tab_per_class.rows"),
            ["class_domain", "precision", "recall", "f1", "recording_support"],
        ) + "}",
        r"\newcommand{\RoutingRows}{" + generic_rows(
            get(data, "tables.tab_routing_calibration.rows"),
            ["diagnostic", "estimate", "interval_or_threshold", "interpretation"],
        ) + "}",
        r"\newcommand{\HypothesisRows}{" + decision_rows(decisions) + "}",
        r"\newcommand{\IntegrityRows}{" + generic_rows(
            get(data, "tables.tab_integrity.rows"),
            ["audit", "status", "evidence_path_hash"],
        ) + "}",
    ])
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
