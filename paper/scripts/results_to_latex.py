#!/usr/bin/env python3
"""Generate LaTeX result macros from a validated real-confirmatory result file."""
from __future__ import annotations
import argparse, json
from pathlib import Path

MAP = {
    "resEligibleDomains": "eligibility.eligible_domains",
    "resIntegrityStatus": "integrity.status",
    "resRelationCoverage": "mechanism.relation_coverage.aggregate",
    "resHtwoEstimate": "hypotheses.H2.estimate",
    "resHtwoP": "hypotheses.H2.raw_p",
    "resHtwoStatus": "hypotheses.H2.status",
    "resDetectionAUROC": "detection.summary.auroc",
    "resDetectionAUPRC": "detection.summary.auprc",
    "resDetectionFAR": "detection.summary.far",
    "resDetectionMiss": "detection.summary.miss_rate",
}

CORE_ABLATION_IDS = (
    "W", "C", "W+C", "W+C+E", "W+C+E+S",
    "active_minus_W", "active_minus_C", "active_minus_E", "active_minus_S",
)

def get(obj, path):
    for key in path.split("."):
        if isinstance(obj, list):
            matches = [row for row in obj if isinstance(row, dict) and row.get("condition_id") == key]
            if len(matches) != 1:
                raise KeyError(path)
            obj = matches[0]
        elif isinstance(obj, dict) and key in obj:
            obj = obj[key]
        else:
            raise KeyError(path)
    if obj is None:
        raise ValueError(f"null required value: {path}")
    return obj

def tex(value):
    if value is None:
        return r"\TBDReal"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list) and len(value) == 2:
        return f"[{tex(value[0])}, {tex(value[1])}]"
    return str(value).replace("_", r"\_").replace("%", r"\%")

def model_rows(rows):
    out = []
    for r in rows:
        ci = f"[{tex(r['ci95_low'])}, {tex(r['ci95_high'])}]"
        cost = f"{tex(r['parameters'])}/{tex(r['macs'])}/{tex(r['latency_ms'])}"
        out.append(f"{tex(r['condition_id'])} & {tex(r['mean_macro_f1'])} & {ci} & {tex(r['worst_domain_macro_f1'])} & {cost} \\\\")
    return " ".join(out)

def primary_rows(rows):
    out = []
    for r in rows:
        ci = f"[{tex(r['ci95_low'])}, {tex(r['ci95_high'])}]"
        cost = f"{tex(r['parameters'])}/{tex(r['macs'])}"
        out.append(f"{tex(r['condition_id'])} & {tex(r['mean_macro_f1'])} & {ci} & {tex(r['worst_domain_macro_f1'])} & {tex(r['balanced_accuracy'])} & {cost} \\\\")
    return " ".join(out)

def ablation_rows(rows):
    out = []
    for r in rows:
        delta = f"{tex(r['delta_vs_pcrd'])} [{tex(r['delta_ci95_low'])}, {tex(r['delta_ci95_high'])}]"
        diag = f"cos={tex(r['cosine'])}; NMAE={tex(r['nmae'])}; probe={tex(r['domain_probe_accuracy'])}"
        out.append(f"{tex(r['condition_id'])} & {tex(r['mean_macro_f1'])} & {delta} & {diag} & {tex(r['interpretation'])} \\\\")
    return " ".join(out)

def generic_rows(rows, fields):
    return " ".join(" & ".join(tex(r[f]) for f in fields) + r" \\" for r in rows)

def decision_rows(rows):
    output = []
    for row in rows:
        status = tex(row.get("status"))
        reason = tex(row.get("decision_reason"))
        decision = f"{status}: {reason}"
        output.append(
            " & ".join(
                [tex(row.get("id")), tex(row.get("estimand")), tex(row.get("raw_p")),
                 tex(row.get("holm_p")), decision]
            ) + r" \\"
        )
    return " ".join(output)

def main():
    # Runtime validation depends on the experiment package; keep the import here
    # so the null-valued paper template can reuse formatting helpers standalone.
    from validate_results import validate_payload, validate_runtime_results

    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--output", type=Path, default=Path("generated/results_macros.tex"))
    args = ap.parse_args()
    data = json.loads(args.results.read_text())
    try:
        validate_payload(data)
        validate_runtime_results(data, args.results.parent)
    except ValueError as error:
        raise SystemExit(f"refusing output: canonical validation failed: {error}") from error
    lines = ["% Generated from validated canonical results; do not edit manually."]
    for macro, path in MAP.items():
        try:
            value = get(data, path)
        except (KeyError, ValueError):
            value = None
        lines.append(rf"\newcommand{{\{macro}}}{{{tex(value)}}}")
    lines.append(r"\newcommand{\AllModelRows}{" + model_rows(get(data, "tables.tab_primary_results.rows")) + "}")
    lines.append(r"\newcommand{\AllAblationRows}{" + ablation_rows(get(data, "tables.tab_ablations.rows")) + "}")
    core_rows = [
        row for row in get(data, "tables.tab_ablations.rows")
        if row.get("condition_id") in CORE_ABLATION_IDS
    ]
    lines.append(r"\newcommand{\CoreAblationRows}{" + ablation_rows(core_rows) + "}")
    primary = get(data, "tables.tab_primary_results.rows")
    lines.append(r"\newcommand{\PrimaryResultRows}{" + primary_rows(primary) + "}")
    lines.append(r"\newcommand{\MoneRows}{" + generic_rows(get(data, "tables.tab_m1.rows"), ["component", "estimate", "interval_or_threshold", "status"]) + "}")
    lines.append(r"\newcommand{\MainAblationRows}{" + ablation_rows(get(data, "tables.tab_ablations.rows")) + "}")
    decisions = get(data, "tables.tab_hypotheses.rows")
    lines.append(r"\newcommand{\DecisionRows}{" + decision_rows(decisions) + "}")
    lines.append(r"\newcommand{\RobustnessRows}{" + generic_rows(get(data, "tables.tab_robustness.rows"), ["condition_id", "snr_auc", "sir_auc", "worst_nuisance", "worst_degradation"]) + "}")
    detection_rows = data["tables"]["tab_detection"].get("rows", [])
    detection_tex = (
        generic_rows(detection_rows, ["condition_id", "auroc", "auprc", "far", "miss_rate", "ci95"])
        if detection_rows
        else r"\multicolumn{6}{c}{\TBDReal---detection ineligible without shared background} \\"
    )
    lines.append(r"\newcommand{\DetectionRows}{" + detection_tex + "}")
    lines.append(r"\newcommand{\PerClassRows}{" + generic_rows(get(data, "tables.tab_per_class.rows"), ["class_domain", "precision", "recall", "f1", "recording_support"]) + "}")
    lines.append(r"\newcommand{\RoutingRows}{" + generic_rows(get(data, "tables.tab_routing_calibration.rows"), ["diagnostic", "estimate", "interval_or_threshold", "interpretation"]) + "}")
    lines.append(r"\newcommand{\HypothesisRows}{" + decision_rows(decisions) + "}")
    lines.append(r"\newcommand{\IntegrityRows}{" + generic_rows(get(data, "tables.tab_integrity.rows"), ["audit", "status", "evidence_path_hash"]) + "}")
    args.output.write_text("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
