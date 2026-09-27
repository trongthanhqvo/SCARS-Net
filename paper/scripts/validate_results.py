#!/usr/bin/env python3
"""Fail-closed structural validator for the manuscript result artifact."""
from __future__ import annotations
import argparse, json, sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from scars.results.validator import validate_results as validate_runtime_results
from scars.results.registry import manuscript_registry_envelope

REQUIRED = {"schema_version", "evidence_type", "registry", "eligibility", "integrity", "tables", "mechanism", "hypotheses", "detection", "figures", "provenance"}

def null_paths(obj, prefix=""):
    out = []
    if obj is None:
        return [prefix]
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not key.startswith("_") and key not in {"row_fields"}:
                out.extend(null_paths(value, f"{prefix}.{key}" if prefix else key))
    elif isinstance(obj, list):
        if not obj:
            out.append(prefix)
        else:
            for i, value in enumerate(obj):
                out.extend(null_paths(value, f"{prefix}[{i}]"))
    return out

def check_ci(obj, prefix=""):
    if isinstance(obj, dict):
        if "ci95_low" in obj or "ci95_high" in obj:
            lo, hi = obj.get("ci95_low"), obj.get("ci95_high")
            if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)) or lo > hi:
                raise SystemExit(f"invalid CI at {prefix}")
        for key, value in obj.items():
            check_ci(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            check_ci(value, f"{prefix}[{i}]")

def require_rows(obj, table_id, expected, fields):
    rows = obj["tables"][table_id].get("rows")
    if not isinstance(rows, list):
        raise SystemExit(f"{table_id}.rows must be a list")
    ids = [row.get("condition_id") for row in rows if isinstance(row, dict)]
    if set(ids) != set(expected) or len(ids) != len(set(ids)):
        raise SystemExit(f"{table_id} condition registry mismatch")
    for row in rows:
        absent = [f for f in fields if f not in row or row[f] is None]
        if absent:
            raise SystemExit(f"{table_id}.{row.get('condition_id')} missing {absent}")

def check_figures(figures, registry):
    for fig_id, spec in figures.items():
        if "series" in spec:
            for s in spec["series"]:
                x, y = s.get("x"), s.get("y")
                if not isinstance(x, list) or not isinstance(y, list) or len(x) != len(y) or not x:
                    raise SystemExit(f"{fig_id}: unmatched x/y")
                for key in ("ci95_low", "ci95_high", "recording_support"):
                    if key in s and (not isinstance(s[key], list) or len(s[key]) != len(y)):
                        raise SystemExit(f"{fig_id}: unmatched {key}")
        elif "matrix" in spec:
            matrix = spec["matrix"]
            if not isinstance(matrix, list) or not matrix or any(len(row) != len(matrix[0]) for row in matrix):
                raise SystemExit(f"{fig_id}: invalid matrix")
        elif "panels" in spec:
            if not isinstance(spec["panels"], list) or not spec["panels"]:
                raise SystemExit(f"{fig_id}: invalid panels")
            fields = spec.get("panel_fields")
            if not fields:
                raise SystemExit(f"{fig_id}: missing panel_fields")
            for i, panel in enumerate(spec["panels"]):
                absent = [f for f in fields if f not in panel or panel[f] is None]
                if absent:
                    raise SystemExit(f"{fig_id}.panels[{i}] missing {absent}")
                if fig_id == "fig_scars_selection_h2" and panel.get("panel_id") == "selection":
                    n = len(panel["configuration_id"])
                    for key in ("source_instability", "source_macro_f1", "cost", "feasible", "pareto", "selected"):
                        if len(panel[key]) != n:
                            raise SystemExit(f"{fig_id}: unmatched {key}")
        if fig_id == "fig_compute_tradeoff":
            names = {s.get("x_name", s.get("panel_id")) for s in spec["series"]}
            if set(spec.get("x_names", [])) - names:
                raise SystemExit("fig_compute_tradeoff: missing latency/MACs/bytes axis")

def validate_payload(obj, allow_placeholder=False):
    missing = REQUIRED - set(obj)
    if missing:
        raise ValueError(f"missing top-level keys: {sorted(missing)}")
    if not allow_placeholder:
        if obj["evidence_type"] != "real_confirmatory":
            raise ValueError("not a real_confirmatory artifact")
        if int(obj["eligibility"].get("eligible_domains") or 0) < 3:
            raise ValueError("fewer than three eligible held domains")
        if obj["integrity"].get("status") != "passed" or obj["integrity"].get("critical_failures"):
            raise ValueError("critical integrity gate is not passed")
        if obj["mechanism"].get("m1", {}).get("status") not in {"supported", "partial", "negative", "untestable"}:
            raise ValueError("invalid M1 state")
        if obj.get("registry") != manuscript_registry_envelope()["registry"]:
            raise ValueError("result registry differs from the frozen manuscript registry")
        projection = deepcopy({key: obj[key] for key in REQUIRED})
        detection_eligible = bool(obj.get("eligibility", {}).get("detection_eligible"))
        if not detection_eligible:
            projection["detection"] = {
                "eligible": False,
                "status": "ineligible_no_shared_canonical_background",
            }
            projection["tables"]["tab_detection"] = {
                "row_fields": obj["tables"]["tab_detection"].get("row_fields", []),
                "rows": "not_applicable",
            }
        missing_values = null_paths(projection)
        if missing_values:
            raise ValueError("null/empty required values: " + ", ".join(missing_values[:20]))
        if obj["registry"].get("required_seeds") != [11, 23, 37, 53, 71]:
            raise ValueError("paired seed registry mismatch")
        allowed = set(obj["registry"].get("allowed_status", []))
        for hid in ("H1", "H2", "H3", "H4"):
            if obj["hypotheses"][hid]["status"] not in allowed:
                raise ValueError(f"invalid status for {hid}")
        statuses = [obj["hypotheses"][hid]["status"] for hid in ("H1", "H2", "H3", "H4")]
        if "untestable" in statuses and len(set(statuses)) != 1:
            raise ValueError("the four-way Holm family cannot be partially decided")
        require_rows(obj, "tab_primary_results", obj["registry"]["recognition_conditions"], obj["registry"]["recognition_row_fields"])
        require_rows(obj, "tab_ablations", obj["registry"]["ablation_conditions"], obj["registry"]["ablation_row_fields"])
        for table_id, table in obj["tables"].items():
            if table_id == "tab_detection" and not detection_eligible:
                continue
            fields = table.get("row_fields")
            if not fields or table_id in {"tab_primary_results", "tab_ablations"}:
                continue
            rows = table.get("rows")
            if not isinstance(rows, list) or not rows:
                raise SystemExit(f"{table_id}.rows must be a nonempty list")
            for i, row in enumerate(rows):
                absent = [f for f in fields if f not in row or row[f] is None]
                if absent:
                    raise SystemExit(f"{table_id}.rows[{i}] missing {absent}")
        for row in obj["tables"]["tab_primary_results"]["rows"]:
            for key in ("mean_macro_f1", "worst_domain_macro_f1", "balanced_accuracy"):
                if not 0 <= row[key] <= 1:
                    raise SystemExit(f"range error: {row['condition_id']}.{key}")
        check_figures(obj["figures"], obj["registry"])
        check_ci(obj)
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--allow-placeholder", action="store_true")
    args = ap.parse_args()
    obj = json.loads(args.results.read_text())
    try:
        validate_payload(obj, allow_placeholder=args.allow_placeholder)
        if not args.allow_placeholder:
            validate_runtime_results(obj, args.results.parent)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print("schema structure: OK")

if __name__ == "__main__":
    main()
