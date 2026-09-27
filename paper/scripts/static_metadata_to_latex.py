#!/usr/bin/env python3
"""Render deterministic, non-empirical SCARS metadata into LaTeX macros."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def grouped(value: int) -> str:
    return f"{int(value):,}".replace(",", "{,}")


def pathcode(value: str) -> str:
    return rf"\pathcode{{{value}}}"


def formula_tex(formula: dict[str, object] | None) -> str:
    if formula is None:
        return r"data-fit dependent"
    intercept = grouped(int(formula["intercept"]))
    slope = grouped(int(formula["per_class"]))
    return rf"${intercept}+{slope}K$"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--output", type=Path, default=Path("generated/static_metadata.tex"))
    args = parser.parse_args()
    data = json.loads(args.metadata.read_text())
    if data.get("status") != "computed_without_real_iq_or_training":
        raise SystemExit("refusing static LaTeX: unexpected metadata status")
    if data.get("empirical_evidence") is not False:
        raise SystemExit("refusing static LaTeX: metadata is not explicitly non-empirical")
    if any(value is not None for value in data["real_run_only"].values()):
        raise SystemExit("refusing static LaTeX: a real-run-only field was populated")

    counts = data["registry_counts"]
    protocol = data["protocol"]
    representation = data["representation"]
    nuisance = data["nuisance_grid"]
    nominal_wst = next(
        row
        for row in representation["wst_path_profiles"]
        if row["J"] == representation["wst_nominal"]["J"]
        and row["Q"] == representation["wst_nominal"]["Q"]
    )
    digest = hashlib.sha256(args.metadata.read_bytes()).hexdigest()
    lines = [
        "% Generated from deterministic static_metadata.json; do not edit manually.",
        f"% input-sha256: {digest}",
        rf"\newcommand{{\StaticRecognitionConditions}}{{{counts['recognition_conditions']}}}",
        rf"\newcommand{{\StaticAblationConditions}}{{{counts['ablation_conditions']}}}",
        rf"\newcommand{{\StaticSeedCount}}{{{counts['required_seeds']}}}",
        rf"\newcommand{{\StaticNuisanceFamilies}}{{{nuisance['family_count']}}}",
        rf"\newcommand{{\StaticNuisanceCases}}{{{nuisance['severity_case_count']}}}",
        rf"\newcommand{{\StaticWindowSamples}}{{{grouped(protocol['window_samples'])}}}",
        rf"\newcommand{{\StaticWindowBytes}}{{{grouped(protocol['input_window_bytes'])}}}",
        rf"\newcommand{{\StaticFramesPerWindow}}{{{representation['frames_per_window']}}}",
        rf"\newcommand{{\StaticNominalTensorBytes}}{{{grouped(representation['nominal_full_tensor_bytes'])}}}",
        rf"\newcommand{{\StaticNominalWSTPaths}}{{{nominal_wst['total_paths']}}}",
        rf"\newcommand{{\StaticBootstrapResamples}}{{{grouped(protocol['registered_resamples'])}}}",
    ]

    tensor_rows = []
    for row in representation["tensor_profiles"]:
        resolution = row["resolution"]
        shape = rf"$4\times{resolution}\times{resolution}$"
        tensor_rows.append(
            f"{resolution} & {shape} & {grouped(row['bytes_per_family'])} & "
            f"{grouped(row['full_tensor_bytes'])} & ${grouped(4 * resolution * resolution)}A$ \\\\"
        )
    lines.append(r"\newcommand{\StaticTensorRows}{" + " ".join(tensor_rows) + "}")

    model_rows = []
    for condition_id, profile in data["models"]["conditions"].items():
        input_shape = profile.get("input_shape")
        input_text = "--" if input_shape is None else "$[" + r"\times".join(map(str, input_shape[1:])) + "]$"
        model_rows.append(
            f"{pathcode(condition_id)} & {input_text} & "
            f"{formula_tex(profile.get('parameter_formula'))} & "
            f"{formula_tex(profile.get('mac_formula'))} \\\\"
        )
    lines.append(r"\newcommand{\StaticModelFormulaRows}{" + " ".join(model_rows) + "}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
