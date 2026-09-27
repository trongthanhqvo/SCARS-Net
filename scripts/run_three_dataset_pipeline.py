#!/usr/bin/env python3
"""Three-domain exploratory campaign with bounded-memory source training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_dronerfa_dronerfb_pilot_pipeline import _link_dataset, _source_decision_blocks

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dronerfa", "dronerfb", "drff-r2"):
        parser.add_argument("--" + name + "-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-samples", type=int, default=4096)
    parser.add_argument("--hop-samples", type=int, default=2048)
    parser.add_argument("--max-windows-per-recording", type=int, default=16)
    parser.add_argument("--authorize-pilot-target", action="store_true")
    parser.add_argument(
        "--allow-memory-implementation-amendment",
        action="store_true",
        help="Permit audited memory-only resume of a compatible pre-target source campaign.",
    )
    parser.add_argument("--stop-after", choices=["preflight", "freeze", "train", "evaluate", "finalize"], default="finalize")
    return parser


def main():
    args = build_parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    roots = {
        "DroneRFa_2024": args.dronerfa_dir.expanduser().resolve(),
        "DroneRFb-DIR_2025": args.dronerfb_dir.expanduser().resolve(),
        "DRFF-R2_2026": args.drff_r2_dir.expanduser().resolve(),
    }
    for root in roots.values():
        if not root.is_dir():
            raise FileNotFoundError(root)
    if not all((roots["DroneRFb-DIR_2025"] / part).is_dir() for part in ("train", "test")):
        raise ValueError("--dronerfb-dir must point to twin_droneRF containing train/ and test/")
    if not (roots["DRFF-R2_2026"] / "dataset7-environment").is_dir():
        raise ValueError("--drff-r2-dir must contain the seven dataset1-... through dataset7-... folders")
    layout = output / "_dataset_layout"
    for name, root in roots.items():
        _link_dataset(root, layout / name / "dataset")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["RAW_IQ_DATASET_PATH"] = str(layout)
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from scars.experiment.common import atomic_json
    from scars.results.provenance import sha256_file
    contract = PROJECT_ROOT / "configs/label_contract.three_dataset_binary.json"
    exclusion_config = PROJECT_ROOT / "configs/exclusions.three_dataset.json"
    invocation = {
        "campaign_mode": "pilot_three_dataset",
        "data_roots": {key: str(value) for key, value in roots.items()},
        "window_samples": args.window_samples, "hop_samples": args.hop_samples,
        "max_windows_per_recording": args.max_windows_per_recording,
        "label_contract_sha256": sha256_file(contract),
        "exclusion_manifest_sha256": sha256_file(exclusion_config),
    }
    invocation_path = output / "input_contract.json"
    if invocation_path.exists() and json.loads(invocation_path.read_text()) != invocation:
        raise RuntimeError("Resume input/ontology/windowing changed; create a new campaign")
    atomic_json(invocation_path, invocation)
    exclusions = output / "pilot_exclusions.json"
    if not exclusions.exists():
        atomic_json(exclusions, json.loads(exclusion_config.read_text()))
    elif json.loads(exclusions.read_text()) != json.loads(exclusion_config.read_text()):
        raise RuntimeError("Frozen exclusion manifest changed")
    preflight, campaign = output / "preflight", output / "scars-pilot"
    window = ["--window-samples", str(args.window_samples), "--hop-samples", str(args.hop_samples),
              "--max-windows-per-recording", str(args.max_windows_per_recording)]

    def run(module, arguments):
        command = [sys.executable, "-m", module, *map(str, arguments)]
        print(" ".join(command), flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)

    if not (preflight / "preflight.json").exists():
        run("scars.cli.preflight_mat", [
            "--dataset-root", layout, "--output-dir", preflight, "--label-contract", contract,
            "--exclusion-manifest", exclusions, "--datasets", "DroneRFa", "DroneRFb-DIR", "DRFF-R2",
            "--pilot-three-dataset", *window])
    pre = json.loads((preflight / "preflight.json").read_text())
    if pre.get("campaign_mode") != "pilot_three_dataset" or pre.get("status") != "ready":
        raise RuntimeError("Preflight is blocked; review preflight.json and use a new output after resolving inputs")
    if args.stop_after == "preflight":
        return 0
    if not (campaign / "source_campaign.json").exists():
        run("scars.cli.freeze_source", ["--preflight-dir", preflight, "--output-dir", campaign, "--seed", "24021", *window])
    if args.stop_after == "freeze":
        return 0
    status = json.loads((campaign / "source_campaign.json").read_text())["status"]
    if status not in {"all_source_models_frozen", "target_evaluated"}:
        if _source_decision_blocks(output):
            return 2
        amendment = (["--allow-memory-implementation-amendment"]
                     if args.allow_memory_implementation_amendment else [])
        run("scars.cli.train_source_models", ["--preflight-dir", preflight, "--source-campaign-dir", campaign,
            "--device", args.device, "--max-epochs", "100", "--patience", "10", *amendment, *window])
    if status != "target_evaluated" and _source_decision_blocks(output, after_training=True):
        return 2
    if args.stop_after == "train":
        return 0
    if not args.authorize_pilot_target:
        raise PermissionError("Explicit --authorize-pilot-target is required")
    if not (campaign / "target_campaign.json").exists():
        run("scars.cli.evaluate_target", ["--preflight-dir", preflight, "--source-campaign-dir", campaign,
            "--pilot-three-dataset", "--authorize-target", "--device", args.device, *window])
    if args.stop_after == "evaluate":
        return 0
    run("scars.cli.finalize_results", ["--campaign-dir", campaign, "--output", campaign / "results.json", "--resamples", "10000"])
    print("Results: " + str(campaign / "results.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
