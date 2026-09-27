#!/usr/bin/env python3
"""Run the full SCARS pipeline as a non-confirmatory two-dataset pilot.

This script accepts the same direct dataset paths used by the MAT-to-image
converters.  It creates a persistent symlink layout under the output directory
and then runs:

preflight_mat -> freeze_source -> train_source_models -> evaluate_target -> finalize_results

The run is explicitly tagged as pilot_two_dataset.  It is useful for debugging
the complete pipeline on DroneRFa and DroneRFb-DIR before DRFF-R2 is available,
but it is not a replacement for the frozen three-dataset confirmatory campaign.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABEL_CONTRACT = (
    PROJECT_ROOT / "configs" / "label_contract.dronerfa_dronerfb_binary_pilot.json"
)
STAGES = ("preflight", "freeze", "train", "evaluate", "finalize")


def _run(command: list[str], env: dict[str, str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def _preflight_is_ready(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "ready"
    except (OSError, json.JSONDecodeError):
        return False


def _source_decision_blocks(output: Path, *, after_training: bool = False) -> bool:
    campaign_path = output / "scars-pilot" / "source_campaign.json"
    if not campaign_path.is_file():
        return False
    status = json.loads(campaign_path.read_text(encoding="utf-8")).get("status", "")
    if not str(status).startswith("source_decision_rejected") and not after_training:
        return False
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from scars.cli.source_resume_status import write_report

    report_path = output / "SOURCE_RESUME_STATUS.json"
    report = write_report(campaign_path.parent, report_path)
    if report["source_status_allows_evaluation"]:
        return False
    print(f"Source decision blocks continuation: {status}", flush=True)
    for fold in report["folds"]:
        print(f"  {fold['fold_id']}: active_families={fold['active_families']}; "
              f"model_status={fold['model_status']}", flush=True)
    print(f"Diagnostic report: {report_path}\n{report['next_action']}", flush=True)
    return True


def _link_dataset(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"Existing dataset link points elsewhere: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a non-confirmatory SCARS pilot on DroneRFa + DroneRFb-DIR."
    )
    parser.add_argument(
        "--dronerfa-dir",
        type=Path,
        required=False,
        help="Path to the DroneRFa directory containing .mat files.",
    )
    parser.add_argument(
        "--dronerfb-dir",
        type=Path,
        required=False,
        help="Path to twin_droneRF containing train/, test/, and label files.",
    )
    parser.add_argument(
        "--dronerfa-npy-dir",
        type=Path,
        help="DroneRFa IQ-cache directory containing iq_recordings.json.",
    )
    parser.add_argument(
        "--dronerfb-npy-dir",
        type=Path,
        help="DroneRFb-DIR IQ-cache directory containing iq_recordings.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or resumable pilot output directory.",
    )
    parser.add_argument(
        "--label-contract",
        type=Path,
        default=DEFAULT_LABEL_CONTRACT,
        help="Pilot label contract. Default maps DroneRFa/DroneRFb labels to background vs uav.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-samples", type=int, default=4096)
    parser.add_argument("--hop-samples", type=int, default=2048)
    parser.add_argument("--max-windows-per-recording", type=int, default=16)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--reuse-source-run", type=Path,
                        help="Import verified pre-target source-freeze artifacts from an old pilot into a NEW output directory.")
    parser.add_argument(
        "--stop-after",
        choices=STAGES,
        default="finalize",
        help="Stop after the selected stage.",
    )
    parser.add_argument(
        "--no-hash-files",
        action="store_true",
        help="Skip MAT SHA-256 hashing. Preflight will be blocked for confirmatory use but may still help debug parsing.",
    )
    parser.add_argument(
        "--authorize-pilot-target",
        action="store_true",
        help="Required to run the pilot held-dataset evaluation stage.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    raw_mode = args.dronerfa_dir is not None or args.dronerfb_dir is not None
    npy_mode = args.dronerfa_npy_dir is not None or args.dronerfb_npy_dir is not None
    if raw_mode == npy_mode:
        raise ValueError(
            "Choose exactly one complete input mode: --dronerfa-dir/--dronerfb-dir "
            "or --dronerfa-npy-dir/--dronerfb-npy-dir"
        )
    if raw_mode:
        if args.dronerfa_dir is None or args.dronerfb_dir is None:
            raise ValueError("Raw mode requires both dataset directories")
        dronerfa = args.dronerfa_dir.expanduser().resolve()
        dronerfb = args.dronerfb_dir.expanduser().resolve()
    else:
        if args.dronerfa_npy_dir is None or args.dronerfb_npy_dir is None:
            raise ValueError("NPY mode requires both IQ-cache directories")
        dronerfa = args.dronerfa_npy_dir.expanduser().resolve()
        dronerfb = args.dronerfb_npy_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if args.reuse_source_run is not None:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from scars.cli.prepare_pilot_amendment import prepare
        prepare(args.reuse_source_run, output, PROJECT_ROOT)
    if _source_decision_blocks(output):
        return 2
    if not dronerfa.exists():
        raise FileNotFoundError(f"DroneRFa input directory does not exist: {dronerfa}")
    if raw_mode:
        if not (dronerfb / "train").is_dir() or not (dronerfb / "test").is_dir():
            raise FileNotFoundError(f"DroneRFb-DIR path must contain train/ and test/: {dronerfb}")
    else:
        for cache in (dronerfa, dronerfb):
            if not (cache / "iq_recordings.json").is_file():
                raise FileNotFoundError(f"IQ-cache manifest does not exist: {cache / 'iq_recordings.json'}")
    if not args.label_contract.is_file():
        raise FileNotFoundError(f"Label contract does not exist: {args.label_contract}")
    output.mkdir(parents=True, exist_ok=True)

    dataset_root = output / "_dataset_layout"
    _link_dataset(dronerfa, dataset_root / "DroneRFa_2024" / "dataset")
    _link_dataset(dronerfb, dataset_root / "DroneRFb-DIR_2025" / "dataset")
    empty_exclusions = output / "pilot_empty_exclusions.json"
    if not empty_exclusions.exists():
        empty_exclusions.write_text(
            json.dumps(
                {
                    "schema_version": "scars-pilot-exclusions-1.0",
                    "recording_ids": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["RAW_IQ_DATASET_PATH"] = str(dataset_root)
    common_window = [
        "--window-samples",
        str(args.window_samples),
        "--hop-samples",
        str(args.hop_samples),
        "--max-windows-per-recording",
        str(args.max_windows_per_recording),
    ]

    preflight_dir = output / "preflight"
    source_dir = output / "scars-pilot"

    command = [
        sys.executable,
        "-m",
        "scars.cli.preflight_mat",
        "--dataset-root",
        str(dataset_root),
        "--output-dir",
        str(preflight_dir),
        "--label-contract",
        str(args.label_contract.resolve()),
        "--exclusion-manifest",
        str(empty_exclusions),
        "--datasets",
        "DroneRFa",
        "DroneRFb-DIR",
        "--pilot-two-dataset",
        *common_window,
    ]
    if args.no_hash_files:
        command.append("--no-hash-files")
    if not _preflight_is_ready(preflight_dir / "preflight.json"):
        _run(command, env)
    if args.stop_after == "preflight":
        return 0

    if not (source_dir / "source_campaign.json").is_file():
        _run(
            [
                sys.executable,
                "-m",
                "scars.cli.freeze_source",
                "--preflight-dir",
                str(preflight_dir),
                "--output-dir",
                str(source_dir),
                "--seed",
                "24021",
                *common_window,
            ],
            env,
        )
    if args.stop_after == "freeze":
        return 0

    campaign_status = json.loads((source_dir / "source_campaign.json").read_text())["status"]
    if campaign_status not in {"all_source_models_frozen", "target_evaluated"}:
        _run(
        [
            sys.executable,
            "-m",
            "scars.cli.train_source_models",
            "--preflight-dir",
            str(preflight_dir),
            "--source-campaign-dir",
            str(source_dir),
            "--device",
            args.device,
            *common_window,
            "--max-epochs",
            "100",
            "--patience",
            "10",
        ],
        env,
    )
    if args.stop_after == "train":
        return 2 if _source_decision_blocks(output, after_training=True) else 0

    if campaign_status != "target_evaluated" and _source_decision_blocks(output, after_training=True):
        return 2

    if not args.authorize_pilot_target:
        raise PermissionError("--authorize-pilot-target is required for pilot held-dataset evaluation")
    if not (source_dir / "target_campaign.json").is_file():
        _run(
            [
                sys.executable,
                "-m",
                "scars.cli.evaluate_target",
                "--preflight-dir",
                str(preflight_dir),
                "--source-campaign-dir",
                str(source_dir),
                "--authorize-target",
                "--pilot-two-dataset",
                "--device",
                args.device,
                *common_window,
            ],
            env,
        )
    if args.stop_after == "evaluate":
        return 0

    _run(
        [
            sys.executable,
            "-m",
            "scars.cli.finalize_results",
            "--campaign-dir",
            str(source_dir),
            "--output",
            str(source_dir / "results.json"),
            "--resamples",
            str(args.resamples),
        ],
        env,
    )
    print(f"Pilot results: {source_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
