#!/usr/bin/env python3
"""Convert DroneRFb-DIR MAT files into the frozen W/C/E/S representation.

The input directory is ``twin_droneRF`` and must contain ``train/``, ``test/``,
``train_labels.txt`` and ``test_labels.txt``.  Only the train split is used to
fit cyclic frequencies and source-global normalizers; the same frozen transform
is then applied to train and test files.  No parameter is fitted on test.

Each exported window contains W.png, C.png, E.png, S.png and, by default,
tensor.npy in canonical [W,C,E,S] order.  PNG files are display encodings;
tensor.npy is the lossless float32 model input.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scars.cli.export_mat_images import run_export  # noqa: E402
from scars.cli.export_iq_npy_cache import run_iq_cache_export  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert DroneRFb-DIR I/Q MAT files to SCARS W/C/E/S images."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to twin_droneRF (contains train/, test/, and label text files).",
    )
    parser.add_argument(
        "--iq-cache-only",
        action="store_true",
        help=(
            "Write lossless complex64 IQ-window .npy files plus iq_recordings.json "
            "for the experiment pipeline; skip W/C/E/S PNG generation."
        ),
    )
    parser.add_argument(
        "--cache-windows-per-file",
        type=int,
        default=16,
        help="Windows per MAT stream for --iq-cache-only (default: 16).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty output directory.",
    )
    parser.add_argument(
        "--max-windows-per-file",
        type=int,
        default=1,
        help=(
            "Maximum windows per MAT file (default: 1 for a safe preview; "
            "0 exports every valid window and can require very large storage)."
        ),
    )
    parser.add_argument(
        "--preview-scale",
        type=int,
        default=16,
        help="Nearest-neighbour scale for RGBA/montage previews; family PNGs stay 16x16.",
    )
    parser.add_argument(
        "--no-save-tensor",
        action="store_true",
        help="Do not save canonical tensor.npy files (not recommended for model use).",
    )
    parser.add_argument(
        "--no-hash-inputs",
        action="store_true",
        help="Skip SHA-256 hashing of MAT inputs (reduces provenance).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    required = [input_dir / "train", input_dir / "test", input_dir / "test_labels.txt"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete DroneRFb-DIR input; missing: {missing}")
    if not any((input_dir / "train").rglob("*.mat")):
        raise FileNotFoundError(f"No training .mat files found under: {input_dir / 'train'}")
    if args.max_windows_per_file < 0:
        raise ValueError("--max-windows-per-file must be >= 0")
    if args.cache_windows_per_file <= 0:
        raise ValueError("--cache-windows-per-file must be positive")

    # The core adapter expects <dataset-root>/DroneRFb-DIR_2025/dataset.  A
    # temporary symlink supplies that layout without copying or modifying the
    # user's dataset.
    with tempfile.TemporaryDirectory(prefix="scars-dronerfb-layout-") as temp:
        dataset_dir = Path(temp) / "DroneRFb-DIR_2025"
        dataset_dir.mkdir()
        (dataset_dir / "dataset").symlink_to(input_dir, target_is_directory=True)
        if args.iq_cache_only:
            manifest = run_iq_cache_export(
                dataset_root=Path(temp),
                output_dir=output_dir,
                datasets=["DroneRFb-DIR"],
                window_samples=4096,
                hop_samples=2048,
                max_windows_per_stream=args.cache_windows_per_file,
                hash_inputs=not args.no_hash_inputs,
            )
            print(f"Complete IQ cache: {manifest}")
            return 0
        summary = run_export(
            dataset_root=Path(temp),
            output_dir=output_dir,
            datasets=["DroneRFb-DIR"],
            source_datasets=["DroneRFb-DIR"],
            source_splits=["train"],
            window_samples=4096,
            hop_samples=2048,
            fit_windows_per_stream=4,
            max_fit_windows=64,
            max_windows_per_stream=args.max_windows_per_file,
            output_bins=16,
            wst_j=4,
            wst_q=2,
            cyclic_count=8,
            frame_samples=128,
            frame_hop_samples=32,
            normalization="percentile",
            preview_scale=args.preview_scale,
            save_npy=not args.no_save_tensor,
            hash_inputs=not args.no_hash_inputs,
            enforce_group_firewall=False,
        )
    print(f"Complete: {summary}")
    print("Family order: W, C, E, S. PNG is for viewing; tensor.npy is model input.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
