#!/usr/bin/env python3
"""Convert a DroneRFa MAT directory into the frozen W/C/E/S representation.

The input directory is the directory that directly contains files such as
``T0000_D00_S0000.mat``.  The exporter reads RF0_I/RF0_Q and RF1_I/RF1_Q
lazily, frames each stream into 4096-sample windows with a 2048-sample hop,
and uses the exact SCARS representation implementation bundled with this
experiment project.

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
        description="Convert DroneRFa RF0/RF1 complex-IQ MAT files to SCARS W/C/E/S images."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory directly containing DroneRFa T*_D*_S*.mat files.",
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
        help="Windows per RF stream for --iq-cache-only (default: 16).",
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
            "Maximum windows per RF stream (default: 1 for a safe preview; "
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
        help="Skip SHA-256 hashing of large MAT inputs (reduces provenance).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"DroneRFa input directory not found: {input_dir}")
    if not any(input_dir.rglob("*.mat")):
        raise FileNotFoundError(f"No .mat files found under: {input_dir}")
    if args.max_windows_per_file < 0:
        raise ValueError("--max-windows-per-file must be >= 0")
    if args.cache_windows_per_file <= 0:
        raise ValueError("--cache-windows-per-file must be positive")

    # The core adapter expects <dataset-root>/DroneRFa_2024/dataset.  A
    # temporary symlink supplies that layout without copying or modifying the
    # user's dataset.
    with tempfile.TemporaryDirectory(prefix="scars-dronerfa-layout-") as temp:
        dataset_dir = Path(temp) / "DroneRFa_2024"
        dataset_dir.mkdir()
        (dataset_dir / "dataset").symlink_to(input_dir, target_is_directory=True)
        if args.iq_cache_only:
            manifest = run_iq_cache_export(
                dataset_root=Path(temp),
                output_dir=output_dir,
                datasets=["DroneRFa"],
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
            datasets=["DroneRFa"],
            source_datasets=["DroneRFa"],
            source_splits=["unspecified"],
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
        )
    print(f"Complete: {summary}")
    print("Family order: W, C, E, S. PNG is for viewing; tensor.npy is model input.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
