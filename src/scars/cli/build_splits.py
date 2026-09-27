from __future__ import annotations

import argparse
import json
from pathlib import Path

from scars.data.manifest import load_manifest
from scars.data.splits import leave_one_dataset_out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=24021)
    args = parser.parse_args()
    folds = leave_one_dataset_out(load_manifest(args.manifest), args.seed)
    payload = {
        "group_split_before_windowing": True,
        "folds": [
            {
                "fold_id": fold.fold_id,
                "source_fit": [r.recording_id for r in fold.source_fit],
                "source_calibration": [r.recording_id for r in fold.source_calibration],
                "source_selection": [r.recording_id for r in fold.source_selection],
                "source_validation": [r.recording_id for r in fold.source_validation],
                "held_target": [r.recording_id for r in fold.held_target],
            }
            for fold in folds
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
