from __future__ import annotations

import argparse
import json
from pathlib import Path

from scars.data.registry import DatasetRegistry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path, help="Path to your dataset eligibility registry JSON")
    args = parser.parse_args()
    registry = DatasetRegistry.from_json(args.registry)
    eligible = registry.eligible_confirmatory_domains()
    print(json.dumps({"eligible_confirmatory_domains": eligible, "gate_open": len(eligible) >= 3}, indent=2))
    return 0 if len(eligible) >= 3 else 2


if __name__ == "__main__":
    raise SystemExit(main())
