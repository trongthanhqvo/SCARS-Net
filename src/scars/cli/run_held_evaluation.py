from __future__ import annotations

import argparse
import json
from pathlib import Path

from scars.cli.run_campaign import main as campaign_main


def main() -> int:
    return campaign_main()


if __name__ == "__main__":
    raise SystemExit(main())
