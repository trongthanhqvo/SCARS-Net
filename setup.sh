#!/usr/bin/env bash
# Install only into an environment selected by the user; never start a campaign.
set -euo pipefail
cd -- "$(dirname -- "$0")"
profile="${1:-}"
if [[ "$profile" != "cpu" && "$profile" != "cuda" ]]; then
  echo "Usage: bash setup.sh cpu|cuda (activate a Python 3.11 environment first)" >&2
  exit 2
fi
python -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11 for this installation profile"'
python -m pip install -r requirements.txt
if [[ "$profile" == "cuda" ]]; then
  python -m pip install -r requirements-torch-cu121.txt
else
  python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
fi
python -m pip install -e . --no-deps
python -m pip check
python -c 'import scars.results.provenance; import torch, numpy; print("SCARS import OK; torch", torch.__version__, "numpy", numpy.__version__, "CUDA", torch.cuda.is_available())'
