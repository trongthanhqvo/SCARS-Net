from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .validator import validate_results


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_results(path: Path, payload: dict[str, Any]) -> None:
    safe = _json_safe(payload)
    validate_results(safe, artifact_root=path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe, indent=2, sort_keys=True), encoding="utf-8")
