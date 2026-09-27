import importlib.util
from pathlib import Path

import pytest


def _validator():
    path = Path(__file__).resolve().parents[1] / "paper" / "scripts" / "validate_results.py"
    spec = importlib.util.spec_from_file_location("paper_validate_results", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_paper_boundary_rejects_failed_integrity_before_macro_generation():
    module = _validator()
    payload = {key: {} for key in module.REQUIRED}
    payload.update(
        {
            "schema_version": "scars-canonical-results-2.0",
            "evidence_type": "real_confirmatory",
            "eligibility": {"eligible_domains": 3},
            "integrity": {"status": "failed", "critical_failures": ["duplicate"]},
        }
    )
    with pytest.raises(ValueError, match="integrity"):
        module.validate_payload(payload)
