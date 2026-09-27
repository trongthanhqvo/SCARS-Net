from __future__ import annotations

import pytest

from scars.cli.evaluate_target import _aggregate_seed_costs


def _seed_cost(model_parameters: int, model_macs: float, model_ms: float):
    return {
        "parameters": model_parameters,
        "macs": model_macs + 10.0,
        "model_macs": model_macs,
        "representation_macs": 10.0,
        "bytes_per_sample": 4096,
        "latency": {
            "median_ms": model_ms + 2.0,
            "model_median_ms": model_ms,
            "representation_median_ms": 2.0,
            "measurement_valid": True,
            "device": "cuda",
        },
        "peak_inference_memory_bytes": 1000,
    }


def test_deployed_seed_ensemble_sums_models_and_charges_representation_once():
    result = _aggregate_seed_costs(
        [_seed_cost(100, 20.0, 3.0), _seed_cost(100, 20.0, 3.0)]
    )
    assert result["ensemble_size"] == 2
    assert result["parameters"] == 200
    assert result["model_macs"] == pytest.approx(40.0)
    assert result["representation_macs"] == pytest.approx(10.0)
    assert result["macs"] == pytest.approx(50.0)
    assert result["latency"]["model_median_ms"] == pytest.approx(6.0)
    assert result["latency"]["representation_median_ms"] == pytest.approx(2.0)
    assert result["latency"]["median_ms"] == pytest.approx(8.0)
