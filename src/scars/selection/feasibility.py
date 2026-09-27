from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeploymentBudget:
    max_bytes_per_sample: float
    max_batch1_latency_ms: float
    max_estimated_macs: float

    def check(self, cost: dict[str, float]) -> tuple[bool, list[str]]:
        failures = []
        if not cost.get("measurement_valid", False):
            failures.append("invalid_batch1_latency_measurement")
        if cost["bytes_per_sample"] > self.max_bytes_per_sample:
            failures.append("bytes_per_sample")
        if cost["batch1_latency_ms"] > self.max_batch1_latency_ms:
            failures.append("batch1_latency_ms")
        if cost["estimated_macs"] > self.max_estimated_macs:
            failures.append("estimated_macs")
        return not failures, failures
