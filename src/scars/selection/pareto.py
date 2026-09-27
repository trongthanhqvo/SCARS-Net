from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObjectiveRecord:
    stable_id: str
    nuisance_sensitivity: float
    source_selection_macro_f1: float
    batch1_latency_ms: float
    bytes_per_sample: float
    estimated_macs: float
    feasible: bool = True

    def primary_minimize_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.nuisance_sensitivity,
                -self.source_selection_macro_f1,
                self.bytes_per_sample,
                self.estimated_macs,
                self.batch1_latency_ms,
            ]
        )


def pareto_front(records: list[ObjectiveRecord], tolerance: float = 1.0e-12) -> list[str]:
    ids = [record.stable_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Pareto stable_id values must be unique")
    feasible = [record for record in records if record.feasible]
    front = []
    for candidate in feasible:
        c = candidate.primary_minimize_vector()
        dominated = any(
            np.all(other.primary_minimize_vector() <= c + tolerance)
            and np.any(other.primary_minimize_vector() < c - tolerance)
            for other in feasible
            if other.stable_id != candidate.stable_id
        )
        if not dominated:
            front.append(candidate.stable_id)
    return sorted(front)


def select_unique(
    records: list[ObjectiveRecord], source_f1_absolute_tolerance: float = 0.01
) -> tuple[str, list[str]]:
    ids = [record.stable_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Selection stable_id values must be unique")
    feasible = [record for record in records if record.feasible]
    if not feasible:
        raise RuntimeError("No feasible candidate; deployment envelope blocks selection")
    front_ids = pareto_front(records)
    best_source_f1 = max(record.source_selection_macro_f1 for record in feasible)
    eligible = [
        record
        for record in feasible
        if record.stable_id in front_ids
        and record.source_selection_macro_f1 >= best_source_f1 - source_f1_absolute_tolerance
    ]
    if not eligible:
        raise AssertionError("Pareto/F1 tolerance rule unexpectedly produced no candidate")
    chosen = min(
        eligible,
        key=lambda record: (
            record.nuisance_sensitivity,
            record.batch1_latency_ms,
            record.stable_id,
        ),
    )
    return chosen.stable_id, front_ids
