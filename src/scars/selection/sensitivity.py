from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from typing import Iterable

import numpy as np

from scars.representations.tensor import SourceFittedTensor
from .nuisance import NuisanceCase


def family_linf_displacement(clean: np.ndarray, perturbed: np.ndarray) -> np.ndarray:
    clean = np.asarray(clean)
    perturbed = np.asarray(perturbed)
    if clean.shape != perturbed.shape:
        raise ValueError("Clean and perturbed family tensors must have equal shape")
    if clean.ndim < 2:
        raise ValueError("Expected a batch axis plus coefficient axes")
    axes = tuple(range(1, clean.ndim))
    return np.max(np.abs(perturbed - clean), axis=axes)


def append_invariance_check(
    clean: np.ndarray, perturbed: np.ndarray, appended_coordinates: int = 7
) -> bool:
    base = family_linf_displacement(clean, perturbed)
    clean_flat = clean.reshape(len(clean), -1)
    perturbed_flat = perturbed.reshape(len(perturbed), -1)
    stable = np.zeros((len(clean), appended_coordinates), dtype=clean.dtype)
    augmented = family_linf_displacement(
        np.concatenate([clean_flat, stable], axis=1),
        np.concatenate([perturbed_flat, stable], axis=1),
    )
    return bool(np.array_equal(base, augmented))


def recording_family_domain_reduce(
    family_values: dict[str, np.ndarray], recording_ids: np.ndarray, domains: np.ndarray
) -> float:
    """Median windows per family/recording, then max family, then domain balance."""
    recording_family: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, (recording_id, domain) in enumerate(zip(recording_ids, domains)):
        key = (str(domain), str(recording_id))
        for family, values in family_values.items():
            recording_family[key][family].append(float(values[index]))
    domain_values: dict[str, list[float]] = defaultdict(list)
    for (domain, _recording), by_family in recording_family.items():
        family_medians = [float(np.median(values)) for values in by_family.values()]
        domain_values[domain].append(max(family_medians))
    return float(np.mean([np.median(values) for values in domain_values.values()]))


def recording_family_details(
    family_values: dict[str, np.ndarray], recording_ids: np.ndarray, domains: np.ndarray
) -> list[dict[str, object]]:
    recording_family: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, (recording_id, domain) in enumerate(zip(recording_ids, domains)):
        for family, values in family_values.items():
            recording_family[(str(domain), str(recording_id))][family].append(
                float(values[index])
            )
    output = []
    for (domain, recording_id), by_family in sorted(recording_family.items()):
        medians = {family: float(np.median(values)) for family, values in sorted(by_family.items())}
        output.append(
            {
                "domain": domain,
                "recording_id": recording_id,
                "family_medians": medians,
                "recording_max": max(medians.values()),
            }
        )
    return output


def source_instability(
    representation: SourceFittedTensor,
    windows: np.ndarray,
    recording_ids: np.ndarray,
    domains: np.ndarray,
    cases: Iterable[NuisanceCase],
    seed: int,
) -> dict[str, object]:
    clean = representation.transform_by_family(windows)
    case_values: dict[str, float] = {}
    case_recordings: dict[str, list[dict[str, object]]] = {}
    case_draws = []
    config_payload = representation.source_artifact().get("config", {})
    config_hash = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    for case_index, case in enumerate(cases):
        case_seed = int(seed + case_index)
        rng = np.random.default_rng(case_seed)
        perturbed_windows = np.stack(
            [case.transform(window, rng, case.severity) for window in windows]
        )
        perturbed = representation.transform_by_family(perturbed_windows)
        family_values = {
            family: family_linf_displacement(clean[family], perturbed[family])
            for family in clean
        }
        case_values[case.id] = recording_family_domain_reduce(
            family_values, recording_ids, domains
        )
        case_recordings[case.id] = recording_family_details(
            family_values, recording_ids, domains
        )
        case_draws.append(
            {"case_id": case.id, "seed": case_seed, "representation_config_sha256": config_hash}
        )
    return {
        "metric": "worst_case_family_linf_after_source_global_clip",
        "cases": case_values,
        "recording_values": case_recordings,
        "nuisance_draws": case_draws,
        "value": max(case_values.values()),
        "aggregation_order": (
            "family_linf_per_window -> median_windows_per_family_recording -> "
            "max_family_per_recording -> median_recordings_per_domain -> "
            "equal_mean_domains -> max_nuisance_severity"
        ),
    }
