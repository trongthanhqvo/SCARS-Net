from __future__ import annotations

import numpy as np

from scars.probes.ridge import RidgeProbe


def domain_probe_accuracy(
    features: np.ndarray, domains: np.ndarray, recording_ids: np.ndarray, *, allow_insufficient: bool = False
) -> float:
    """Recording-level domain probe with recording-disjoint train/test sets."""
    features = np.asarray(features)
    domains = np.asarray(domains)
    recording_ids = np.asarray(recording_ids)
    if not (len(features) == len(domains) == len(recording_ids)):
        raise ValueError("Domain probe arrays must align")
    unique_recordings = sorted(np.unique(recording_ids), key=str)
    recording_features = []
    recording_domains = []
    for recording_id in unique_recordings:
        mask = recording_ids == recording_id
        observed = np.unique(domains[mask])
        if len(observed) != 1:
            raise ValueError("A domain-probe recording crosses source domains")
        recording_features.append(np.mean(features[mask], axis=0))
        recording_domains.append(observed[0])
    features = np.asarray(recording_features)
    domains = np.asarray(recording_domains)
    unique = np.unique(domains)
    if len(unique) < 2:
        return float("nan")
    mapping = {domain: index for index, domain in enumerate(unique)}
    labels = np.asarray([mapping[value] for value in domains])
    train, test = [], []
    for label in range(len(unique)):
        indices = np.where(labels == label)[0]
        if len(indices) < 2:
            if allow_insufficient:
                return float("nan")
            raise ValueError("Domain probe needs at least two physical recordings per domain")
        train.extend(indices[::2])
        test.extend(indices[1::2])
    prediction = RidgeProbe(l2=0.05).fit(features[train], labels[train]).predict(features[test])
    return float(np.mean(prediction == labels[test]))


def domain_probe_shuffle_threshold(
    features: np.ndarray,
    domains: np.ndarray,
    recording_ids: np.ndarray,
    *,
    permutations: int = 999,
    seed: int = 24024,
    quantile: float = 0.95,
    allow_insufficient: bool = False,
) -> dict[str, object]:
    """Recording-level domain-label permutation null for the shortcut gate."""
    ids = np.asarray(recording_ids, dtype=object)
    labels = np.asarray(domains, dtype=object)
    unique_ids = np.asarray(sorted(set(ids.tolist()), key=str), dtype=object)
    recording_domains = []
    for recording_id in unique_ids:
        observed = np.unique(labels[ids == recording_id])
        if len(observed) != 1:
            raise ValueError("A domain-probe recording crosses source domains")
        recording_domains.append(observed[0])
    recording_domains = np.asarray(recording_domains, dtype=object)
    if allow_insufficient and any(np.sum(recording_domains == domain) < 2 for domain in np.unique(recording_domains)):
        return {"permutations": int(permutations), "executed_permutations": 0, "seed": int(seed),
                "quantile": float(quantile), "threshold": float("nan"), "values": [],
                "unit": "physical_recording", "status": "unavailable_insufficient_source_domain_support"}
    if len(np.unique(recording_domains)) < 2:
        return {
            "permutations": int(permutations), "seed": int(seed),
            "quantile": float(quantile), "threshold": float("nan"),
            "values": [], "unit": "physical_recording",
            "status": "unavailable_single_source_domain", "executed_permutations": 0,
        }
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(permutations):
        shuffled_recording = recording_domains[rng.permutation(len(recording_domains))]
        mapping = dict(zip(unique_ids.tolist(), shuffled_recording.tolist()))
        shuffled_windows = np.asarray([mapping[value] for value in ids], dtype=object)
        values.append(domain_probe_accuracy(features, shuffled_windows, ids))
    return {
        "permutations": int(permutations),
        "seed": int(seed),
        "quantile": float(quantile),
        "threshold": float(np.quantile(values, quantile, method="higher")),
        "values": [float(value) for value in values],
        "unit": "physical_recording",
    }
