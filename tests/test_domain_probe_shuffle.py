import numpy as np

from scars.audits.domain_probe import domain_probe_shuffle_threshold


def test_domain_probe_shuffle_is_recording_level_and_deterministic():
    features = np.arange(48, dtype=float).reshape(8, 6)
    recording_ids = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
    domains = np.asarray(["x", "x", "x", "x", "y", "y", "y", "y"])
    first = domain_probe_shuffle_threshold(
        features, domains, recording_ids, permutations=9, seed=7
    )
    second = domain_probe_shuffle_threshold(
        features, domains, recording_ids, permutations=9, seed=7
    )
    assert first == second
    assert first["unit"] == "physical_recording"
    assert len(first["values"]) == 9
