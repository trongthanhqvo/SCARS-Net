from __future__ import annotations

import hashlib

import numpy as np

from scars.data.windowing import WindowBatch


def duplicate_audit(batch: WindowBatch) -> dict[str, object]:
    hashes = [hashlib.sha256(np.ascontiguousarray(window).view(np.uint8)).hexdigest() for window in batch.iq]
    locations: dict[str, set[str]] = {}
    for digest, domain in zip(hashes, batch.domains):
        locations.setdefault(digest, set()).add(str(domain))
    cross_partition = sorted(digest for digest, domains in locations.items() if len(domains) > 1)
    return {
        "sha256_duplicate_count": len(hashes) - len(set(hashes)),
        "cross_partition_hashes": cross_partition,
        "cross_domain_hashes": cross_partition,
        "partition_field": "audit_role",
    }


def temporal_adjacency_audit(batch: WindowBatch, forbidden_gap_samples: int = 0) -> dict[str, object]:
    violations = []
    for recording in np.unique(batch.recording_ids):
        indices = np.where(batch.recording_ids == recording)[0]
        order = indices[np.argsort(batch.starts[indices])]
        for left, right in zip(order[:-1], order[1:]):
            if batch.starts[right] - batch.ends[left] < forbidden_gap_samples:
                violations.append([str(recording), int(batch.starts[left]), int(batch.starts[right])])
    return {"forbidden_gap_samples": forbidden_gap_samples, "violations": violations}


def split_group_leakage_audit(split_folds: list[dict[str, object]]) -> dict[str, object]:
    """Measure physical-record overlap across every frozen fold partition."""
    violations = []
    for fold in split_folds:
        partitions = {
            role: set(
                fold.get(role, fold.get(f"{role}_recordings", []))
            )
            for role in (
                "source_fit",
                "source_calibration",
                "source_selection",
                "source_validation",
                "held_target",
            )
        }
        names = tuple(partitions)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                for recording_id in sorted(partitions[left].intersection(partitions[right])):
                    violations.append(
                        {
                            "fold_id": fold.get("fold_id"),
                            "recording_id": str(recording_id),
                            "left_partition": left,
                            "right_partition": right,
                        }
                    )
    return {
        "status": "passed" if not violations else "failed",
        "method": "physical_recording_id_partition_intersection",
        "cross_partition_recording_violations": violations,
        "within_partition_window_overlap_is_not_counted_as_leakage": True,
    }


def near_duplicate_audit(batch: WindowBatch, cosine_threshold: float = 0.999) -> dict[str, object]:
    """Cross-partition spectral-envelope screening; diagnostic, not a hash replacement."""
    recording_ids = sorted(set(batch.recording_ids.tolist()))
    fingerprints = []
    domains = []
    for recording_id in recording_ids:
        mask = batch.recording_ids == recording_id
        spectra = np.stack(
            [np.log1p(np.abs(np.fft.rfft(np.abs(window), n=256))) for window in batch.iq[mask]]
        )
        spectrum = np.mean(spectra, axis=0)
        spectrum -= spectrum.mean()
        spectrum /= max(float(np.linalg.norm(spectrum)), 1.0e-12)
        fingerprints.append(spectrum)
        record_domains = np.unique(batch.domains[mask])
        if len(record_domains) != 1:
            raise ValueError("Recording crosses domains during near-duplicate audit")
        domains.append(str(record_domains[0]))
    fingerprints = np.asarray(fingerprints)
    pairs = []
    for left in range(len(fingerprints)):
        for right in range(left + 1, len(fingerprints)):
            if domains[left] == domains[right]:
                continue
            cosine = float(np.dot(fingerprints[left], fingerprints[right]))
            if cosine >= cosine_threshold:
                pairs.append(
                    {
                        "left_recording": str(recording_ids[left]),
                        "right_recording": str(recording_ids[right]),
                        "cosine": cosine,
                    }
                )
    return {
        "method": "cross_role_recording_mean_log_envelope_spectrum_256_cosine",
        "cosine_threshold": cosine_threshold,
        "cross_domain_pair_count": len(pairs),
        "cross_partition_pair_count": len(pairs),
        "pairs": pairs,
    }
