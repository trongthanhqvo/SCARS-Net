"""Disk-backed, deterministic relation tensors for large source campaigns.

This module changes storage only.  A relation sample retains the historical
ordering ``recording-major, registered-nuisance-major`` and the same seeded
perturbation draw as the original in-memory implementation.  It deliberately
does not inspect held-target data or alter the registered sample population.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterator, Sequence

import numpy as np

from scars.data.windowing import WindowBatch
from scars.results.provenance import sha256_file
from scars.selection.nuisance import NuisanceCase, registered_nuisance_cases


SCHEMA_VERSION = "scars-disk-relation-tensors-1.0"


def _json_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _ids_hash(values: Sequence[object]) -> str:
    return _json_hash([str(value) for value in values])


def _atomic_json(path: Path, payload: object) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


@dataclass(frozen=True)
class RelationTensorCache:
    """A validated pair of source-fitted W/C/E/S tensors stored as ``.npy``."""

    directory: Path
    clean_path: Path
    perturbed_path: Path
    manifest_path: Path
    sample_count: int
    active_indices: tuple[int, ...]
    case_ids: tuple[str, ...]
    source_recording_ids: tuple[str, ...]
    source_labels: tuple[str, ...]
    seed: int

    @property
    def clean(self) -> np.memmap:
        return np.load(self.clean_path, mmap_mode="r")

    @property
    def perturbed(self) -> np.memmap:
        return np.load(self.perturbed_path, mmap_mode="r")

    @property
    def case_count(self) -> int:
        return len(self.case_ids)

    def metadata(self, start: int, stop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return recording, label, nuisance and severity in canonical order."""
        if start < 0 or stop < start or stop > self.sample_count:
            raise ValueError("Invalid relation-cache slice")
        cases = registered_nuisance_cases()
        if tuple(case.id for case in cases) != self.case_ids:
            raise RuntimeError("Registered nuisance contract drifted after relation cache creation")
        flat = np.arange(start, stop, dtype=np.int64)
        recording_index = flat // len(cases)
        case_index = flat % len(cases)
        return (
            np.asarray([self.source_recording_ids[index] for index in recording_index], dtype=object),
            np.asarray([self.source_labels[index] for index in recording_index], dtype=object),
            np.asarray([cases[index].id.split(":", 1)[0] for index in case_index], dtype=object),
            np.asarray([float(cases[index].severity) for index in case_index], dtype=float),
        )

    def iter_iq_chunks(
        self, batch: WindowBatch, *, chunk_size: int
    ) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
        """Regenerate the exact clean/perturbed IQ pairs without retaining them."""
        _validate_source_batch(batch, self)
        yield from iter_relation_iq_chunks(batch, seed=self.seed, chunk_size=chunk_size)


def _validate_source_batch(batch: WindowBatch, cache: RelationTensorCache) -> None:
    if _ids_hash(batch.recording_ids) != _ids_hash(cache.source_recording_ids):
        raise RuntimeError("Relation cache source recording order/provenance drift")
    if _ids_hash(batch.labels) != _ids_hash(cache.source_labels):
        raise RuntimeError("Relation cache source label order/provenance drift")
    if len(batch.iq) * cache.case_count != cache.sample_count:
        raise RuntimeError("Relation cache source sample count drift")


def iter_relation_iq_chunks(
    batch: WindowBatch, *, seed: int, chunk_size: int
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield the historical relation expansion in bounded deterministic chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    cases = registered_nuisance_cases()
    case_count = len(cases)
    total = len(batch.iq) * case_count
    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        count = stop - start
        clean = np.empty((count, batch.iq.shape[1]), dtype=np.complex64)
        perturbed = np.empty_like(clean)
        for local, flat_index in enumerate(range(start, stop)):
            window_index = flat_index // case_count
            case_index = flat_index % case_count
            window = batch.iq[window_index]
            case = cases[case_index]
            # Keep precisely the original random-draw identity.
            rng = np.random.default_rng(seed + 1009 * window_index + case_index)
            clean[local] = window
            perturbed[local] = case.transform(window, rng, case.severity)
        yield start, stop, clean, perturbed


def _validate_existing_cache(
    directory: Path,
    *,
    batch: WindowBatch,
    seed: int,
    active_indices: Sequence[int],
    representation_sha256: str,
) -> RelationTensorCache | None:
    manifest_path = directory / "manifest.json"
    clean_path = directory / "clean.npy"
    perturbed_path = directory / "perturbed.npy"
    if not (manifest_path.is_file() and clean_path.is_file() and perturbed_path.is_file()):
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "active_indices": list(map(int, active_indices)),
        "representation_sha256": representation_sha256,
        "source_recording_ids_sha256": _ids_hash(batch.recording_ids),
        "source_labels_sha256": _ids_hash(batch.labels),
        "case_ids": [case.id for case in registered_nuisance_cases()],
        "sample_count": len(batch.iq) * len(registered_nuisance_cases()),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Existing relation cache provenance drift at {key}")
    for path, key in ((clean_path, "clean_sha256"), (perturbed_path, "perturbed_sha256")):
        if sha256_file(path) != payload.get(key):
            raise RuntimeError(f"Existing relation cache checksum mismatch: {path.name}")
    clean = np.load(clean_path, mmap_mode="r")
    perturbed = np.load(perturbed_path, mmap_mode="r")
    if clean.shape != perturbed.shape or clean.shape[0] != expected["sample_count"]:
        raise RuntimeError("Existing relation cache tensor shape mismatch")
    return RelationTensorCache(
        directory=directory,
        clean_path=clean_path,
        perturbed_path=perturbed_path,
        manifest_path=manifest_path,
        sample_count=int(expected["sample_count"]),
        active_indices=tuple(map(int, active_indices)),
        case_ids=tuple(expected["case_ids"]),
        source_recording_ids=tuple(map(str, batch.recording_ids)),
        source_labels=tuple(map(str, batch.labels)),
        seed=int(seed),
    )


def build_relation_tensor_cache(
    directory: Path,
    batch: WindowBatch,
    representation: object,
    *,
    seed: int,
    active_indices: Sequence[int],
    representation_sha256: str,
    chunk_size: int = 64,
) -> RelationTensorCache:
    """Materialize source-fitted tensors to disk, never full IQ relation arrays.

    The final directory is published only after both arrays and their hashes are
    complete.  Interrupted runs leave a separate temporary directory and are
    rebuilt; they are never silently reused.
    """
    existing = _validate_existing_cache(
        directory,
        batch=batch,
        seed=seed,
        active_indices=active_indices,
        representation_sha256=representation_sha256,
    )
    if existing is not None:
        return existing
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=directory.name + ".partial-", dir=directory.parent))
    try:
        first = next(iter_relation_iq_chunks(batch, seed=seed, chunk_size=min(chunk_size, 1)))
        _, _, first_clean_iq, first_perturbed_iq = first
        first_clean = np.asarray(representation.transform(first_clean_iq), dtype=np.float32)[:, active_indices]
        first_perturbed = np.asarray(representation.transform(first_perturbed_iq), dtype=np.float32)[:, active_indices]
        if first_clean.shape != first_perturbed.shape or first_clean.ndim != 4:
            raise ValueError("Representation must produce aligned [N,C,H,W] tensors")
        sample_count = len(batch.iq) * len(registered_nuisance_cases())
        shape = (sample_count, *first_clean.shape[1:])
        clean_path = temporary / "clean.npy"
        perturbed_path = temporary / "perturbed.npy"
        clean_out = np.lib.format.open_memmap(clean_path, mode="w+", dtype=np.float32, shape=shape)
        perturbed_out = np.lib.format.open_memmap(perturbed_path, mode="w+", dtype=np.float32, shape=shape)
        for start, stop, clean_iq, perturbed_iq in iter_relation_iq_chunks(
            batch, seed=seed, chunk_size=chunk_size
        ):
            clean_tensor = np.asarray(representation.transform(clean_iq), dtype=np.float32)[:, active_indices]
            perturbed_tensor = np.asarray(representation.transform(perturbed_iq), dtype=np.float32)[:, active_indices]
            if clean_tensor.shape != (stop - start, *shape[1:]) or perturbed_tensor.shape != clean_tensor.shape:
                raise RuntimeError("Representation output shape drift during relation cache construction")
            clean_out[start:stop] = clean_tensor
            perturbed_out[start:stop] = perturbed_tensor
        clean_out.flush()
        perturbed_out.flush()
        del clean_out, perturbed_out
        cases = registered_nuisance_cases()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "storage": "disk_backed_npy_float32",
            "seed": int(seed),
            "chunk_size": int(chunk_size),
            "active_indices": list(map(int, active_indices)),
            "representation_sha256": representation_sha256,
            "source_recording_ids_sha256": _ids_hash(batch.recording_ids),
            "source_labels_sha256": _ids_hash(batch.labels),
            "case_ids": [case.id for case in cases],
            "sample_count": int(sample_count),
            "tensor_shape": list(shape),
            "clean_sha256": sha256_file(clean_path),
            "perturbed_sha256": sha256_file(perturbed_path),
        }
        _atomic_json(temporary / "manifest.json", payload)
        os.replace(temporary, directory)
    except Exception:
        # Do not delete partial evidence: a user may inspect it after a failed run.
        raise
    return _validate_existing_cache(
        directory,
        batch=batch,
        seed=seed,
        active_indices=active_indices,
        representation_sha256=representation_sha256,
    ) or (_ for _ in ()).throw(RuntimeError("Relation-cache publication failed"))


class ChannelSubsetArray:
    """Lazy channel subset that prevents NumPy fancy indexing of a full memmap."""

    def __init__(self, array: np.ndarray, indices: Sequence[int]):
        self.array = array
        self.indices = tuple(map(int, indices))
        if array.ndim != 4 or not self.indices:
            raise ValueError("ChannelSubsetArray requires non-empty [N,C,H,W] input")
        self.shape = (len(array), len(self.indices), *array.shape[2:])
        self.dtype = array.dtype

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index):
        value = np.asarray(self.array[index])
        if value.ndim == 3:
            return value[list(self.indices)]
        if value.ndim == 4:
            return value[:, self.indices]
        raise IndexError("Unsupported tensor index for ChannelSubsetArray")


def fit_global_channel_statistics(
    values: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray],
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Source-fit global mean/std without a dataset-sized float64 temporary."""
    if chunk_size <= 0 or len(values) == 0:
        raise ValueError("Statistics require a non-empty positive-sized source iterator")
    sums: np.ndarray | None = None
    sumsquares: np.ndarray | None = None
    count = 0
    feature_ndim: int | None = None
    for start in range(0, len(values), chunk_size):
        chunk = np.asarray(transform(np.asarray(values[start : start + chunk_size])), dtype=np.float64)
        if chunk.ndim < 2:
            raise ValueError("Feature transform must return [N,C,...]")
        axes = (0, *range(2, chunk.ndim))
        part_sum = chunk.sum(axis=axes)
        part_square = np.square(chunk).sum(axis=axes)
        part_count = int(np.prod([chunk.shape[index] for index in axes], dtype=np.int64))
        sums = part_sum if sums is None else sums + part_sum
        sumsquares = part_square if sumsquares is None else sumsquares + part_square
        count += part_count
        feature_ndim = chunk.ndim
    assert sums is not None and sumsquares is not None and feature_ndim is not None
    mean = sums / count
    variance = np.maximum(sumsquares / count - np.square(mean), 0.0)
    shape = (1, len(mean), *([1] * (feature_ndim - 2)))
    return mean.reshape(shape), np.maximum(np.sqrt(variance), 1.0e-6).reshape(shape)


def build_disk_feature_pair(
    directory: Path,
    relation_cache: RelationTensorCache,
    relation_batch: WindowBatch,
    transform: Callable[[np.ndarray], np.ndarray],
    *,
    source_fit_iq: np.ndarray,
    validation_iq: np.ndarray,
    chunk_size: int = 64,
) -> tuple[np.memmap, np.memmap, np.ndarray, dict[str, object]]:
    """Create one disk-backed paired baseline input and source-only normalizer."""
    directory.mkdir(parents=True, exist_ok=True)
    mean, scale = fit_global_channel_statistics(source_fit_iq, transform, chunk_size=chunk_size)
    first_start, first_stop, first_clean_iq, first_perturbed_iq = next(
        relation_cache.iter_iq_chunks(relation_batch, chunk_size=min(chunk_size, 1))
    )
    del first_start, first_stop
    first_clean = np.asarray(transform(first_clean_iq), dtype=np.float32)
    first_perturbed = np.asarray(transform(first_perturbed_iq), dtype=np.float32)
    if first_clean.shape != first_perturbed.shape or first_clean.ndim < 3:
        raise ValueError("Baseline transform must produce aligned [N,C,...] tensors")
    shape = (relation_cache.sample_count, *first_clean.shape[1:])
    clean_path = directory / "clean.npy"
    perturbed_path = directory / "perturbed.npy"
    clean_out = np.lib.format.open_memmap(clean_path, mode="w+", dtype=np.float32, shape=shape)
    perturbed_out = np.lib.format.open_memmap(perturbed_path, mode="w+", dtype=np.float32, shape=shape)
    for start, stop, clean_iq, perturbed_iq in relation_cache.iter_iq_chunks(
        relation_batch, chunk_size=chunk_size
    ):
        clean_out[start:stop] = ((np.asarray(transform(clean_iq)) - mean) / scale).astype(np.float32)
        perturbed_out[start:stop] = ((np.asarray(transform(perturbed_iq)) - mean) / scale).astype(np.float32)
    clean_out.flush()
    perturbed_out.flush()
    del clean_out, perturbed_out
    validation = ((np.asarray(transform(validation_iq)) - mean) / scale).astype(np.float32)
    return (
        np.load(clean_path, mmap_mode="r"),
        np.load(perturbed_path, mmap_mode="r"),
        validation,
        {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "axis_contract": "global_scalar_per_input_channel_source_fit_only",
            "storage": "disk_backed_npy_float32",
        },
    )


def build_disk_representation_pair(
    directory: Path,
    relation_cache: RelationTensorCache,
    relation_batch: WindowBatch,
    representation: object,
    validation_iq: np.ndarray,
    *,
    chunk_size: int = 64,
) -> tuple[np.memmap, np.memmap, np.ndarray, dict[str, object]]:
    """Disk-backed pair for an already source-fitted representation baseline."""
    directory.mkdir(parents=True, exist_ok=True)
    _, _, first_clean_iq, first_perturbed_iq = next(
        relation_cache.iter_iq_chunks(relation_batch, chunk_size=min(chunk_size, 1))
    )
    first_clean = np.asarray(representation.transform(first_clean_iq), dtype=np.float32)
    first_perturbed = np.asarray(representation.transform(first_perturbed_iq), dtype=np.float32)
    if first_clean.shape != first_perturbed.shape or first_clean.ndim != 4:
        raise ValueError("Representation baseline must produce aligned [N,C,H,W] tensors")
    shape = (relation_cache.sample_count, *first_clean.shape[1:])
    clean_path = directory / "clean.npy"
    perturbed_path = directory / "perturbed.npy"
    clean_out = np.lib.format.open_memmap(clean_path, mode="w+", dtype=np.float32, shape=shape)
    perturbed_out = np.lib.format.open_memmap(perturbed_path, mode="w+", dtype=np.float32, shape=shape)
    for start, stop, clean_iq, perturbed_iq in relation_cache.iter_iq_chunks(
        relation_batch, chunk_size=chunk_size
    ):
        clean_out[start:stop] = np.asarray(representation.transform(clean_iq), dtype=np.float32)
        perturbed_out[start:stop] = np.asarray(representation.transform(perturbed_iq), dtype=np.float32)
    clean_out.flush()
    perturbed_out.flush()
    del clean_out, perturbed_out
    return (
        np.load(clean_path, mmap_mode="r"),
        np.load(perturbed_path, mmap_mode="r"),
        np.asarray(representation.transform(validation_iq), dtype=np.float32),
        {"storage": "disk_backed_npy_float32"},
    )


def build_teacher_family_training_array(
    directory: Path,
    batch: WindowBatch,
    representation: object,
    *,
    family_index: int,
    seed: int,
    chunk_size: int = 64,
) -> np.memmap:
    """Materialize one historical clean/one-cycle-perturbed teacher bank.

    The ordering is deliberately identical to the old implementation:
    ``[all clean source_fit windows, all perturb-cycle source_fit windows]``.
    Only a single representation-family channel is stored and the temporary
    file can be released immediately after that family teacher is frozen.
    """
    directory.mkdir(parents=True, exist_ok=True)
    if len(batch.iq) == 0:
        raise ValueError("Teacher bank requires source-fit windows")
    first_iq = np.asarray(batch.iq[:1], dtype=np.complex64)
    first_tensor = np.asarray(representation.transform(first_iq), dtype=np.float32)
    if first_tensor.ndim != 4 or family_index < 0 or family_index >= first_tensor.shape[1]:
        raise ValueError("Invalid family index for teacher tensor")
    count = len(batch.iq)
    path = directory / "family_train.npy"
    output = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(2 * count, 1, *first_tensor.shape[2:]),
    )
    cases = registered_nuisance_cases()
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        clean_iq = np.asarray(batch.iq[start:stop], dtype=np.complex64)
        perturbed_iq = np.empty_like(clean_iq)
        for offset, window in enumerate(clean_iq):
            index = start + offset
            case = cases[index % len(cases)]
            perturbed_iq[offset] = case.transform(
                window, np.random.default_rng(seed + index), case.severity
            )
        clean_tensor = np.asarray(representation.transform(clean_iq), dtype=np.float32)
        perturbed_tensor = np.asarray(representation.transform(perturbed_iq), dtype=np.float32)
        output[start:stop, 0] = clean_tensor[:, family_index]
        output[count + start:count + stop, 0] = perturbed_tensor[:, family_index]
    output.flush()
    del output
    return np.load(path, mmap_mode="r")
