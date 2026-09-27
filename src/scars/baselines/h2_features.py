from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from time import perf_counter
from typing import Protocol

import numpy as np

from scars.representations.normalization import SourceGlobalNormalizer
from scars.representations.resize import resize_2d
from scars.representations.stft_anchor import stft_anchor
from scars.representations.tensor import RepresentationConfig, SourceFittedTensor


H2_BANK_ORDER = (
    "raw_iq",
    "log_stft",
    "cwt",
    "wst_J3Q2",
    "wst_J4Q2",
    "wst_J5Q2",
    "cyclic_A4",
    "cyclic_A8",
    "cyclic_A16",
    "wst_cyclic_resolution_8",
    "wst_cyclic_resolution_16",
    "wst_cyclic_resolution_32",
)


class SourceFittedRepresentation(Protocol):
    config: object

    def fit(
        self,
        source_windows: np.ndarray,
        source_fold: str,
        fit_split_kind: str = "source_fit",
        recording_ids: np.ndarray | None = None,
    ) -> "SourceFittedRepresentation": ...

    def transform(self, windows: np.ndarray) -> np.ndarray: ...

    def transform_by_family(self, windows: np.ndarray) -> dict[str, np.ndarray]: ...

    def measured_cost(self, windows: np.ndarray, warmups: int = 2, repeats: int = 5) -> dict[str, float]: ...

    def source_artifact(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class ClassicalConfig:
    stable_id: str
    kind: str
    output_bins: int = 16
    cwt_scales: int = 16
    frame_samples: int = 128
    hop_samples: int = 32
    normalization: str = "percentile"


class SourceFittedClassical:
    """Source-fitted raw-IQ, log-STFT, and CWT controls for the H2 bank."""

    def __init__(self, config: ClassicalConfig):
        if config.kind not in {"raw_iq", "log_stft", "cwt"}:
            raise ValueError(f"Unknown classical feature kind {config.kind}")
        self.config = config
        self.normalizer = SourceGlobalNormalizer(config.normalization)
        self.source_fold: str | None = None
        self.fitted = False

    def _cwt(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        frequency = np.fft.fftfreq(n)
        x_fft = np.fft.fft(x)
        rows = []
        for center in np.geomspace(0.02, 0.45, self.config.cwt_scales):
            bandwidth = max(float(center) / 4.0, 0.008)
            wavelet = np.exp(-0.5 * ((frequency - center) / bandwidth) ** 2)
            wavelet[frequency < 0] = 0.0
            rows.append(np.log1p(np.abs(np.fft.ifft(x_fft * wavelet))))
        return resize_2d(np.stack(rows), self.config.output_bins, self.config.output_bins)

    def _raw(self, x: np.ndarray) -> dict[str, np.ndarray]:
        b = self.config.output_bins
        if self.config.kind == "raw_iq":
            # The channel axis is explicit; resizing only establishes a common
            # fixed probe dimension and does not mix I with Q.
            return {
                "I": resize_2d(np.real(x)[None, :], b, b),
                "Q": resize_2d(np.imag(x)[None, :], b, b),
            }
        if self.config.kind == "log_stft":
            return {"S": stft_anchor(x, b, self.config.frame_samples, self.config.hop_samples)}
        return {"CWT": self._cwt(x)}

    def fit(
        self,
        source_windows: np.ndarray,
        source_fold: str,
        fit_split_kind: str = "source_fit",
        recording_ids: np.ndarray | None = None,
    ) -> "SourceFittedClassical":
        del recording_ids
        if fit_split_kind != "source_fit":
            raise PermissionError("Classical controls may be fit only on source_fit")
        if len(source_windows) == 0:
            raise ValueError("Cannot fit on zero source windows")
        by_family: dict[str, list[np.ndarray]] = {}
        for x in source_windows:
            for family, value in self._raw(x).items():
                by_family.setdefault(family, []).append(value)
        for family, values in by_family.items():
            self.normalizer.fit_family(family, np.stack(values), source_fold, fit_split_kind)
        self.source_fold = source_fold
        self.fitted = True
        return self

    def transform_one_by_family(self, x: np.ndarray) -> dict[str, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("SourceFittedClassical must be fitted first")
        return {
            family: self.normalizer.transform(family, value)
            for family, value in self._raw(x).items()
        }

    def transform_by_family(self, windows: np.ndarray) -> dict[str, np.ndarray]:
        batches = [self.transform_one_by_family(x) for x in windows]
        return {
            family: np.stack([batch[family] for batch in batches])
            for family in batches[0]
        }

    def transform(self, windows: np.ndarray) -> np.ndarray:
        families = self.transform_by_family(windows)
        return np.concatenate([families[name][:, None] for name in families], axis=1)

    def measured_cost(
        self, windows: np.ndarray, warmups: int = 2, repeats: int = 5
    ) -> dict[str, float]:
        subset = windows[:1]
        for _ in range(warmups):
            self.transform(subset)
        samples = []
        for _ in range(repeats):
            start = perf_counter()
            self.transform(subset)
            samples.append(1000.0 * (perf_counter() - start) / len(subset))
        n = windows.shape[1]
        family_count = 2 if self.config.kind == "raw_iq" else 1
        fft_terms = 0 if self.config.kind == "raw_iq" else (
            self.config.cwt_scales if self.config.kind == "cwt" else 1
        )
        return {
            "bytes_per_sample": float(family_count * self.config.output_bins**2 * 4),
            "batch1_latency_ms": float(np.median(samples)),
            "estimated_macs": float(
                max(n, self.config.output_bins**2 * family_count)
                if fft_terms == 0
                else fft_terms * n * np.log2(max(n, 2)) * 5.0
            ),
            "warmups": float(warmups),
            "repeats": float(repeats),
            "samples_ms": [float(value) for value in samples],
            "batch_size": 1,
            "synchronized": True,
            "measurement_valid": bool(
                len(samples) == repeats and np.all(np.isfinite(samples))
            ),
        }

    def source_artifact(self) -> dict[str, object]:
        if not self.fitted:
            raise RuntimeError("No source artifact before fit")
        return {
            "representation_type": "classical",
            "config": asdict(self.config),
            "source_fold": self.source_fold,
            "normalizers": {
                family: asdict(record) for family, record in self.normalizer.records.items()
            },
        }

    @classmethod
    def from_source_artifact(cls, artifact: dict[str, object]) -> "SourceFittedClassical":
        instance = cls(ClassicalConfig(**artifact["config"]))
        instance.source_fold = str(artifact["source_fold"])
        instance.normalizer.load_records(artifact["normalizers"])
        instance.fitted = True
        return instance


def h2_configuration_factory() -> list[SourceFittedRepresentation]:
    configurations: list[SourceFittedRepresentation] = [
        SourceFittedClassical(ClassicalConfig("raw_iq", "raw_iq")),
        SourceFittedClassical(ClassicalConfig("log_stft", "log_stft")),
        SourceFittedClassical(ClassicalConfig("cwt", "cwt")),
        SourceFittedTensor(RepresentationConfig("wst_J3Q2", use_w=True, use_c=False, output_bins=16, wst_j=3, wst_q=2)),
        SourceFittedTensor(RepresentationConfig("wst_J4Q2", use_w=True, use_c=False, output_bins=16, wst_j=4, wst_q=2)),
        SourceFittedTensor(RepresentationConfig("wst_J5Q2", use_w=True, use_c=False, output_bins=16, wst_j=5, wst_q=2)),
        SourceFittedTensor(RepresentationConfig("cyclic_A4", use_w=False, use_c=True, output_bins=16, cyclic_count=4)),
        SourceFittedTensor(RepresentationConfig("cyclic_A8", use_w=False, use_c=True, output_bins=16, cyclic_count=8)),
        SourceFittedTensor(RepresentationConfig("cyclic_A16", use_w=False, use_c=True, output_bins=16, cyclic_count=16)),
        SourceFittedTensor(RepresentationConfig("wst_cyclic_resolution_8", output_bins=8, wst_j=4, wst_q=2, cyclic_count=8)),
        SourceFittedTensor(RepresentationConfig("wst_cyclic_resolution_16", output_bins=16, wst_j=4, wst_q=2, cyclic_count=8)),
        SourceFittedTensor(RepresentationConfig("wst_cyclic_resolution_32", output_bins=32, wst_j=4, wst_q=2, cyclic_count=8)),
    ]
    observed = tuple(item.config.stable_id for item in configurations)
    if observed != H2_BANK_ORDER:
        raise AssertionError(f"H2 bank drift: {observed}")
    return configurations


def effective_h2_signature(representation: SourceFittedRepresentation) -> dict[str, object]:
    """Return every method-affecting H2 field, excluding the display ID."""
    config = representation.config
    if isinstance(config, ClassicalConfig):
        families = {
            "raw_iq": ("I", "Q"),
            "log_stft": ("S",),
            "cwt": ("CWT",),
        }[config.kind]
        return {
            "representation_kind": config.kind,
            "active_families": families,
            "wst_J": None,
            "wst_Q": None,
            "cyclic_count": None,
            "output_resolution": config.output_bins,
            "normalization": config.normalization,
            "frame_samples": config.frame_samples,
            "hop_samples": config.hop_samples,
            "cwt_scales": config.cwt_scales if config.kind == "cwt" else None,
            "probe": "ridge_closed_form_l2_0.01_source_fit_feature_standardization",
        }
    return {
        "representation_kind": "wst_cyclic_tensor",
        "active_families": config.active_families(),
        "wst_J": config.wst_j if config.use_w else None,
        "wst_Q": config.wst_q if config.use_w else None,
        "cyclic_count": config.cyclic_count if config.use_c else None,
        "output_resolution": config.output_bins,
        "normalization": config.normalization,
        "frame_samples": config.frame_samples,
        "hop_samples": config.hop_samples,
        "cwt_scales": None,
        "probe": "ridge_closed_form_l2_0.01_source_fit_feature_standardization",
    }


def assert_unique_h2_effective_configurations() -> dict[str, dict[str, object]]:
    signatures: dict[str, str] = {}
    by_id: dict[str, dict[str, object]] = {}
    for representation in h2_configuration_factory():
        effective = effective_h2_signature(representation)
        signature = json.dumps(effective, sort_keys=True, separators=(",", ":"))
        previous = signatures.get(signature)
        if previous is not None:
            raise AssertionError(
                f"Duplicate effective H2 configurations: {previous} and {representation.config.stable_id}"
            )
        signatures[signature] = representation.config.stable_id
        by_id[representation.config.stable_id] = effective
    return by_id


def load_h2_representation(artifact: dict[str, object]) -> SourceFittedRepresentation:
    if artifact.get("representation_type") == "classical":
        return SourceFittedClassical.from_source_artifact(artifact)
    if artifact.get("representation_type") == "tensor":
        return SourceFittedTensor.from_source_artifact(artifact)
    raise ValueError("Unknown frozen H2 representation type")
