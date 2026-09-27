from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np

from .cyclostationary import discover_cyclic_bins, permute_frozen_bins, spectral_correlation_map
from .energy import normalize_energy, raw_energy_map, source_energy_reference
from .normalization import SourceGlobalNormalizer
from .scattering import ScatteringMetadata, scattering_map, scattering_path_count
from .stft_anchor import stft_anchor


@dataclass(frozen=True)
class RepresentationConfig:
    stable_id: str
    use_w: bool = True
    use_c: bool = True
    use_e: bool = False
    use_s: bool = False
    output_bins: int = 16
    wst_j: int = 4
    wst_q: int = 2
    cyclic_count: int = 8
    frame_samples: int = 128
    hop_samples: int = 32
    normalization: str = "percentile"
    cyclic_permutation: bool = False
    permutation_seed: int = 24024

    def active_families(self) -> tuple[str, ...]:
        return tuple(
            family
            for family, enabled in (
                ("W", self.use_w),
                ("C", self.use_c),
                ("E", self.use_e),
                ("S", self.use_s),
            )
            if enabled
        )


class SourceFittedTensor:
    def __init__(self, config: RepresentationConfig):
        if not config.active_families():
            raise ValueError("At least one channel family is required")
        self.config = config
        self.normalizer = SourceGlobalNormalizer(config.normalization)
        self.cyclic_bins: np.ndarray | None = None
        self.energy_reference: float | None = None
        self.scattering_metadata: ScatteringMetadata | None = None
        self.source_fold: str | None = None
        self.fitted = False

    def _raw(self, x: np.ndarray) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        if self.config.use_w:
            value, metadata = scattering_map(
                x, self.config.wst_j, self.config.wst_q, self.config.output_bins
            )
            result["W"] = value
            self.scattering_metadata = metadata
        if self.config.use_c:
            if self.cyclic_bins is None:
                raise RuntimeError("Cyclic bins must be source-fitted before transform")
            bins = (
                permute_frozen_bins(self.cyclic_bins, self.config.permutation_seed)
                if self.config.cyclic_permutation
                else self.cyclic_bins
            )
            result["C"] = spectral_correlation_map(
                x,
                bins,
                self.config.output_bins,
                frame=self.config.frame_samples,
                hop=self.config.hop_samples,
            )
        if self.config.use_e:
            result["E"] = raw_energy_map(
                x,
                self.config.output_bins,
                frame=self.config.frame_samples,
                hop=self.config.hop_samples,
            )
        if self.config.use_s:
            result["S"] = stft_anchor(
                x,
                self.config.output_bins,
                frame=self.config.frame_samples,
                hop=self.config.hop_samples,
            )
        return result

    def fit(
        self,
        source_windows: np.ndarray,
        source_fold: str,
        fit_split_kind: str = "source_fit",
        recording_ids: np.ndarray | None = None,
    ) -> "SourceFittedTensor":
        if fit_split_kind not in {"source_fit", "source_train"}:
            raise PermissionError("Representation parameters may be fit only on source_fit")
        if len(source_windows) == 0:
            raise ValueError("Cannot fit a representation on zero source windows")
        if recording_ids is None:
            recording_ids = np.asarray([f"source-window-{i}" for i in range(len(source_windows))])
        recording_ids = np.asarray(recording_ids)
        if len(recording_ids) != len(source_windows):
            raise ValueError("recording_ids must align one-to-one with source windows")
        if self.config.use_c:
            self.cyclic_bins = discover_cyclic_bins(
                source_windows,
                self.config.cyclic_count,
                effective_frame=self.config.frame_samples,
                recording_ids=recording_ids,
            )
        else:
            self.cyclic_bins = np.empty(0, dtype=np.float64)
        raw_by_family = {family: [] for family in self.config.active_families()}
        for x in source_windows:
            for family, values in self._raw(x).items():
                raw_by_family[family].append(values)
        for family, values in raw_by_family.items():
            stacked = np.stack(values)
            if family == "E":
                self.energy_reference = source_energy_reference(stacked, recording_ids)
            else:
                self.normalizer.fit_family(family, stacked, source_fold, fit_split_kind)
        self.source_fold = source_fold
        self.fitted = True
        return self

    def transform_one_by_family(self, x: np.ndarray) -> dict[str, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("SourceFittedTensor must be fitted before transformation")
        result: dict[str, np.ndarray] = {}
        for family, values in self._raw(x).items():
            if family == "E":
                if self.energy_reference is None:
                    raise AssertionError("Missing source energy reference")
                result[family] = normalize_energy(values, self.energy_reference).astype(np.float32)
            else:
                result[family] = self.normalizer.transform(family, values)
            if result[family].shape != (self.config.output_bins, self.config.output_bins):
                raise AssertionError(f"{family} has inconsistent shape {result[family].shape}")
        return result

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        families = self.transform_one_by_family(x)
        return np.stack([families[name] for name in self.config.active_families()])

    def transform(self, windows: np.ndarray) -> np.ndarray:
        return np.stack([self.transform_one(x) for x in windows])

    def transform_by_family(self, windows: np.ndarray) -> dict[str, np.ndarray]:
        batches = [self.transform_one_by_family(x) for x in windows]
        return {
            family: np.stack([batch[family] for batch in batches])
            for family in self.config.active_families()
        }

    def measured_cost(
        self, windows: np.ndarray, warmups: int = 2, repeats: int = 5
    ) -> dict[str, float]:
        # Deployment metric is a true batch-of-one call, not an amortized
        # batch-of-four throughput measurement.
        subset = windows[:1]
        for _ in range(warmups):
            self.transform(subset)
        samples = []
        for _ in range(repeats):
            start = perf_counter()
            self.transform(subset)
            samples.append(1000.0 * (perf_counter() - start) / len(subset))
        midpoint = max(1, len(samples) // 2)
        first_half = float(np.median(samples[:midpoint]))
        second_values = samples[midpoint:] or samples[:midpoint]
        second_half = float(np.median(second_values))
        drift_ratio = max(first_half, second_half) / max(min(first_half, second_half), 1.0e-12)
        channels = len(self.config.active_families())
        n = windows.shape[1]
        first_paths, second_paths = scattering_path_count(
            self.config.wst_j, self.config.wst_q
        )
        # W executes one input FFT, one inverse FFT per first path, one FFT for
        # every eligible first-order output, and one inverse FFT per second
        # path. This remains an estimator, but it counts the implemented path
        # graph instead of only J*Q first-order terms.
        fft_terms = int(self.config.use_w) * (2 * first_paths + second_paths)
        fft_terms += int(self.config.use_c) * self.config.cyclic_count
        fft_terms += int(self.config.use_s)
        point_terms = int(self.config.use_e) * n
        # Bilinear resize plus materialization is part of the deployed
        # representation graph and makes the resolution sweep auditable.
        resize_terms = channels * self.config.output_bins**2 * 8
        return {
            "bytes_per_sample": float(channels * self.config.output_bins**2 * 4),
            "batch1_latency_ms": float(np.median(samples)),
            "estimated_macs": float(
                fft_terms * n * np.log2(max(n, 2)) * 5.0
                + point_terms
                + resize_terms
            ),
            "mac_estimator_version": "scars-representation-v2-path-and-resize-counted",
            "warmups": float(warmups),
            "repeats": float(repeats),
            "samples_ms": [float(value) for value in samples],
            "batch_size": 1,
            "synchronized": True,
            "measurement_valid": bool(
                warmups == 20
                and repeats == 100
                and len(samples) == repeats
                and np.all(np.isfinite(samples))
                and drift_ratio <= 1.25
            ),
            "device": "cpu",
            "timing_stability": {
                "first_half_median_ms": first_half,
                "second_half_median_ms": second_half,
                "drift_ratio": drift_ratio,
                "drift_valid": drift_ratio <= 1.25,
            },
        }

    def source_artifact(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("No source artifact before fit")
        return {
            "representation_type": "tensor",
            "config": asdict(self.config),
            "source_fold": self.source_fold,
            "cyclic_frequencies_cycles_per_sample": self.cyclic_bins.tolist()
            if self.cyclic_bins is not None
            else [],
            "effective_cyclic_shifts": [
                max(
                    1,
                    min(
                        self.config.frame_samples - 1,
                        int(np.floor(float(alpha) * self.config.frame_samples + 0.5)),
                    ),
                )
                for alpha in (self.cyclic_bins if self.cyclic_bins is not None else [])
            ],
            "normalizers": {key: asdict(value) for key, value in self.normalizer.records.items()},
            "energy_reference": self.energy_reference,
            "scattering": None
            if self.scattering_metadata is None
            else asdict(self.scattering_metadata),
        }

    @classmethod
    def from_source_artifact(cls, artifact: dict[str, Any]) -> "SourceFittedTensor":
        config = RepresentationConfig(**artifact["config"])
        instance = cls(config)
        instance.cyclic_bins = np.asarray(
            artifact.get("cyclic_frequencies_cycles_per_sample", []), dtype=np.float64
        )
        instance.energy_reference = artifact.get("energy_reference")
        instance.source_fold = artifact.get("source_fold")
        instance.normalizer.load_records(artifact.get("normalizers", {}))
        instance.fitted = True
        return instance
