from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib

import numpy as np


@dataclass(frozen=True)
class NormalizationRecord:
    family: str
    mode: str
    offset: float
    scale: float
    source_fold: str
    fit_timestamp: str
    config_hash: str


class SourceGlobalNormalizer:
    def __init__(self, mode: str = "percentile"):
        if mode not in {"percentile", "zscore", "none"}:
            raise ValueError(f"Unknown normalization mode {mode}")
        self.mode = mode
        self.records: dict[str, NormalizationRecord] = {}

    def fit_family(
        self, family: str, values: np.ndarray, source_fold: str, fit_split_kind: str
    ) -> None:
        if fit_split_kind not in {"source_fit", "source_train"}:
            raise PermissionError("Normalization may be fit only on source_fit")
        values = np.asarray(values, dtype=np.float64)
        if self.mode == "percentile":
            low, high = np.percentile(values, [1.0, 99.0])
            offset, scale = float(low), max(float(high - low), 1.0e-8)
        elif self.mode == "zscore":
            offset, scale = float(values.mean()), max(float(values.std()), 1.0e-8)
        else:
            offset, scale = 0.0, 1.0
        digest = hashlib.sha256(
            f"{family}|{self.mode}|{source_fold}|{offset:.17g}|{scale:.17g}".encode()
        ).hexdigest()
        self.records[family] = NormalizationRecord(
            family=family,
            mode=self.mode,
            offset=offset,
            scale=scale,
            source_fold=source_fold,
            fit_timestamp=datetime.now(timezone.utc).isoformat(),
            config_hash=digest,
        )

    def transform(self, family: str, values: np.ndarray) -> np.ndarray:
        record = self.records[family]
        normalized = (np.asarray(values) - record.offset) / record.scale
        if record.mode == "percentile":
            normalized = np.clip(normalized, 0.0, 1.0)
        return normalized.astype(np.float32)

    def load_records(self, records: dict[str, dict[str, object]]) -> None:
        loaded = {}
        for family, payload in records.items():
            record = NormalizationRecord(**payload)
            if record.mode != self.mode or record.family != family:
                raise ValueError("Frozen normalization record does not match its family/mode")
            loaded[family] = record
        self.records = loaded
