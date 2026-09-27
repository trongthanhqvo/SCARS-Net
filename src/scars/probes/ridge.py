from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


@dataclass
class RidgeProbe:
    l2: float = 0.01
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    weights_: np.ndarray | None = None
    classes_: np.ndarray | None = None
    solver_: str | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "RidgeProbe":
        x = np.asarray(features, dtype=np.float64).reshape(len(features), -1)
        y = np.asarray(labels)
        self.classes_ = np.unique(y)
        self.mean_ = x.mean(axis=0)
        self.scale_ = np.maximum(x.std(axis=0), 1.0e-6)
        z = np.column_stack(((x - self.mean_) / self.scale_, np.ones(len(x))))
        target = np.stack([(y == label).astype(float) for label in self.classes_], axis=1)
        # The primal and dual closed forms are mathematically equivalent.  Use
        # the smaller Gram matrix so memory scales with min(n_samples,
        # n_features + 1)^2 rather than always with n_samples^2.  This matters
        # for the real MAT corpora, where source_fit can contain tens of
        # thousands of windows but the SCARS tensor has only 1024 features.
        if z.shape[1] <= z.shape[0]:
            self.solver_ = "primal"
            gram = z.T @ z
            gram.flat[:: gram.shape[0] + 1] += self.l2
            self.weights_ = np.linalg.solve(gram, z.T @ target)
        else:
            self.solver_ = "dual"
            gram = z @ z.T
            gram.flat[:: gram.shape[0] + 1] += self.l2
            dual = np.linalg.solve(gram, target)
            self.weights_ = z.T @ dual
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        if self.weights_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("RidgeProbe is not fitted")
        x = np.asarray(features, dtype=np.float64).reshape(len(features), -1)
        z = np.column_stack(((x - self.mean_) / self.scale_, np.ones(len(x))))
        return z @ self.weights_

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = self.decision_function(features)
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1.0e-12)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("RidgeProbe is not fitted")
        return self.classes_[np.argmax(self.predict_proba(features), axis=1)]

    def source_artifact(self) -> dict[str, object]:
        if any(value is None for value in (self.mean_, self.scale_, self.weights_, self.classes_)):
            raise RuntimeError("No ridge artifact before fit")

        def array_record(value: np.ndarray) -> dict[str, object]:
            contiguous = np.ascontiguousarray(value)
            values = contiguous.tolist()
            encoded = (
                json.dumps(values, sort_keys=True).encode()
                if contiguous.dtype.kind == "O"
                else contiguous.view(np.uint8)
            )
            return {
                "dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "values": values,
            }

        return {
            "probe": "ridge_closed_form",
            "l2": self.l2,
            "solver": self.solver_,
            "mean": array_record(self.mean_),
            "scale": array_record(self.scale_),
            "weights": array_record(self.weights_),
            "classes": array_record(self.classes_),
        }

    @classmethod
    def from_source_artifact(cls, artifact: dict[str, object]) -> "RidgeProbe":
        instance = cls(l2=float(artifact["l2"]))
        instance.solver_ = str(artifact.get("solver", "artifact_weights"))
        instance.mean_ = np.asarray(artifact["mean"]["values"], dtype=np.float64)
        instance.scale_ = np.asarray(artifact["scale"]["values"], dtype=np.float64)
        instance.weights_ = np.asarray(artifact["weights"]["values"], dtype=np.float64)
        instance.classes_ = np.asarray(artifact["classes"]["values"])
        return instance
