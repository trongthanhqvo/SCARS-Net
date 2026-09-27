from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

import numpy as np

from scars.models.baselines import EarlyFusionResNet
from scars.training.baselines import fit_classifier_with_oom_backoff, predict_classifier
from scars.training.engine import TrainingSpec, resolve_device


SMALL_RESNET_SPEC = {
    "widths": [32, 64, 128],
    "blocks_per_stage": [1, 1, 1],
    "global_pool": True,
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "normalization": "GroupNorm",
    "activation": "GELU",
    "class_weighting": "inverse_frequency_source_fit",
    "batch_size": 32,
    "effective_batch_size": 64,
    "minimum_batch_size": 4,
    "amp": True,
    "max_epochs": 100,
    "source_validation_early_stopping_patience": 10,
}


def seed_everything(seed: int) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    state: dict[str, object] = {"seed": seed, "numpy": True, "python": True, "torch": False}
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        state["torch"] = True
    except ImportError:
        pass
    return state


def _make_model(input_channels: int, class_count: int):
    return EarlyFusionResNet(input_channels, class_count, base_width=32)


class SmallResNetProbe:
    def __init__(self, seed: int, device: str | None = None):
        self.seed = seed
        self.device_name = device
        self.model = None
        self.classes_: np.ndarray | None = None
        self.best_epoch: int | None = None
        self.best_source_validation_macro_f1: float | None = None

    def fit(
        self,
        train_tensor: np.ndarray,
        train_labels: np.ndarray,
        validation_tensor: np.ndarray,
        validation_labels: np.ndarray,
        validation_recording_ids: np.ndarray,
    ) -> "SmallResNetProbe":
        seed_everything(self.seed)
        self.classes_ = np.unique(train_labels)
        spec = TrainingSpec(
            learning_rate=float(SMALL_RESNET_SPEC["learning_rate"]),
            weight_decay=float(SMALL_RESNET_SPEC["weight_decay"]),
            max_epochs=int(SMALL_RESNET_SPEC["max_epochs"]),
            patience=int(SMALL_RESNET_SPEC["source_validation_early_stopping_patience"]),
            batch_size=int(SMALL_RESNET_SPEC["batch_size"]),
            effective_batch_size=int(SMALL_RESNET_SPEC["effective_batch_size"]),
            minimum_batch_size=int(SMALL_RESNET_SPEC["minimum_batch_size"]),
            amp=bool(SMALL_RESNET_SPEC["amp"]),
        )
        artifact = fit_classifier_with_oom_backoff(
            lambda: _make_model(train_tensor.shape[1], len(self.classes_)),
            train_tensor,
            train_labels,
            validation_tensor,
            validation_labels,
            validation_recording_ids,
            seed=self.seed,
            spec=spec,
            device_name=self.device_name or "auto",
        )
        self.model = artifact.model
        self.device_name = artifact.report.device
        self.best_epoch = artifact.report.best_epoch
        self.best_source_validation_macro_f1 = artifact.report.best_source_validation_macro_f1
        return self

    def predict_proba(self, tensor: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("SmallResNetProbe is not fitted")
        import torch

        device = resolve_device(self.device_name or "auto")
        return predict_classifier(
            self.model,
            tensor,
            device=device,
            batch_size=int(SMALL_RESNET_SPEC["batch_size"]),
        )

    def save_artifact(self, path: Path) -> dict[str, Any]:
        if self.model is None or self.classes_ is None:
            raise RuntimeError("No learned artifact before fit")
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "state_dict": {
                    key: value.detach().cpu() for key, value in self.model.state_dict().items()
                },
                "classes": self.classes_.tolist(),
                "spec": SMALL_RESNET_SPEC,
                "seed": self.seed,
                "best_epoch": self.best_epoch,
                "best_source_validation_macro_f1": self.best_source_validation_macro_f1,
            },
            temporary,
        )
        temporary.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "path": str(path),
            "sha256": digest,
            "seed": self.seed,
            "device": self.device_name,
            "best_epoch": self.best_epoch,
            "best_source_validation_macro_f1": self.best_source_validation_macro_f1,
            "spec": SMALL_RESNET_SPEC,
        }
