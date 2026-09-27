from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from scars.evaluation.classification import recording_level_metrics
from scars.training.engine import (
    TrainingReport,
    TrainingSpec,
    make_grad_scaler,
    resolve_device,
    seed_everything,
)


@dataclass(frozen=True)
class LearnedBaselineArtifact:
    model: nn.Module
    classes: np.ndarray
    report: TrainingReport


class PairedBaselineDataset(Dataset):
    """Same sample unit and shuffle order as SCARS-Net's paired trainer."""

    def __init__(self, clean: np.ndarray, perturbed: np.ndarray, labels: np.ndarray):
        if clean.shape != perturbed.shape or len(clean) != len(labels):
            raise ValueError("clean, perturbed and labels must align")
        self.clean = clean
        self.perturbed = perturbed
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            torch.as_tensor(np.asarray(self.clean[index]), dtype=torch.float32),
            torch.as_tensor(np.asarray(self.perturbed[index]), dtype=torch.float32),
            torch.as_tensor(self.labels[index], dtype=torch.long),
        )


def _class_weights(labels: np.ndarray, class_count: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=class_count)
    if np.any(counts == 0):
        raise ValueError("Every class must occur in source_fit")
    values = len(labels) / (class_count * counts.astype(np.float64))
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def predict_classifier(
    model: nn.Module,
    array: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(array), batch_size):
            batch = torch.as_tensor(
                np.asarray(array[start : start + batch_size]),
                dtype=torch.float32,
                device=device,
            )
            output.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    if not output:
        raise ValueError("Cannot predict an empty array")
    return np.concatenate(output)


def _fit_once(
    model_factory: Callable[[], nn.Module],
    train_array: np.ndarray,
    train_labels: np.ndarray,
    validation_array: np.ndarray,
    validation_labels: np.ndarray,
    validation_recording_ids: np.ndarray,
    *,
    seed: int,
    spec: TrainingSpec,
    batch_size: int,
    device: torch.device,
    perturbed_train_array: np.ndarray | None = None,
) -> LearnedBaselineArtifact:
    seed_everything(seed)
    started = perf_counter()
    classes = np.unique(train_labels)
    class_index = {label: index for index, label in enumerate(classes)}
    y_train = np.asarray([class_index[value] for value in train_labels], dtype=np.int64)
    model = model_factory().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay
    )
    weights = _class_weights(y_train, len(classes), device)
    use_amp = bool(spec.amp and device.type == "cuda")
    scaler = make_grad_scaler(use_amp)
    if spec.effective_batch_size < batch_size or spec.effective_batch_size % batch_size:
        raise ValueError("effective_batch_size must be divisible by the CUDA micro-batch")
    accumulation_steps = spec.effective_batch_size // batch_size
    generator = torch.Generator().manual_seed(seed)
    dataset = (
        TensorDataset(
            torch.as_tensor(train_array, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.long),
        )
        if perturbed_train_array is None
        else PairedBaselineDataset(train_array, perturbed_train_array, y_train)
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=spec.num_workers,
        pin_memory=device.type == "cuda",
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_metric = -np.inf
    best_epoch = -1
    stale = 0
    epochs_completed = 0
    for epoch in range(spec.max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, values in enumerate(loader):
            if perturbed_train_array is None:
                batch, target = values
                perturbed = None
            else:
                batch, perturbed, target = values
                perturbed = perturbed.to(device, non_blocking=True)
            batch = batch.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=use_amp,
            ):
                loss = F.cross_entropy(model(batch), target, weight=weights)
                if perturbed is not None:
                    loss = loss + F.cross_entropy(model(perturbed), target, weight=weights)
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            boundary = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader)
            if boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), spec.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        epochs_completed = epoch + 1
        probability = predict_classifier(
            model, validation_array, device=device, batch_size=batch_size
        )
        metric = float(
            recording_level_metrics(
                validation_labels,
                validation_recording_ids,
                probability,
                classes,
            )["macro_f1"]
        )
        if metric > best_metric + 1.0e-12:
            best_metric = metric
            best_epoch = epoch
            best_state = deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
            stale = 0
        else:
            stale += 1
            if stale >= spec.patience:
                break
    if best_state is None:
        raise RuntimeError("Learned baseline produced no source-validation checkpoint")
    model.load_state_dict(best_state)
    report = TrainingReport(
        seed=seed,
        device=str(device),
        amp=use_amp,
        batch_size=batch_size,
        best_epoch=best_epoch,
        best_source_validation_macro_f1=best_metric,
        epochs_completed=epochs_completed,
        peak_gpu_memory_bytes=(
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        elapsed_sec=float(perf_counter() - started),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )
    return LearnedBaselineArtifact(model=model, classes=classes, report=report)


def fit_classifier_with_oom_backoff(
    model_factory: Callable[[], nn.Module],
    train_array: np.ndarray,
    train_labels: np.ndarray,
    validation_array: np.ndarray,
    validation_labels: np.ndarray,
    validation_recording_ids: np.ndarray,
    *,
    seed: int,
    spec: TrainingSpec = TrainingSpec(),
    device_name: str = "auto",
) -> LearnedBaselineArtifact:
    """Fit one learned control without ever using held-target observations."""
    device = resolve_device(device_name)
    batch_size = spec.batch_size
    while batch_size >= spec.minimum_batch_size:
        try:
            return _fit_once(
                model_factory,
                train_array,
                train_labels,
                validation_array,
                validation_labels,
                validation_recording_ids,
                seed=seed,
                spec=spec,
                batch_size=batch_size,
                device=device,
            )
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda":
                raise
            torch.cuda.empty_cache()
            batch_size //= 2
    raise RuntimeError(
        f"CUDA OOM persisted below minimum batch size {spec.minimum_batch_size}"
    )


def fit_paired_classifier_with_oom_backoff(
    model_factory: Callable[[], nn.Module],
    clean_train_array: np.ndarray,
    perturbed_train_array: np.ndarray,
    train_labels: np.ndarray,
    validation_array: np.ndarray,
    validation_labels: np.ndarray,
    validation_recording_ids: np.ndarray,
    *,
    seed: int,
    spec: TrainingSpec = TrainingSpec(),
    device_name: str = "auto",
) -> LearnedBaselineArtifact:
    """Fit a baseline on exactly the paired clean/perturbed SCARS sample unit."""
    device = resolve_device(device_name)
    batch_size = spec.batch_size
    while batch_size >= spec.minimum_batch_size:
        try:
            return _fit_once(
                model_factory,
                clean_train_array,
                train_labels,
                validation_array,
                validation_labels,
                validation_recording_ids,
                seed=seed,
                spec=spec,
                batch_size=batch_size,
                device=device,
                perturbed_train_array=perturbed_train_array,
            )
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda":
                raise
            torch.cuda.empty_cache()
            batch_size //= 2
    raise RuntimeError(
        f"CUDA OOM persisted below minimum batch size {spec.minimum_batch_size}"
    )
