from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import os
import random
from time import perf_counter
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from scars.evaluation.classification import recording_level_metrics
from scars.models.scars_net import FamilyTeacher, SCARSNet
from scars.training.pcrd import ParetoRelation, pcrd_macro_loss, relation_macro_probabilities


# Must be set before the first CUDA GEMM when deterministic algorithms are
# enabled; required by cuBLAS on CUDA 10.2+.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@dataclass(frozen=True)
class TrainingSpec:
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    max_epochs: int = 100
    patience: int = 10
    batch_size: int = 32
    effective_batch_size: int = 64
    minimum_batch_size: int = 4
    num_workers: int = 0
    amp: bool = True
    gradient_clip_norm: float = 5.0


@dataclass(frozen=True)
class TrainingReport:
    seed: int
    device: str
    amp: bool
    batch_size: int
    best_epoch: int
    best_source_validation_macro_f1: float
    epochs_completed: int
    peak_gpu_memory_bytes: int | None
    elapsed_sec: float
    parameter_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def make_grad_scaler(enabled: bool):
    """Use the modern AMP API when available, otherwise the legacy CUDA API."""
    modern = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if modern is not None:
        return modern("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


class PairedTensorDataset(Dataset):
    def __init__(self, clean: np.ndarray, perturbed: np.ndarray, labels: np.ndarray):
        if clean.shape != perturbed.shape or len(clean) != len(labels):
            raise ValueError("clean, perturbed and labels must align")
        self.clean = clean
        self.perturbed = perturbed
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            torch.as_tensor(np.asarray(self.clean[index]), dtype=torch.float32),
            torch.as_tensor(np.asarray(self.perturbed[index]), dtype=torch.float32),
            torch.as_tensor(self.labels[index], dtype=torch.long),
            index,
        )


class CleanTensorDataset(Dataset):
    def __init__(self, tensor: np.ndarray, labels: np.ndarray):
        if len(tensor) != len(labels):
            raise ValueError("tensor and labels must align")
        self.tensor = tensor
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            torch.as_tensor(np.asarray(self.tensor[index]), dtype=torch.float32),
            torch.as_tensor(self.labels[index], dtype=torch.long),
        )


def _class_weights(labels: np.ndarray, class_count: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=class_count)
    if np.any(counts == 0):
        raise ValueError("Every class must occur in source_fit")
    weights = len(labels) / (class_count * counts.astype(np.float64))
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _local_relations(
    global_indices: Sequence[int], relation_lookup: dict[int, list[ParetoRelation]]
) -> list[ParetoRelation]:
    output: list[ParetoRelation] = []
    for local_index, global_index in enumerate(global_indices):
        output.extend(
            replace(relation, sample_index=local_index)
            for relation in relation_lookup.get(int(global_index), [])
        )
    return output


def _accumulation_steps(spec: TrainingSpec, batch_size: int) -> int:
    if spec.effective_batch_size < batch_size:
        raise ValueError("effective_batch_size cannot be smaller than the CUDA micro-batch")
    if spec.effective_batch_size % batch_size:
        raise ValueError("effective_batch_size must be divisible by every OOM back-off batch")
    return spec.effective_batch_size // batch_size


def predict_scars(
    model: SCARSNet,
    tensor: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(tensor), batch_size):
            batch = torch.as_tensor(
                np.asarray(tensor[start : start + batch_size]),
                dtype=torch.float32,
                device=device,
            )
            output = model(batch)
            probabilities.append(torch.softmax(output["logits"], dim=1).cpu().numpy())
            weights.append(output["weights"].cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(weights)


def _fit_scars_once(
    model_factory: Callable[[], SCARSNet],
    clean_train: np.ndarray,
    perturbed_train: np.ndarray,
    train_labels: np.ndarray,
    relations: Sequence[ParetoRelation],
    validation_tensor: np.ndarray,
    validation_labels: np.ndarray,
    validation_recording_ids: np.ndarray,
    *,
    seed: int,
    lambda_pcrd: float,
    spec: TrainingSpec,
    batch_size: int,
    device: torch.device,
) -> tuple[SCARSNet, TrainingReport]:
    seed_everything(seed)
    started = perf_counter()
    model = model_factory().to(device)
    class_count = model.classifier.out_features
    weights = _class_weights(train_labels, class_count, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay
    )
    scaler = make_grad_scaler(spec.amp and device.type == "cuda")
    accumulation_steps = _accumulation_steps(spec, batch_size)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        PairedTensorDataset(clean_train, perturbed_train, train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=spec.num_workers,
        pin_memory=device.type == "cuda",
    )
    if lambda_pcrd > 0 and not relations:
        raise RuntimeError("PCRD training requires a nonempty frozen relation cache")
    relation_probability = relation_macro_probabilities(relations) if relations else None
    relation_rng = np.random.default_rng(seed + 91_003)
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
        for batch_index, (clean, perturbed, labels, global_indices) in enumerate(loader):
            clean = clean.to(device, non_blocking=True)
            perturbed = perturbed.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=spec.amp and device.type == "cuda",
            ):
                clean_output = model(clean)
                perturbed_output = model(perturbed)
                loss = F.cross_entropy(clean_output["logits"], labels, weight=weights)
                loss = loss + F.cross_entropy(perturbed_output["logits"], labels, weight=weights)
                if lambda_pcrd > 0:
                    sampled = relation_rng.choice(
                        len(relations), size=len(labels), replace=True, p=relation_probability
                    )
                    sampled_relations = [relations[int(index)] for index in sampled]
                    relation_tensor = torch.as_tensor(
                        np.asarray(
                            perturbed_train[
                                [relation.sample_index for relation in sampled_relations]
                            ]
                        ),
                        dtype=torch.float32,
                        device=device,
                    )
                    relation_logits = model(relation_tensor)["gate_logits"]
                    local_relations = [
                        replace(relation, sample_index=index)
                        for index, relation in enumerate(sampled_relations)
                    ]
                    # Sampling probabilities are the exact nested macro weights,
                    # so this simple mean is an unbiased minibatch estimator.
                    relation_loss = torch.stack(
                        [
                            F.relu(
                                relation_logits[index, relation.loser]
                                - relation_logits[index, relation.winner]
                                + 0.2
                            )
                            for index, relation in enumerate(local_relations)
                        ]
                    ).mean()
                    loss = loss + lambda_pcrd * relation_loss
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
        probability, _ = predict_scars(
            model, validation_tensor, device=device, batch_size=batch_size
        )
        validation = recording_level_metrics(
            validation_labels,
            validation_recording_ids,
            probability,
            np.arange(class_count),
        )
        metric = float(validation["macro_f1"])
        if metric > best_metric + 1.0e-12:
            best_metric = metric
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= spec.patience:
                break
    if best_state is None:
        raise RuntimeError("Training produced no source-validation checkpoint")
    model.load_state_dict(best_state)
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    report = TrainingReport(
        seed=seed,
        device=str(device),
        amp=bool(spec.amp and device.type == "cuda"),
        batch_size=batch_size,
        best_epoch=best_epoch,
        best_source_validation_macro_f1=best_metric,
        epochs_completed=epochs_completed,
        peak_gpu_memory_bytes=peak,
        elapsed_sec=float(perf_counter() - started),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )
    return model, report


def fit_scars_with_oom_backoff(
    model_factory: Callable[[], SCARSNet],
    clean_train: np.ndarray,
    perturbed_train: np.ndarray,
    train_labels: np.ndarray,
    relations: Sequence[ParetoRelation],
    validation_tensor: np.ndarray,
    validation_labels: np.ndarray,
    validation_recording_ids: np.ndarray,
    *,
    seed: int,
    lambda_pcrd: float,
    spec: TrainingSpec = TrainingSpec(),
    device_name: str = "auto",
) -> tuple[SCARSNet, TrainingReport]:
    """Train sequentially and halve batch size on a real CUDA OOM."""
    device = resolve_device(device_name)
    batch_size = spec.batch_size
    while batch_size >= spec.minimum_batch_size:
        try:
            return _fit_scars_once(
                model_factory,
                clean_train,
                perturbed_train,
                train_labels,
                relations,
                validation_tensor,
                validation_labels,
                validation_recording_ids,
                seed=seed,
                lambda_pcrd=lambda_pcrd,
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


def train_family_teacher(
    family_tensor: np.ndarray,
    labels: np.ndarray,
    validation_tensor: np.ndarray,
    validation_labels: np.ndarray,
    validation_recording_ids: np.ndarray,
    *,
    class_count: int,
    seed: int,
    spec: TrainingSpec = TrainingSpec(),
    device_name: str = "auto",
) -> tuple[FamilyTeacher, TrainingReport]:
    """Train on source_fit; caller must pass source_selection as validation arrays."""
    from scars.training.baselines import fit_classifier_with_oom_backoff

    artifact = fit_classifier_with_oom_backoff(
        lambda: FamilyTeacher(class_count),
        family_tensor,
        labels,
        validation_tensor,
        validation_labels,
        validation_recording_ids,
        seed=seed,
        spec=spec,
        device_name=device_name,
    )
    if not isinstance(artifact.model, FamilyTeacher):
        raise AssertionError("Teacher factory returned the wrong model type")
    return artifact.model, artifact.report
