from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ParetoRelation:
    sample_index: int
    recording_id: str
    label: str
    nuisance: str
    severity: float
    winner: int
    loser: int
    winner_family: str
    loser_family: str


def _recording_weights(recording_ids: Sequence[str], device: torch.device) -> torch.Tensor:
    ids = np.asarray(recording_ids, dtype=object)
    weights = np.zeros(len(ids), dtype=np.float32)
    unique = sorted(set(map(str, ids.tolist())))
    for recording_id in unique:
        mask = np.asarray([str(value) == recording_id for value in ids])
        weights[mask] = 1.0 / (len(unique) * int(mask.sum()))
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def fit_recording_balanced_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    recording_ids: Sequence[str],
    *,
    lower: float = 0.05,
    upper: float = 20.0,
    max_iter: int = 80,
    allow_boundary: bool = False,
) -> float:
    """Fit one positive source-calibration temperature with equal recording weight."""
    if logits.ndim != 2 or labels.ndim != 1 or len(logits) != len(labels):
        raise ValueError("Temperature inputs must be aligned [N,K] logits and [N] labels")
    if len(recording_ids) != len(labels):
        raise ValueError("recording_ids must align with calibration examples")
    detached = logits.detach()
    target = labels.detach().long().to(detached.device)
    weights = _recording_weights(recording_ids, detached.device)
    log_temperature = torch.nn.Parameter(torch.zeros((), device=detached.device))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.25, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temperature).clamp(lower, upper)
        loss = torch.sum(F.cross_entropy(detached / temperature, target, reduction="none") * weights)
        loss.backward()
        return loss

    optimizer.step(closure)
    value = float(torch.exp(log_temperature.detach()).clamp(lower, upper).cpu())
    if not np.isfinite(value):
        raise RuntimeError("Temperature calibration produced a nonfinite value")
    if not allow_boundary and (value <= lower + 1.0e-6 or value >= upper - 1.0e-6):
        raise RuntimeError("Temperature calibration hit a registered bound")
    return value


def sensitivity_scale(
    clean_log_probability: np.ndarray,
    perturbed_log_probability: np.ndarray,
    clean_family_tensor: np.ndarray,
    perturbed_family_tensor: np.ndarray,
    recording_ids: Sequence[str],
    *,
    epsilon: float = 1.0e-8,
) -> float:
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Sensitivity epsilon must be finite and positive")
    clean_log_probability = np.asarray(clean_log_probability, dtype=float)
    perturbed_log_probability = np.asarray(perturbed_log_probability, dtype=float)
    clean_family_tensor = np.asarray(clean_family_tensor, dtype=float)
    perturbed_family_tensor = np.asarray(perturbed_family_tensor, dtype=float)
    if clean_log_probability.shape != perturbed_log_probability.shape:
        raise ValueError("Teacher log-probability arrays must have identical shapes")
    if clean_family_tensor.shape != perturbed_family_tensor.shape:
        raise ValueError("Clean and perturbed family tensors must have identical shapes")
    if len(clean_log_probability) != len(clean_family_tensor) or len(recording_ids) != len(clean_family_tensor):
        raise ValueError("Sensitivity inputs must align per example")
    if len(clean_family_tensor) == 0 or len(recording_ids) == 0:
        raise ValueError("Sensitivity scale needs at least one physical recording")
    if not all(
        np.all(np.isfinite(values))
        for values in (
            clean_log_probability,
            perturbed_log_probability,
            clean_family_tensor,
            perturbed_family_tensor,
        )
    ):
        raise RuntimeError("Sensitivity inputs contain nonfinite values")
    numerator = np.linalg.norm(perturbed_log_probability - clean_log_probability, axis=1)
    denominator = np.mean(np.abs(perturbed_family_tensor - clean_family_tensor), axis=tuple(range(1, clean_family_tensor.ndim)))
    ratio = numerator / (denominator + epsilon)
    if not np.all(np.isfinite(ratio)):
        raise RuntimeError("Sensitivity ratio is nonfinite")
    ids = np.asarray(recording_ids, dtype=object)
    recording_medians = np.asarray(
        [np.median(ratio[ids == recording_id]) for recording_id in sorted(set(ids.tolist()), key=str)]
    )
    if len(recording_medians) == 0 or not np.all(np.isfinite(recording_medians)):
        raise RuntimeError("Sensitivity recording support is empty or nonfinite")
    value = float(np.quantile(recording_medians, 0.95, method="higher"))
    if not np.isfinite(value):
        raise RuntimeError("Sensitivity scale kappa is undefined")
    return value


def build_pareto_relations(
    risk: np.ndarray,
    margin: np.ndarray,
    active_families: Sequence[str],
    recording_ids: Sequence[str],
    labels: Sequence[str],
    nuisances: Sequence[str],
    severities: Sequence[float],
    *,
    tolerance: float = 1.0e-6,
) -> list[ParetoRelation]:
    risk = np.asarray(risk, dtype=float)
    margin = np.asarray(margin, dtype=float)
    if risk.shape != margin.shape or risk.ndim != 2 or risk.shape[1] != len(active_families):
        raise ValueError("risk/margin must be [examples, active_families]")
    metadata = (recording_ids, labels, nuisances, severities)
    if any(len(values) != len(risk) for values in metadata):
        raise ValueError("Relation metadata must align per example")
    if not np.all(np.isfinite(risk)) or not np.all(np.isfinite(margin)):
        raise RuntimeError("PCRD risk/margin evidence contains nonfinite values")
    output: list[ParetoRelation] = []
    for sample in range(len(risk)):
        for first in range(len(active_families)):
            for second in range(first + 1, len(active_families)):
                first_dominates = (
                    risk[sample, first] <= risk[sample, second] + tolerance
                    and margin[sample, first] >= margin[sample, second] - tolerance
                    and (
                        risk[sample, first] < risk[sample, second] - tolerance
                        or margin[sample, first] > margin[sample, second] + tolerance
                    )
                )
                second_dominates = (
                    risk[sample, second] <= risk[sample, first] + tolerance
                    and margin[sample, second] >= margin[sample, first] - tolerance
                    and (
                        risk[sample, second] < risk[sample, first] - tolerance
                        or margin[sample, second] > margin[sample, first] + tolerance
                    )
                )
                if first_dominates == second_dominates:
                    continue
                winner, loser = (first, second) if first_dominates else (second, first)
                output.append(
                    ParetoRelation(
                        sample_index=sample,
                        recording_id=str(recording_ids[sample]),
                        label=str(labels[sample]),
                        nuisance=str(nuisances[sample]),
                        severity=float(severities[sample]),
                        winner=winner,
                        loser=loser,
                        winner_family=str(active_families[winner]),
                        loser_family=str(active_families[loser]),
                    )
                )
    return output


def pcrd_macro_loss(
    gate_logits: torch.Tensor,
    relations: Sequence[ParetoRelation],
    *,
    gamma: float = 0.2,
) -> torch.Tensor:
    """Equal recording -> nuisance -> severity -> comparable-pair PCRD mean."""
    if gate_logits.ndim != 2:
        raise ValueError("gate_logits must be [examples, active_families]")
    if not relations:
        raise RuntimeError("PCRD relation cache is empty")
    by_recording: dict[str, dict[str, dict[float, list[torch.Tensor]]]] = {}
    for relation in relations:
        if relation.sample_index >= len(gate_logits):
            raise IndexError("Relation sample index is outside gate-logit batch")
        value = F.relu(
            gate_logits[relation.sample_index, relation.loser]
            - gate_logits[relation.sample_index, relation.winner]
            + gamma
        )
        by_recording.setdefault(relation.recording_id, {}).setdefault(
            relation.nuisance, {}
        ).setdefault(relation.severity, []).append(value)
    recording_losses = []
    for nuisances in by_recording.values():
        nuisance_losses = []
        for severities_for_nuisance in nuisances.values():
            severity_losses = [torch.stack(values).mean() for values in severities_for_nuisance.values()]
            nuisance_losses.append(torch.stack(severity_losses).mean())
        recording_losses.append(torch.stack(nuisance_losses).mean())
    return torch.stack(recording_losses).mean()


def relation_macro_probabilities(relations: Sequence[ParetoRelation]) -> np.ndarray:
    """Probability of each relation under equal recording/nuisance/severity/pair mass."""
    if not relations:
        raise RuntimeError("PCRD relation cache is empty")
    cells: dict[str, dict[str, dict[float, list[int]]]] = {}
    for index, relation in enumerate(relations):
        cells.setdefault(relation.recording_id, {}).setdefault(
            relation.nuisance, {}
        ).setdefault(relation.severity, []).append(index)
    probabilities = np.zeros(len(relations), dtype=np.float64)
    recording_count = len(cells)
    for nuisances in cells.values():
        nuisance_count = len(nuisances)
        for severities in nuisances.values():
            severity_count = len(severities)
            for indices in severities.values():
                mass = 1.0 / (
                    recording_count * nuisance_count * severity_count * len(indices)
                )
                probabilities[indices] = mass
    if not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1.0e-12):
        raise AssertionError("PCRD macro probabilities do not sum to one")
    return probabilities


def shuffle_relations_within_strata(
    relations: Sequence[ParetoRelation], seed: int
) -> list[ParetoRelation]:
    """Stratified negative control preserving class/nuisance/severity counts."""
    rng = np.random.default_rng(seed)
    output = list(relations)
    strata: dict[tuple[str, str, float], list[int]] = {}
    for index, relation in enumerate(relations):
        strata.setdefault((relation.label, relation.nuisance, relation.severity), []).append(index)
    for indices in strata.values():
        if len(indices) < 2:
            continue
        pairs = [(relations[index].winner, relations[index].loser, relations[index].winner_family, relations[index].loser_family) for index in indices]
        order = rng.permutation(len(indices))
        if np.array_equal(order, np.arange(len(indices))):
            order = np.roll(order, 1)
        for destination, source in zip(indices, order):
            winner, loser, winner_family, loser_family = pairs[int(source)]
            output[destination] = replace(
                relations[destination],
                winner=winner,
                loser=loser,
                winner_family=winner_family,
                loser_family=loser_family,
            )
    before = [(item.winner, item.loser) for item in relations]
    after = [(item.winner, item.loser) for item in output]
    if after == before:
        raise RuntimeError(
            "Stratified PCRD permutation control is degenerate; M1 must be untestable"
        )
    return output


def write_relation_cache(path: Path, relations: Iterable[ParetoRelation]) -> dict[str, object]:
    records = [asdict(relation) for relation in relations]
    payload = {"schema_version": "1.0.0", "relations": records}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return {"path": path.name, "sha256": hashlib.sha256(encoded).hexdigest(), "count": len(records)}
