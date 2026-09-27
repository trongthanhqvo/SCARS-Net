from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from scars.experiment.common import atomic_json
from scars.models.scars_net import FAMILY_ORDER, SCARSNet
from scars.training.engine import TrainingSpec, fit_scars_with_oom_backoff, resolve_device
from scars.training.pcrd import ParetoRelation, relation_macro_probabilities


def _fixture(seed: int):
    rng = np.random.default_rng(seed)
    count = 24
    labels = np.repeat(np.arange(2), count // 2)
    clean = rng.normal(0.0, 0.2, (count, 4, 16, 16)).astype(np.float32)
    clean[labels == 1, 0, 3:8, 3:8] += 1.0
    clean[labels == 0, 1, 8:13, 8:13] += 1.0
    perturbed = clean + rng.normal(0.0, 0.08, clean.shape).astype(np.float32)
    recording_ids = np.asarray([f"smoke-recording-{index:02d}" for index in range(count)])
    relations = [
        ParetoRelation(
            sample_index=index,
            recording_id=str(recording_ids[index]),
            label=str(labels[index]),
            nuisance="awgn",
            severity=0.0,
            winner=0 if labels[index] else 1,
            loser=1 if labels[index] else 0,
            winner_family="W" if labels[index] else "C",
            loser_family="C" if labels[index] else "W",
        )
        for index in range(count)
    ]
    return clean, perturbed, labels, recording_ids, relations


def run(output_dir: Path, seed: int, device: str) -> Path:
    started = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    clean, perturbed, labels, recording_ids, relations = _fixture(seed)
    probability = relation_macro_probabilities(relations)
    model, report = fit_scars_with_oom_backoff(
        lambda: SCARSNet(FAMILY_ORDER, 2),
        clean,
        perturbed,
        labels,
        relations,
        clean,
        labels,
        recording_ids,
        seed=seed,
        lambda_pcrd=0.1,
        spec=TrainingSpec(
            max_epochs=2,
            patience=2,
            batch_size=8,
            effective_batch_size=8,
            minimum_batch_size=4,
            amp=True,
        ),
        device_name=device,
    )
    resolved = resolve_device(device)
    model.to(resolved).eval()
    with torch.inference_mode():
        batch = torch.as_tensor(clean[:2], device=resolved)
        output = model(batch)
        reconstructed = torch.sum(output["weights"].unsqueeze(-1) * output["tokens"], dim=1)
    stem_parameters = [
        {id(parameter) for parameter in model.stems[family].parameters()}
        for family in FAMILY_ORDER
    ]
    nonshared = all(
        not stem_parameters[left] & stem_parameters[right]
        for left in range(len(stem_parameters))
        for right in range(left + 1, len(stem_parameters))
    )
    checks = {
        "input_contract_float32_n4hw": clean.dtype == np.float32 and clean.shape[1:] == (4, 16, 16),
        "family_order": list(model.active_families) == list(FAMILY_ORDER),
        "nonshared_stems": nonshared,
        "simplex_weights": bool(torch.allclose(output["weights"].sum(1), torch.ones(2, device=resolved), atol=1.0e-6)),
        "strict_weighted_sum": bool(torch.allclose(output["fused"], reconstructed, atol=1.0e-6)),
        "single_fused_classifier": model.classifier.in_features == model.token_dim,
        "pcrd_macro_probability": bool(np.isclose(probability.sum(), 1.0, atol=1.0e-12)),
    }
    payload = {
        "schema_version": "scars-contract-smoke-1.0",
        "status": "passed" if all(checks.values()) else "failed",
        "synthetic_dev": True,
        "paper_evidence": False,
        "seed": seed,
        "device": str(resolved),
        "checks": checks,
        "training_report": report.to_dict(),
        "hypotheses": {key: "open" for key in ("H1", "H2", "H3", "H4")},
        "elapsed_sec": float(perf_counter() - started),
        "warning": "Synthetic engineering contract check; never populate manuscript results.",
    }
    if payload["status"] != "passed":
        raise RuntimeError(f"SCARS-Net/PCRD contract smoke failed: {checks}")
    path = output_dir / "contract_smoke.json"
    atomic_json(path, payload)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic SCARS-Net/PCRD contract smoke")
    parser.add_argument("--output-dir", type=Path, default=Path("results/contract-smoke"))
    parser.add_argument("--seed", type=int, default=24021)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> int:
    path = run(**vars(build_parser().parse_args()))
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
