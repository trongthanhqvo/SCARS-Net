from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
from typing import Callable

import yaml
import numpy as np

from scars.baselines.registry import baseline_definitions, torch_model_factory
from scars.evaluation.costs import estimate_model_macs, model_parameter_count, tensor_bytes
from scars.models.baselines import EarlyFusionResNet, UniformLateFusion
from scars.models.scars_net import FAMILY_ORDER, FamilyTeacher, SCARSNet
from scars.representations.scattering import scattering_path_count
from scars.results.registry import ABLATION_CONDITIONS, RECOGNITION_CONDITIONS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _affine_profile(
    factory: Callable[[int], object],
    example_factory: Callable[[], object],
) -> dict[str, object]:
    values: dict[int, tuple[int, int]] = {}
    for class_count in (2, 3, 5):
        model = factory(class_count)
        values[class_count] = (
            model_parameter_count(model),
            estimate_model_macs(model, example_factory()),
        )
    parameter_slope = values[3][0] - values[2][0]
    parameter_intercept = values[2][0] - 2 * parameter_slope
    mac_slope = values[3][1] - values[2][1]
    mac_intercept = values[2][1] - 2 * mac_slope
    if values[5] != (
        parameter_intercept + 5 * parameter_slope,
        mac_intercept + 5 * mac_slope,
    ):
        raise AssertionError("Model parameter/MAC count is not affine in class_count")
    return {
        "class_symbol": "K",
        "parameters": None,
        "parameter_formula": {
            "intercept": parameter_intercept,
            "per_class": parameter_slope,
            "display": f"{parameter_intercept} + {parameter_slope} K",
        },
        "macs": None,
        "mac_formula": {
            "intercept": mac_intercept,
            "per_class": mac_slope,
            "display": f"{mac_intercept} + {mac_slope} K",
        },
        "exact_when": "Substitute the frozen shared ontology class count K.",
    }


def _torch_profiles(window_samples: int, resolution: int) -> dict[str, object]:
    import torch

    definitions = {item.condition_id: item for item in baseline_definitions()}
    profiles: dict[str, object] = {}
    input_shapes = {
        "raw_iq_cnn": [1, 2, window_samples],
        "magnitude_phase_cnn": [1, 2, window_samples],
        "patch_transformer": [1, 1, resolution, resolution],
        "log_stft_cnn": [1, 1, resolution, resolution],
        "cwt_cnn": [1, 1, resolution, resolution],
        "wst_only": [1, 1, resolution, resolution],
        "cyclic_only": [1, 1, resolution, resolution],
        "log_psd_sobel": [1, 3, resolution, resolution],
        "early_fusion_resnet": [1, 4, resolution, resolution],
        "uniform_late_fusion": [1, 4, resolution, resolution],
        "ordinary_gate": [1, 4, resolution, resolution],
        "shuffled_pcrd": [1, 4, resolution, resolution],
        "pcrd": [1, 4, resolution, resolution],
    }
    for condition_id in RECOGNITION_CONDITIONS:
        definition = definitions[condition_id]
        if condition_id not in input_shapes:
            profiles[condition_id] = {
                "input_kind": definition.input_kind,
                "trainer": definition.trainer,
                "input_shape": None,
                "parameters": None,
                "macs": None,
                "reason_null": "Fitted ridge/XGBoost size depends on frozen feature and ontology artifacts.",
            }
            continue
        shape = input_shapes[condition_id]

        def factory(class_count: int, condition: str = condition_id):
            return torch_model_factory(
                condition,
                class_count=class_count,
                active_families=FAMILY_ORDER,
                early_fusion_width=32,
            )()

        profile = _affine_profile(factory, lambda shape=shape: torch.zeros(*shape))
        profile.update(
            {
                "input_kind": definition.input_kind,
                "trainer": definition.trainer,
                "input_shape": shape,
                "active_families_assumed": list(FAMILY_ORDER)
                if definition.input_kind == "active_tensor"
                else None,
                "scope_note": (
                    "Topology profile at base_width=32; the reported early-fusion row remains null "
                    "until the capacity-matched width is frozen."
                    if condition_id == "early_fusion_resnet"
                    else "Exact architecture formula before training; K remains data-contract dependent."
                ),
            }
        )
        profiles[condition_id] = profile

    active_family_profiles: dict[str, object] = {}
    for active_count in range(1, len(FAMILY_ORDER) + 1):
        families = FAMILY_ORDER[:active_count]
        image = lambda active_count=active_count: torch.zeros(1, active_count, resolution, resolution)
        active_family_profiles[str(active_count)] = {
            "families": list(families),
            "scars_net": _affine_profile(
                lambda class_count, families=families: SCARSNet(families, class_count), image
            ),
            "uniform_late_fusion": _affine_profile(
                lambda class_count, families=families: UniformLateFusion(families, class_count), image
            ),
        }
    teacher_profile = _affine_profile(
        lambda class_count: FamilyTeacher(class_count),
        lambda: torch.zeros(1, 1, resolution, resolution),
    )
    early_fusion_width32 = _affine_profile(
        lambda class_count: EarlyFusionResNet(4, class_count, 32),
        lambda: torch.zeros(1, 4, resolution, resolution),
    )
    return {
        "mac_definition": "Conv/Linear/MultiheadAttention MACs; excludes normalization, activation, pooling, softmax, indexing, and I/O.",
        "mac_estimator_version": "scars-model-macs-v2-conv-linear-attention",
        "conditions": profiles,
        "active_family_profiles": active_family_profiles,
        "family_teacher": teacher_profile,
        "early_fusion_width32_reference": early_fusion_width32,
    }


def build_static_metadata(project_root: Path) -> dict[str, object]:
    root = project_root.resolve()
    config_dir = root / "configs"
    representation_path = config_dir / "representation.yaml"
    experiments_path = config_dir / "experiments.yaml"
    baselines_path = config_dir / "baselines.yaml"
    nuisance_path = config_dir / "nuisance_grid.yaml"
    representation = yaml.safe_load(representation_path.read_text())
    experiments = yaml.safe_load(experiments_path.read_text())
    baselines = yaml.safe_load(baselines_path.read_text())
    nuisance = yaml.safe_load(nuisance_path.read_text())

    window_samples = int(representation["input"]["window_samples"])
    window_hop = int(representation["input"]["hop_samples"])
    frame = int(representation["cyclostationary"]["frame_samples"])
    frame_hop = int(representation["cyclostationary"]["hop_samples"])
    frames_per_window = 1 + (window_samples - frame) // frame_hop
    resolutions = [int(value) for value in representation["common"]["output_resolutions"]]
    primary_resolution = int(representation["common"]["primary_resolution"])
    family_order = list(representation["tensor"]["family_order"])
    dtype_bytes = 4

    tensor_profiles = []
    for resolution in resolutions:
        tensor_profiles.append(
            {
                "resolution": resolution,
                "full_family_shape": [len(family_order), resolution, resolution],
                "bytes_per_family": tensor_bytes((1, resolution, resolution), dtype_bytes),
                "full_tensor_bytes": tensor_bytes((len(family_order), resolution, resolution), dtype_bytes),
                "active_tensor_bytes_formula": f"{dtype_bytes * resolution * resolution} A",
                "active_family_symbol": "A",
            }
        )

    wst_profiles = []
    for j in representation["wavelet_scattering"]["sweep"]["J"]:
        for q in representation["wavelet_scattering"]["sweep"]["Q"]:
            first, second = scattering_path_count(int(j), int(q))
            wst_profiles.append(
                {"J": int(j), "Q": int(q), "order1_paths": first, "order2_paths": second, "total_paths": first + second}
            )

    nuisances = nuisance["nuisances"]
    nuisance_cases = sum(len(item["severities"]) for item in nuisances)
    robustness = experiments["robustness"]
    required_seeds = list(baselines["tier_b"]["stochastic_seeds"])
    learned_condition_count = sum(
        1 for item in baseline_definitions() if item.trainer.startswith("torch") or item.trainer.startswith("scars_net")
    )
    source_paths = [
        root / "src/scars/cli/collect_static_metadata.py",
        root / "src/scars/evaluation/costs.py",
        root / "src/scars/models/baselines.py",
        root / "src/scars/models/scars_net.py",
        root / "src/scars/representations/scattering.py",
        root / "src/scars/representations/tensor.py",
        root / "src/scars/baselines/registry.py",
        root / "src/scars/results/registry.py",
        root / "src/scars/results/schema.py",
        root / "src/scars/results/validator.py",
        root / "src/scars/cli/finalize_results.py",
    ]
    import torch

    return {
        "status": "computed_without_real_iq_or_training",
        "empirical_evidence": False,
        "generated_from": {
            "project": "scars-uav-rf",
            "config_sha256": {
                path.name: _sha256(path)
                for path in (representation_path, experiments_path, baselines_path, nuisance_path)
            },
            "source_sha256": {
                path.relative_to(root).as_posix(): _sha256(path) for path in source_paths
            },
            "collector_environment": {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "pyyaml": yaml.__version__,
                "platform": sys.platform,
            },
        },
        "registry_counts": {
            "candidate_datasets": 3,
            "recognition_conditions": len(RECOGNITION_CONDITIONS),
            "neural_conditions": learned_condition_count,
            "ablation_conditions": len(ABLATION_CONDITIONS),
            "required_seeds": len(required_seeds),
            "candidate_held_domain_folds": 3,
            "nominal_neural_fold_seed_runs": learned_condition_count * 3 * len(required_seeds),
            "nominal_recognition_fold_seed_cells": 3 * ((len(RECOGNITION_CONDITIONS) - 1) * len(required_seeds) + 1),
            "ablation_fold_seed_cells": None,
            "ablation_fold_seed_formula": "sum_d (26 + 5 A_d) = 78 + 5 sum_d A_d for three eligible folds",
            "scope_note": (
                "Nominal three-fold recognition count follows the runtime seed contract: "
                "five seeds for 14 conditions and one deterministic fixed-wavelet row. "
                "The ablation count remains symbolic because A_d is selected independently per fold; 26 "
                "representation ablations run once per fold and each active-family deletion "
                "runs five seeds. Actual eligible folds and A_d remain null until preflight/source freeze."
            ),
        },
        "protocol": {
            "candidate_datasets": ["DroneRFa", "DroneRFb-DIR", "DRFF-R2"],
            "required_seeds": required_seeds,
            "window_samples": window_samples,
            "window_hop_samples": window_hop,
            "window_overlap_samples": window_samples - window_hop,
            "window_overlap_fraction": (window_samples - window_hop) / window_samples,
            "input_complex_dtype": representation["input"]["dtype"],
            "input_window_bytes": window_samples * 8,
            "group_split_before_windowing": bool(representation["input"]["group_split_before_windowing"]),
            "registered_resamples": 10000,
            "latency_warmups": 20,
            "latency_repeats": 100,
            "source_only_tuning": bool(baselines["common"]["source_only_tuning"]),
            "target_tuning": baselines["common"]["target_tuning"],
        },
        "representation": {
            "stored_dtype": "float32",
            "bytes_per_value": dtype_bytes,
            "family_order": family_order,
            "candidate_resolutions": resolutions,
            "primary_resolution": primary_resolution,
            "tensor_profiles": tensor_profiles,
            "nominal_full_tensor_shape": [len(family_order), primary_resolution, primary_resolution],
            "nominal_full_tensor_bytes": tensor_bytes((len(family_order), primary_resolution, primary_resolution), dtype_bytes),
            "frame_samples": frame,
            "frame_hop_samples": frame_hop,
            "frames_per_window": frames_per_window,
            "native_stft_shape": [frame, frames_per_window],
            "native_energy_shape": [frames_per_window, 2],
            "nominal_native_cyclic_shape": [int(representation["cyclostationary"]["nominal_cyclic_count"]), frame],
            "nominal_cyclic_count": int(representation["cyclostationary"]["nominal_cyclic_count"]),
            "cyclic_count_sweep": list(representation["cyclostationary"]["cyclic_count_sweep"]),
            "cyclic_frequency_values": None,
            "cyclic_frequency_reason_null": "Source-fitted from eligible source recordings.",
            "wst_nominal": dict(representation["wavelet_scattering"]["nominal"]),
            "wst_path_profiles": wst_profiles,
            "normalization_quantiles": [
                float(representation["normalization"]["lower_quantile"]),
                float(representation["normalization"]["upper_quantile"]),
            ],
            "normalization_values": None,
            "energy_reference": None,
            "active_families": None,
            "source_selected_configuration": None,
        },
        "nuisance_grid": {
            "family_count": len(nuisances),
            "severity_case_count": nuisance_cases,
            "families": [
                {"id": item["id"], "parameter": item["parameter"], "severities": item["severities"], "units": item["units"]}
                for item in nuisances
            ],
        },
        "robustness_grid": {
            "snr_db": list(robustness["snr_db"]),
            "sir_db": list(robustness["sir_db"]),
            "metrics": list(robustness["metrics"]),
        },
        "models": _torch_profiles(window_samples, primary_resolution),
        "real_run_only": {
            "dataset_eligibility": None,
            "shared_ontology_and_class_count_K": None,
            "recording_group_and_class_counts": None,
            "source_fitted_normalizers": None,
            "source_frozen_cyclic_frequency_values": None,
            "source_selected_active_family_count_A": None,
            "capacity_matched_early_fusion_width": None,
            "trained_checkpoint_parameters_if_selection_changes_topology": None,
            "representation_and_model_latency": None,
            "peak_memory_and_energy": None,
            "recognition_detection_robustness_and_calibration_metrics": None,
            "confidence_intervals_effect_sizes_and_p_values": None,
            "H1_H4_and_M1_decisions": None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect deterministic SCARS metadata without reading IQ data")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_static_metadata(args.project_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
