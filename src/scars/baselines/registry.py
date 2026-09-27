from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from torch import nn

from scars.models.baselines import (
    EarlyFusionResNet,
    MagnitudePhaseCNN,
    PatchTransformer,
    RawIQ1DCNN,
    UniformLateFusion,
)
from scars.models.scars_net import SCARSNet
from scars.results.registry import RECOGNITION_CONDITIONS


@dataclass(frozen=True)
class BaselineDefinition:
    condition_id: str
    input_kind: str
    trainer: str
    perturbation_pairing: bool
    source_checkpoint_role: str
    purpose: str


_DEFINITIONS = (
    BaselineDefinition("raw_iq_cnn", "real_imag_waveform", "torch", True, "source_validation", "end_to_end_waveform"),
    BaselineDefinition("patch_transformer", "log_stft_image", "torch", True, "source_validation", "token_mixing"),
    BaselineDefinition("magnitude_phase_cnn", "magnitude_phase_waveform", "torch", True, "source_validation", "iq_parameterization"),
    BaselineDefinition("log_stft_cnn", "log_stft_image", "torch", True, "source_validation", "time_frequency"),
    BaselineDefinition("cwt_cnn", "cwt_image", "torch", True, "source_validation", "multiscale_image"),
    BaselineDefinition("wst_only", "W", "torch", True, "source_validation", "family_contribution"),
    BaselineDefinition("cyclic_only", "C", "torch", True, "source_validation", "family_contribution"),
    BaselineDefinition("stft_dct_xgboost", "stft_dct_64", "xgboost", False, "source_selection", "engineered"),
    BaselineDefinition("fixed_wavelet_subbands", "db4_packet_energy_8", "ridge", False, "source_selection", "nonlearned_wavelet"),
    BaselineDefinition("log_psd_sobel", "log_psd_sobel_3ch", "torch", True, "source_validation", "structure_enhanced"),
    BaselineDefinition("early_fusion_resnet", "active_tensor", "torch_capacity_matched", True, "source_validation", "early_fusion_control"),
    BaselineDefinition("uniform_late_fusion", "active_tensor", "torch", True, "source_validation", "no_learned_routing"),
    BaselineDefinition("ordinary_gate", "active_tensor", "scars_net_lambda_0", True, "source_validation", "architecture_control"),
    BaselineDefinition("shuffled_pcrd", "active_tensor", "scars_net_shuffled_relations", True, "source_validation", "relation_negative_control"),
    BaselineDefinition("pcrd", "active_tensor", "scars_net_pcrd", True, "source_validation", "proposed_method"),
)


def baseline_definitions() -> tuple[BaselineDefinition, ...]:
    observed = [definition.condition_id for definition in _DEFINITIONS]
    if observed != RECOGNITION_CONDITIONS:
        raise AssertionError(f"Baseline registry drift: {observed}")
    return _DEFINITIONS


def torch_model_factory(
    condition_id: str,
    *,
    class_count: int,
    active_families: Sequence[str] = ("W", "C", "E", "S"),
    early_fusion_width: int = 32,
) -> Callable[[], nn.Module]:
    channels = len(active_families)
    factories: dict[str, Callable[[], nn.Module]] = {
        "raw_iq_cnn": lambda: RawIQ1DCNN(class_count),
        "patch_transformer": lambda: PatchTransformer(class_count, input_channels=1, image_size=16, patch_size=4),
        "magnitude_phase_cnn": lambda: MagnitudePhaseCNN(class_count),
        "log_stft_cnn": lambda: EarlyFusionResNet(1, class_count),
        "cwt_cnn": lambda: EarlyFusionResNet(1, class_count),
        "wst_only": lambda: EarlyFusionResNet(1, class_count),
        "cyclic_only": lambda: EarlyFusionResNet(1, class_count),
        "log_psd_sobel": lambda: EarlyFusionResNet(3, class_count),
        "early_fusion_resnet": lambda: EarlyFusionResNet(channels, class_count, early_fusion_width),
        "uniform_late_fusion": lambda: UniformLateFusion(active_families, class_count),
        "ordinary_gate": lambda: SCARSNet(active_families, class_count),
        "shuffled_pcrd": lambda: SCARSNet(active_families, class_count),
        "pcrd": lambda: SCARSNet(active_families, class_count),
    }
    if condition_id not in factories:
        raise KeyError(f"{condition_id} is not a torch baseline")
    return factories[condition_id]
