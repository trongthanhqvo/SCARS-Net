from .h2_features import (
    H2_BANK_ORDER,
    assert_unique_h2_effective_configurations,
    effective_h2_signature,
    h2_configuration_factory,
    load_h2_representation,
)
from .external import frozen_external_sota_registry
from .features import (
    cwt_images,
    fixed_wavelet_subband_features,
    log_psd_sobel_images,
    log_stft_images,
    magnitude_phase,
    real_imag,
    stft_dct_features,
    SourceGlobalStandardizer,
)

__all__ = [
    "H2_BANK_ORDER",
    "h2_configuration_factory",
    "assert_unique_h2_effective_configurations",
    "effective_h2_signature",
    "load_h2_representation",
    "frozen_external_sota_registry",
    "cwt_images",
    "fixed_wavelet_subband_features",
    "log_psd_sobel_images",
    "log_stft_images",
    "magnitude_phase",
    "real_imag",
    "stft_dct_features",
    "SourceGlobalStandardizer",
    "baseline_definitions",
    "torch_model_factory",
]


def __getattr__(name):
    # Keep result validation and paper macro generation free of a PyTorch/OpenMP
    # import; model factories are loaded only by experiment commands.
    if name in {"BaselineDefinition", "baseline_definitions", "torch_model_factory"}:
        from .registry import BaselineDefinition, baseline_definitions, torch_model_factory

        return {
            "BaselineDefinition": BaselineDefinition,
            "baseline_definitions": baseline_definitions,
            "torch_model_factory": torch_model_factory,
        }[name]
    raise AttributeError(name)
