import numpy as np
import pytest
import torch

from scars.baselines import (
    SourceGlobalStandardizer,
    baseline_definitions,
    cwt_images,
    fixed_wavelet_subband_features,
    log_psd_sobel_images,
    magnitude_phase,
    stft_dct_features,
    torch_model_factory,
)
from scars.models.baselines import capacity_match_variants
from scars.models.scars_net import SCARSNet
from scars.results.registry import RECOGNITION_CONDITIONS


def test_baseline_registry_is_exact_and_reachable():
    definitions = baseline_definitions()
    assert [item.condition_id for item in definitions] == RECOGNITION_CONDITIONS
    torch_ids = [item.condition_id for item in definitions if item.trainer.startswith(("torch", "scars"))]
    for condition_id in torch_ids:
        assert torch_model_factory(condition_id, class_count=3)() is not None


def test_registered_feature_shapes_and_semantics():
    rng = np.random.default_rng(4)
    iq = rng.normal(size=(3, 512)) + 1j * rng.normal(size=(3, 512))
    assert magnitude_phase(iq).shape == (3, 2, 512)
    assert cwt_images(iq, 16).shape == (3, 1, 16, 16)
    assert stft_dct_features(iq).shape == (3, 64)
    assert fixed_wavelet_subband_features(iq).shape == (3, 8)
    assert log_psd_sobel_images(iq).shape == (3, 3, 16, 16)


def test_source_standardizer_is_source_fit_only():
    values = np.arange(48, dtype=np.float32).reshape(2, 2, 12)
    with pytest.raises(PermissionError):
        SourceGlobalStandardizer().fit(values, split_role="source_validation")
    fitted = SourceGlobalStandardizer().fit(values, split_role="source_fit")
    transformed = fitted.transform(values)
    assert np.allclose(transformed.mean(axis=(0, 2)), 0.0, atol=1e-6)


def test_patch_transformer_and_waveform_contracts():
    patch = torch_model_factory("patch_transformer", class_count=3)()
    assert patch(torch.zeros(2, 1, 16, 16)).shape == (2, 3)
    raw = torch_model_factory("raw_iq_cnn", class_count=3)()
    assert raw(torch.zeros(2, 2, 512)).shape == (2, 3)


def test_capacity_matching_emits_both_variants_and_gates():
    reference = SCARSNet(("W", "C", "E", "S"), 3)
    result = capacity_match_variants(reference, 4, 3)
    assert set(result) == {"reference", "parameter_matched", "mac_matched"}
    assert result["parameter_matched"]["parameter_gate"]
    assert result["mac_matched"]["mac_gate"]
