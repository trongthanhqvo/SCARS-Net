from __future__ import annotations

import numpy as np
import pytest

from scars.ablation import assert_exact_channel_masks, channel_mask_grid
from scars.baselines import H2_BANK_ORDER, h2_configuration_factory
from scars.representations.cyclostationary import (
    native_spectral_correlation_from_frames,
    permute_frozen_bins,
)
from scars.representations.normalization import SourceGlobalNormalizer
from scars.representations.tensor import RepresentationConfig, SourceFittedTensor
from scars.representations.scattering import _filter, scattering_path_count


def test_source_only_normalizer_rejects_target_fit():
    normalizer = SourceGlobalNormalizer()
    with pytest.raises(PermissionError):
        normalizer.fit_family("W", np.ones((2, 3)), "fold", "held_target")


def test_source_only_representation_fit_rejects_target(small_windows):
    representation = SourceFittedTensor(RepresentationConfig("C", use_w=False, cyclic_count=4))
    with pytest.raises(PermissionError):
        representation.fit(small_windows.iq, "fold", "held_target")


def test_energy_bypasses_percentile_and_all_shapes_match(small_windows):
    config = RepresentationConfig(
        "W+C+E+S", use_w=True, use_c=True, use_e=True, use_s=True,
        output_bins=4, wst_j=3, wst_q=1, cyclic_count=4,
    )
    representation = SourceFittedTensor(config).fit(small_windows.iq, "fold")
    tensor = representation.transform(small_windows.iq[:2])
    assert tensor.shape == (2, 4, 4, 4)
    assert "E" not in representation.normalizer.records
    assert representation.energy_reference is not None


def test_ablation_masks_remove_exact_families():
    assert_exact_channel_masks()
    grid = channel_mask_grid(4)
    assert grid["W+C+E+S"].active_families() == ("W", "C", "E", "S")


def test_cyclic_permutation_changes_frozen_order():
    frozen = np.asarray([2, 4, 8, 12])
    permuted = permute_frozen_bins(frozen, seed=19)
    assert set(permuted) == set(frozen)
    assert not np.array_equal(permuted, frozen)


def test_native_scf_phase_frame_permutation_and_bin_modulation_symmetries():
    rng = np.random.default_rng(41)
    chunks = rng.normal(size=(7, 128)) + 1j * rng.normal(size=(7, 128))
    alphas = np.asarray([4 / 128, 9 / 128], dtype=float)
    native = native_spectral_correlation_from_frames(chunks, alphas)
    np.testing.assert_allclose(
        native_spectral_correlation_from_frames(chunks * np.exp(1j * 0.73), alphas),
        native,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        native_spectral_correlation_from_frames(chunks[[3, 1, 6, 0, 5, 2, 4]], alphas),
        native,
        atol=1.0e-12,
    )
    bin_shift = 8
    modulation = np.exp(2j * np.pi * bin_shift * np.arange(128) / 128)
    modulated = native_spectral_correlation_from_frames(chunks * modulation, alphas)
    np.testing.assert_allclose(modulated, np.roll(native, bin_shift, axis=1), atol=1.0e-12)


def test_transform_is_deterministic_and_does_not_refit_cyclic_bins(small_windows):
    representation = SourceFittedTensor(
        RepresentationConfig("W+C", output_bins=4, wst_j=3, wst_q=1, cyclic_count=4)
    ).fit(small_windows.iq, "fold")
    frozen = representation.cyclic_bins.copy()
    first = representation.transform(small_windows.iq[:2])
    second = representation.transform(small_windows.iq[:2])
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(frozen, representation.cyclic_bins)


def test_cyclic_bins_have_unique_effective_scf_shifts(small_windows):
    representation = SourceFittedTensor(
        RepresentationConfig(
            "C-unique",
            use_w=False,
            use_c=True,
            output_bins=4,
            cyclic_count=8,
            frame_samples=128,
        )
    ).fit(small_windows.iq, "fold")
    shifts = representation.source_artifact()["effective_cyclic_shifts"]
    assert len(shifts) == len(set(shifts)) == 8


def test_scattering_bound_constants_are_logged_from_applied_operators(small_windows):
    representation = SourceFittedTensor(
        RepresentationConfig("W", use_w=True, use_c=False, output_bins=4, wst_j=3, wst_q=1)
    ).fit(small_windows.iq, "fold")
    metadata = representation.scattering_metadata
    assert metadata is not None
    assert len(metadata.path_l1_products) == len(metadata.paths)
    assert all(value > 0 for value in metadata.path_l1_products)
    assert metadata.pooling_operator_norm > 0
    assert metadata.resize_operator_norm > 0
    assert metadata.log1p_lipschitz_bound == 1.0
    assert representation.normalizer.records["W"].scale > 0
    assert metadata.exact_registered_discrete_graph is True
    assert metadata.continuous_wst_equivalence_claim is False
    assert metadata.backend == "registered_periodic_analytic_morlet_order12"
    assert len(metadata.paths) == sum(scattering_path_count(3, 1))


def test_registered_morlet_is_analytic_and_exactly_zero_mean_on_fft_grid():
    spectrum, _ = _filter(256, center=0.25, bandwidth=0.05)
    frequencies = np.fft.fftfreq(256)
    assert spectrum[0] == 0.0
    assert np.all(spectrum[frequencies <= 0] == 0.0)
    assert abs(np.sum(np.fft.ifft(spectrum))) < 1.0e-12


def test_returned_w_channel_obeys_logged_additive_binary32_bound(small_windows):
    representation = SourceFittedTensor(
        RepresentationConfig("W", use_w=True, use_c=False, output_bins=4, wst_j=3, wst_q=1)
    ).fit(small_windows.iq, "fold")
    metadata = representation.scattering_metadata
    record = representation.normalizer.records["W"]
    assert metadata is not None
    lipschitz = (
        metadata.resize_operator_norm
        * metadata.pooling_operator_norm
        * np.sqrt(np.sum(np.square(metadata.path_l1_products)))
        / record.scale
    )
    left, right = small_windows.iq[0], small_windows.iq[1]
    returned_left = representation.transform_one_by_family(left)["W"]
    returned_right = representation.transform_one_by_family(right)["W"]
    rho32 = np.finfo(np.float32).eps  # conservative absolute error on [0,1]
    additive_bound = lipschitz * np.linalg.norm(left - right) + 2 * 4 * rho32
    assert np.linalg.norm(returned_left - returned_right) <= additive_bound + 1.0e-12


def test_exact_h2_bank_order_is_executable(small_windows):
    bank = h2_configuration_factory()
    assert tuple(item.config.stable_id for item in bank) == H2_BANK_ORDER
    # Execute every registered ID; a family representative is insufficient for
    # the no-deletion confirmatory bank.
    for representation in bank:
        representation.fit(
            small_windows.iq,
            "fold",
            recording_ids=small_windows.recording_ids,
        )
        transformed = representation.transform(small_windows.iq[:1])
        assert transformed.shape[0] == 1


def test_disabled_families_do_not_contribute_macs(small_windows):
    w = SourceFittedTensor(RepresentationConfig("W", use_w=True, use_c=False, output_bins=4, wst_j=3, wst_q=1)).fit(small_windows.iq, "fold")
    c = SourceFittedTensor(RepresentationConfig("C", use_w=False, use_c=True, output_bins=4, cyclic_count=4)).fit(small_windows.iq, "fold")
    wc = SourceFittedTensor(RepresentationConfig("W+C", output_bins=4, wst_j=3, wst_q=1, cyclic_count=4)).fit(small_windows.iq, "fold")
    w_cost = w.measured_cost(small_windows.iq, warmups=0, repeats=1)["estimated_macs"]
    c_cost = c.measured_cost(small_windows.iq, warmups=0, repeats=1)["estimated_macs"]
    wc_cost = wc.measured_cost(small_windows.iq, warmups=0, repeats=1)["estimated_macs"]
    assert wc_cost == w_cost + c_cost


def test_resolution_sweep_has_strictly_increasing_bytes_and_macs(small_windows):
    costs = []
    for resolution in (8, 16, 32):
        representation = SourceFittedTensor(
            RepresentationConfig(
                f"resolution_{resolution}",
                output_bins=resolution,
                wst_j=3,
                wst_q=1,
                cyclic_count=4,
            )
        ).fit(small_windows.iq, "fold")
        costs.append(representation.measured_cost(small_windows.iq, warmups=0, repeats=1))
    assert costs[0]["bytes_per_sample"] < costs[1]["bytes_per_sample"] < costs[2]["bytes_per_sample"]
    assert costs[0]["estimated_macs"] < costs[1]["estimated_macs"] < costs[2]["estimated_macs"]
