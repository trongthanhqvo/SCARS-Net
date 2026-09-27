from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class NuisanceCase:
    id: str
    severity: float
    transform: Callable[[np.ndarray, np.random.Generator, float], np.ndarray]


@dataclass(frozen=True)
class NuisanceSpec:
    """Machine-checkable scientific contract for one registered transform."""

    id: str
    severities: tuple[float, ...]
    parameter: str
    units: str
    mathematical_transform: str
    random_variables: str
    normalization: str
    seed_policy: str
    failure_conditions: str


def registered_execution_seed_contract() -> dict[str, object]:
    """Exact call-site seed/order contract, frozen before target access."""
    return {
        "source_instability": {
            "window_order": "recording_id_lexicographic_earliest_valid_window",
            "fold_base_seed": "24021+10000*fold_index",
            "case_seed": "fold_base_seed+case_index",
            "generator_scope": "one_generator_per_case_consumed_in_window_order",
            "paired_across_configurations": True,
        },
        "family_teacher_cycle": {
            "window_order": "capped_source_fit_pool_manifest_recording_order_then_start",
            "fold_base_seed": "70000+fold_index",
            "case_index": "window_index_mod_registered_case_count",
            "window_seed": "fold_base_seed+window_index",
        },
        "calibration_relations": {
            "window_order": "recording_id_lexicographic_earliest_valid_window",
            "fold_base_seed": "75000+fold_index",
            "cell_seed": "fold_base_seed+1009*window_index+case_index",
        },
        "source_fit_relations": {
            "window_order": "recording_id_lexicographic_earliest_valid_window",
            "fold_base_seed": "80000+fold_index",
            "cell_seed": "fold_base_seed+1009*window_index+case_index",
        },
        "target_robustness_snr_sir": {
            "window_order": "sealed_target_cache_recording_order_then_start",
            "seed": "26000+101*severity_index+window_index",
            "authorization_required": True,
        },
        "target_registered_nuisances": {
            "window_order": "sealed_target_cache_recording_order_then_start",
            "seed": "36000+1009*case_index+window_index",
            "authorization_required": True,
        },
    }


def _validated(x: np.ndarray, severity: float) -> tuple[np.ndarray, float]:
    signal = np.asarray(x)
    value = float(severity)
    if signal.ndim != 1 or signal.size == 0 or not np.iscomplexobj(signal):
        raise ValueError("A nuisance transform requires one nonempty complex-IQ window")
    if not np.all(np.isfinite(signal)) or not np.isfinite(value):
        raise ValueError("A nuisance transform requires finite IQ and finite severity")
    return signal, value


def awgn(x: np.ndarray, rng: np.random.Generator, snr_db: float) -> np.ndarray:
    x, snr_db = _validated(x, snr_db)
    power = max(float(np.mean(np.abs(x) ** 2)), 1.0e-12)
    noise_power = power / 10.0 ** (snr_db / 10.0)
    noise = rng.normal(size=x.size) + 1j * rng.normal(size=x.size)
    return x + noise * np.sqrt(noise_power / 2.0)


def global_phase(x: np.ndarray, rng: np.random.Generator, phase_rad: float) -> np.ndarray:
    x, phase_rad = _validated(x, phase_rad)
    del rng
    return x * np.exp(1j * phase_rad)


def cfo(x: np.ndarray, rng: np.random.Generator, cycles_per_sample: float) -> np.ndarray:
    x, cycles_per_sample = _validated(x, cycles_per_sample)
    del rng
    return x * np.exp(2j * np.pi * cycles_per_sample * np.arange(x.size))


def timing_shift(x: np.ndarray, rng: np.random.Generator, samples: float) -> np.ndarray:
    x, samples = _validated(x, samples)
    del rng
    return np.roll(x, int(samples))


def colored_noise(x: np.ndarray, rng: np.random.Generator, snr_db: float) -> np.ndarray:
    x, snr_db = _validated(x, snr_db)
    white = rng.normal(size=x.size + 2) + 1j * rng.normal(size=x.size + 2)
    taps = np.asarray([0.2, 0.6, 0.2], dtype=float)
    taps /= np.linalg.norm(taps)
    colored = np.convolve(white, taps, mode="valid")
    signal_power = max(float(np.mean(np.abs(x) ** 2)), 1.0e-12)
    noise_power = max(float(np.mean(np.abs(colored) ** 2)), 1.0e-12)
    target = signal_power / 10.0 ** (snr_db / 10.0)
    return x + colored * np.sqrt(target / noise_power)


def multipath(x: np.ndarray, rng: np.random.Generator, max_delay: float) -> np.ndarray:
    x, max_delay = _validated(x, max_delay)
    delay = max(1, int(max_delay))
    phase = rng.uniform(-np.pi, np.pi)
    taps = np.asarray([1.0, 0.5 * np.exp(1j * phase)], dtype=np.complex128)
    taps /= np.linalg.norm(taps)
    y = taps[0] * x + taps[1] * np.roll(x, delay)
    return y


def flat_fading(x: np.ndarray, rng: np.random.Generator, amplitude: float) -> np.ndarray:
    x, amplitude = _validated(x, amplitude)
    return x * float(amplitude) * np.exp(1j * rng.uniform(-np.pi, np.pi))


def time_scaling(x: np.ndarray, rng: np.random.Generator, fraction: float) -> np.ndarray:
    x, fraction = _validated(x, fraction)
    del rng
    source = np.arange(x.size, dtype=float)
    center = 0.5 * (x.size - 1)
    query = (source - center) / (1.0 + float(fraction)) + center
    real = np.interp(query, source, np.real(x), left=0.0, right=0.0)
    imag = np.interp(query, source, np.imag(x), left=0.0, right=0.0)
    return real + 1j * imag


def bandpass_response(x: np.ndarray, rng: np.random.Generator, ripple_db: float) -> np.ndarray:
    x, ripple_db = _validated(x, ripple_db)
    del rng
    ripple = 10.0 ** (float(ripple_db) / 20.0) - 1.0
    taps = np.asarray([0.25 * ripple, 1.0, -0.25 * ripple])
    taps /= max(abs(np.sum(taps)), 1.0e-12)
    return np.convolve(x, taps, mode="same")


def ism_mixing(x: np.ndarray, rng: np.random.Generator, sir_db: float) -> np.ndarray:
    x, sir_db = _validated(x, sir_db)
    n = np.arange(x.size)
    hop_length = 64
    hop_bins = rng.choice(np.asarray([0.07, 0.13, 0.21, 0.31]), size=int(np.ceil(x.size / hop_length)))
    frequency = np.repeat(hop_bins, hop_length)[: x.size]
    interferer = np.exp(2j * np.pi * np.cumsum(frequency))
    interferer *= rng.choice(np.asarray([-1.0, 1.0]), size=x.size)
    signal_power = max(float(np.mean(np.abs(x) ** 2)), 1.0e-12)
    mixing_power = signal_power / 10.0 ** (float(sir_db) / 10.0)
    return x + interferer * np.sqrt(mixing_power)


def registered_nuisance_cases() -> tuple[NuisanceCase, ...]:
    """Exact ordered transform/severity grid frozen in configs/nuisance_grid.yaml."""
    transforms = {
        "awgn": awgn,
        "colored_noise": colored_noise,
        "multipath": multipath,
        "flat_fading": flat_fading,
        "carrier_frequency_offset": cfo,
        "global_phase": global_phase,
        "timing_offset": timing_shift,
        "time_scaling": time_scaling,
        "bandpass_response": bandpass_response,
        "ism_mixing": ism_mixing,
    }
    return tuple(
        NuisanceCase(f"{spec.id}:{severity:.10g}", severity, transforms[spec.id])
        for spec in registered_nuisance_specs()
        for severity in spec.severities
    )


def registered_nuisance_specs() -> tuple[NuisanceSpec, ...]:
    """The exact contract mirrored verbatim by ``configs/nuisance_grid.yaml``.

    Every stochastic call receives a NumPy ``Generator`` whose seed is derived
    before target access.  Source-instability uses ``base_seed + case_index``
    and consumes windows in recording-ID lexicographic order. Relation construction uses
    ``base_seed + 1009*window_index + case_index``.  The complete source tree,
    config files, base seeds, manifest order, and emitted relation cache are
    SHA-256 bound by the campaign provenance.
    """
    seeded = (
        "numpy_generator_from_registered_base_seed; source_instability="
        "fold_base+case_index_in_recording_id_lexicographic_order_shared_across_configurations; "
        "relation=base+1009*window_index+case_index"
    )
    deterministic = "no_random_variable; generator_argument_ignored"
    finite = "finite_complex_input_and_finite_registered_severity_required"
    return (
        NuisanceSpec(
            "awgn", (20.0, 10.0, 0.0, -10.0), "snr_db", "dB",
            "y=x+n; n=sqrt(Px/10^(snr_db/10)/2)*(u+jv); u,v~N(0,I)",
            "independent_standard_normal_real_and_imaginary_samples",
            "noise_scaled_to_measured_mean_window_power_Px_with_floor_1e-12",
            seeded, finite,
        ),
        NuisanceSpec(
            "colored_noise", (20.0, 10.0, 0.0), "snr_db", "dB",
            "y=x+beta*(h*w); h=[0.2,0.6,0.2]/||h||2; valid_convolution",
            "complex_white_standard_normal_length_N_plus_2",
            "beta_sets_filtered_noise_power_to_Px/10^(snr_db/10); power_floors_1e-12",
            seeded, finite,
        ),
        NuisanceSpec(
            "multipath", (1.0, 4.0, 8.0), "delay_samples", "samples",
            "y=a0*x+a1*roll(x,delay); [a0,a1]=[1,0.5*exp(j*phi)]/sqrt(1.25)",
            "phi~Uniform(-pi,pi); delay_is_exact_not_a_random_maximum",
            "two_taps_have_unit_total_energy; circular_delay_boundary",
            seeded, finite,
        ),
        NuisanceSpec(
            "flat_fading", (0.9, 0.7, 0.5), "amplitude_scale", "relative_amplitude",
            "y=amplitude*x*exp(j*phi)",
            "phi~Uniform(-pi,pi); amplitude_is_fixed_by_severity_not_Rayleigh_or_Rician",
            "no_post_transform_power_normalization",
            seeded, finite,
        ),
        NuisanceSpec(
            "carrier_frequency_offset", (0.0001, 0.0005, 0.001),
            "cycles_per_sample", "cycles_per_sample",
            "y[n]=x[n]*exp(j*2*pi*delta*n)", "none",
            "unit_modulus_multiplication", deterministic, finite,
        ),
        NuisanceSpec(
            "global_phase", (0.7853981634, 1.5707963268, 3.1415926536),
            "phase_rad", "radians", "y=x*exp(j*phi)", "none",
            "unit_modulus_multiplication", deterministic, finite,
        ),
        NuisanceSpec(
            "timing_offset", (1.0, 4.0, 16.0), "shift_samples", "samples",
            "y=roll(x,int(shift_samples))", "none",
            "circular_boundary; length_preserved", deterministic, finite,
        ),
        NuisanceSpec(
            "time_scaling", (0.0001, 0.001, 0.005), "fractional_scale", "fraction",
            "query=(n-center)/(1+fraction)+center; real_and_imag_linear_interpolation",
            "none", "fixed_length_output; zero_extrapolation_outside_input_support",
            deterministic, finite,
        ),
        NuisanceSpec(
            "bandpass_response", (0.5, 1.0, 3.0), "ripple_db", "dB",
            "r=10^(ripple_db/20)-1; h=[0.25r,1,-0.25r]/abs(sum(h)); y=conv_same(x,h)",
            "none", "unit_DC_gain_three_tap_FIR; zero_padding_at_convolution_boundary",
            deterministic, finite,
        ),
        NuisanceSpec(
            "ism_mixing", (20.0, 10.0, 0.0, -10.0), "sir_db", "dB",
            "y=x+sqrt(Px/10^(sir_db/10))*i; i=exp(j*2*pi*cumsum(f))*b",
            "each_64_sample_hop_frequency_uniformly_selected_from_[.07,.13,.21,.31]; b_n_uniform_in_{-1,+1}",
            "synthetic_FHSS_BPSK_like_surrogate_has_unit_sample_power; no_external_donor_or_alignment",
            seeded, finite,
        ),
    )


def default_smoke_cases() -> tuple[NuisanceCase, ...]:
    return (
        NuisanceCase("awgn_10db", 10.0, awgn),
        NuisanceCase("phase_pi_over_2", np.pi / 2.0, global_phase),
        NuisanceCase("cfo_5e-4", 5.0e-4, cfo),
        NuisanceCase("timing_4", 4.0, timing_shift),
    )
