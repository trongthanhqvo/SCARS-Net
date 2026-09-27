from __future__ import annotations

from copy import deepcopy

from scars.results.registry import RECOGNITION_CONDITIONS


POLICY_VERSION = "sampling-contract-20260829"

ONE_WINDOW_NEURAL_CONDITIONS = (
    "raw_iq_cnn",
    "patch_transformer",
    "magnitude_phase_cnn",
    "log_stft_cnn",
    "cwt_cnn",
    "wst_only",
    "cyclic_only",
    "log_psd_sobel",
    "early_fusion_resnet",
    "uniform_late_fusion",
    "ordinary_gate",
    "shuffled_pcrd",
    "pcrd",
)

FULL_CAPPED_POOL_CONDITIONS = (
    "stft_dct_xgboost",
    "fixed_wavelet_subbands",
)


def frozen_sampling_contract() -> dict[str, object]:
    """Return the implementation-level sampling contract written into artifacts."""
    covered = set(ONE_WINDOW_NEURAL_CONDITIONS) | set(FULL_CAPPED_POOL_CONDITIONS)
    if covered != set(RECOGNITION_CONDITIONS):
        raise RuntimeError("Every recognition condition must have exactly one sampling policy")
    return deepcopy(
        {
            "policy_version": POLICY_VERSION,
            "candidate_pool": {
                "selection": "evenly_spaced_valid_windows_in_temporal_order",
                "max_windows_per_recording": 64,
                "short_recording_policy": "use_all_valid_windows",
            },
            "pcrd_student_and_one_window_neural_conditions": {
                "condition_ids": list(ONE_WINDOW_NEURAL_CONDITIONS),
                "window_choice": "earliest_valid_window_in_frozen_candidate_pool",
                "clean_windows_per_recording": 1,
                "recording_balance": "exact",
                "nuisance_expansion": "all_registered_nuisance_severity_cases",
                "epoch_resampling": "none",
            },
            "relation_cache": {
                "source_role": "source_fit",
                "window_choice": "earliest_valid_window_in_frozen_candidate_pool",
                "clean_windows_per_recording": 1,
                "epoch_resampling": "not_applicable",
            },
            "calibration_scale_cache": {
                "source_role": "source_calibration",
                "window_choice": "earliest_valid_window_in_frozen_candidate_pool",
                "clean_windows_per_recording": 1,
                "epoch_resampling": "not_applicable",
            },
            "family_teachers": {
                "source_role": "source_fit",
                "window_choice": "all_capped_evenly_spaced_windows",
                "max_clean_windows_per_recording": 64,
                "augmentation": "one_deterministic_cycle_perturbation_per_clean_window",
                "recording_balance": "capped_not_exact",
                "epoch_resampling": "fixed_examples_seeded_shuffle",
            },
            "full_capped_pool_conditions": {
                "condition_ids": list(FULL_CAPPED_POOL_CONDITIONS),
                "window_choice": "all_capped_evenly_spaced_clean_windows",
                "max_clean_windows_per_recording": 64,
                "recording_balance": "capped_not_exact",
                "epoch_resampling": "not_applicable",
            },
        }
    )
