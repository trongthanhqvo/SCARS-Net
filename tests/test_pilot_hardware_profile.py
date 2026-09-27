from __future__ import annotations

from scars.cli.preflight_mat import _audit_hardware


def _rtx3060_host() -> dict[str, object]:
    return {
        "ram_bytes": 15 * 1024**3,
        "gpu": {
            "name": "NVIDIA GeForce RTX 3060",
            "vram_bytes": 12 * 1024**3,
        },
    }


def test_two_dataset_pilot_records_compatible_host_differences_as_warnings():
    issues, warnings = _audit_hardware(
        _rtx3060_host(),
        10 * 1024**3,
        pilot_two_dataset=True,
    )
    assert issues == []
    assert len(warnings) == 3


def test_confirmatory_campaign_keeps_registered_hardware_profile_strict():
    issues, warnings = _audit_hardware(
        _rtx3060_host(),
        10 * 1024**3,
        pilot_two_dataset=False,
    )
    assert warnings == []
    assert len(issues) == 3


def test_cuda_and_vram_floor_remain_blocking_for_pilot():
    no_gpu_issues, _ = _audit_hardware(
        {"ram_bytes": 16 * 1024**3, "gpu": None},
        1,
        pilot_two_dataset=True,
    )
    low_vram_issues, _ = _audit_hardware(
        {
            "ram_bytes": 16 * 1024**3,
            "gpu": {"name": "Any CUDA GPU", "vram_bytes": 6 * 1024**3},
        },
        1,
        pilot_two_dataset=True,
    )
    assert no_gpu_issues == ["CUDA GPU is unavailable"]
    assert low_vram_issues == ["CUDA VRAM is below the 7 GiB safe envelope"]
