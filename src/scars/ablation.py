from __future__ import annotations

from dataclasses import replace

from .representations.tensor import RepresentationConfig


def channel_mask_grid(output_bins: int = 16) -> dict[str, RepresentationConfig]:
    base = RepresentationConfig("W+C", output_bins=output_bins)
    return {
        "W": replace(base, stable_id="W", use_c=False),
        "C": replace(base, stable_id="C", use_w=False),
        "W+C": base,
        "W+C+E": replace(base, stable_id="W+C+E", use_e=True),
        "W+C+E+S": replace(base, stable_id="W+C+E+S", use_e=True, use_s=True),
    }


def assert_exact_channel_masks() -> None:
    expected = {
        "W": ("W",),
        "C": ("C",),
        "W+C": ("W", "C"),
        "W+C+E": ("W", "C", "E"),
        "W+C+E+S": ("W", "C", "E", "S"),
    }
    observed = {key: value.active_families() for key, value in channel_mask_grid().items()}
    if observed != expected:
        raise AssertionError(f"Ablation mask mismatch: {observed}")
