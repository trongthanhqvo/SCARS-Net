from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from scars.cli.collect_static_metadata import build_static_metadata
from scars.evaluation.costs import estimate_model_macs


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_attention_macs_include_projections_and_pairwise_attention():
    class SelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.MultiheadAttention(8, 2, batch_first=True)

        def forward(self, value):
            return self.attention(value, value, value)[0]

    attention = SelfAttention()
    example = torch.zeros(1, 4, 8)
    observed = estimate_model_macs(attention, example)
    expected = 4 * 1 * 4 * 8 * 8 + 2 * 1 * 4 * 4 * 8
    assert observed == expected


def test_static_metadata_contains_only_symbolic_or_protocol_model_counts():
    metadata = build_static_metadata(PROJECT_ROOT)
    assert metadata["status"] == "computed_without_real_iq_or_training"
    assert metadata["empirical_evidence"] is False
    assert metadata["registry_counts"]["recognition_conditions"] == 15
    assert metadata["registry_counts"]["ablation_conditions"] == 30
    assert metadata["registry_counts"]["nominal_recognition_fold_seed_cells"] == 213
    assert metadata["registry_counts"]["ablation_fold_seed_cells"] is None
    assert metadata["registry_counts"]["ablation_fold_seed_formula"] == (
        "sum_d (26 + 5 A_d) = 78 + 5 sum_d A_d for three eligible folds"
    )
    assert metadata["representation"]["nominal_full_tensor_shape"] == [4, 16, 16]
    assert metadata["representation"]["nominal_full_tensor_bytes"] == 4096
    assert metadata["representation"]["frames_per_window"] == 125
    assert metadata["representation"]["cyclic_frequency_values"] is None
    assert metadata["real_run_only"]["shared_ontology_and_class_count_K"] is None
    assert metadata["models"]["conditions"]["pcrd"]["parameters"] is None
    assert metadata["models"]["conditions"]["pcrd"]["parameter_formula"]["per_class"] > 0
