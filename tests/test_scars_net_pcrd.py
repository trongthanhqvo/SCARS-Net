from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from scars.models.scars_net import SCARSNet
from scars.training.pcrd import (
    ParetoRelation,
    build_pareto_relations,
    pcrd_macro_loss,
    relation_macro_probabilities,
    sensitivity_scale,
    shuffle_relations_within_strata,
)


def test_scars_net_has_strict_single_fused_path():
    model = SCARSNet(("W", "C", "E", "S"), class_count=3)
    output = model(torch.randn(5, 4, 16, 16))
    assert output["logits"].shape == (5, 3)
    assert output["gate_logits"].shape == (5, 4)
    assert output["tokens"].shape == (5, 4, 128)
    torch.testing.assert_close(output["weights"].sum(dim=1), torch.ones(5))
    assert len({id(model.stems[name]) for name in model.active_families}) == 4
    assert not hasattr(model, "branch_heads")


def test_pareto_relations_leave_incomparable_pairs_unordered():
    risk = np.asarray([[0.1, 0.2, 0.05], [0.1, 0.2, 0.3]])
    margin = np.asarray([[0.8, 0.7, 0.6], [0.4, 0.5, 0.2]])
    relations = build_pareto_relations(
        risk,
        margin,
        ("W", "C", "E"),
        ("r0", "r1"),
        ("a", "a"),
        ("awgn", "awgn"),
        (10.0, 10.0),
    )
    first_pairs = {(item.winner_family, item.loser_family) for item in relations if item.sample_index == 0}
    assert ("W", "C") in first_pairs
    assert not ({("W", "E"), ("E", "W")} & first_pairs)


def test_pcrd_hinge_pushes_winner_logit_above_loser():
    relation = ParetoRelation(0, "r", "a", "awgn", 10.0, 0, 1, "W", "C")
    good = torch.tensor([[0.5, 0.0]], requires_grad=True)
    bad = torch.tensor([[0.0, 0.5]], requires_grad=True)
    assert float(pcrd_macro_loss(good, [relation])) == pytest.approx(0.0)
    loss = pcrd_macro_loss(bad, [relation])
    assert float(loss) == pytest.approx(0.7)
    loss.backward()
    assert bad.grad[0, 0] < 0
    assert bad.grad[0, 1] > 0


def test_shuffled_relation_control_preserves_registered_strata():
    relations = [
        ParetoRelation(index, f"r{index}", "a", "awgn", 10.0, index % 2, 1 - index % 2, "W" if index % 2 == 0 else "C", "C" if index % 2 == 0 else "W")
        for index in range(4)
    ]
    shuffled = shuffle_relations_within_strata(relations, seed=9)
    assert [(item.label, item.nuisance, item.severity) for item in shuffled] == [
        (item.label, item.nuisance, item.severity) for item in relations
    ]
    assert sorted((item.winner, item.loser) for item in shuffled) == sorted(
        (item.winner, item.loser) for item in relations
    )
    assert [(item.winner, item.loser) for item in shuffled] != [
        (item.winner, item.loser) for item in relations
    ]


def test_sensitivity_scale_fails_closed_on_nonfinite_or_empty_support():
    logp = np.asarray([[0.0, -1.0]])
    tensor = np.zeros((1, 1, 2, 2), dtype=np.float32)
    with pytest.raises(RuntimeError, match="nonfinite"):
        sensitivity_scale(
            np.asarray([[np.nan, -1.0]]), logp, tensor, tensor, ["r0"]
        )
    with pytest.raises(ValueError, match="physical recording"):
        sensitivity_scale(
            np.empty((0, 2)),
            np.empty((0, 2)),
            np.empty((0, 1, 2, 2)),
            np.empty((0, 1, 2, 2)),
            [],
        )
