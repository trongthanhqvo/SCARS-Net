from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


FAMILY_ORDER = ("W", "C", "E", "S")


def _groups(channels: int) -> int:
    for value in (8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class DepthwiseResidual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class FamilyStem(nn.Module):
    """Topology-matched family stem; instances never share parameters."""

    def __init__(self, token_dim: int = 128, width: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
            DepthwiseResidual(width),
            nn.Conv2d(width, token_dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(token_dim), token_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).flatten(1)


class SCARSNet(nn.Module):
    """Exact single-observation SCARS-Net inference path from the manuscript.

    Input is `[B, |A|, H, W]` in the source-frozen active-family order.  There
    is one non-shared stem per family, one gate, strict weighted-sum fusion and
    one fused classifier.  No branch classifier or ungated bypass is exposed.
    """

    def __init__(
        self,
        active_families: Iterable[str],
        class_count: int,
        *,
        token_dim: int = 128,
        stem_width: int = 32,
        gate_dropout: float = 0.1,
    ):
        super().__init__()
        active = tuple(active_families)
        if not active or any(family not in FAMILY_ORDER for family in active):
            raise ValueError(f"active_families must be a non-empty subset of {FAMILY_ORDER}")
        if active != tuple(family for family in FAMILY_ORDER if family in active):
            raise ValueError("active_families must preserve W/C/E/S order")
        if class_count < 2:
            raise ValueError("class_count must be at least two")
        self.active_families = active
        self.token_dim = int(token_dim)
        self.stems = nn.ModuleDict(
            {family: FamilyStem(token_dim, stem_width) for family in self.active_families}
        )
        self.gate = nn.Sequential(
            nn.Linear(token_dim * len(active), token_dim),
            nn.GELU(),
            nn.Dropout(gate_dropout),
            nn.Linear(token_dim, len(active)),
        )
        self.classifier = nn.Linear(token_dim, class_count)

    def forward(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        if tensor.ndim != 4 or tensor.shape[1] != len(self.active_families):
            raise ValueError(
                "SCARS-Net expects [batch, active_families, height, width] in frozen order"
            )
        tokens = torch.stack(
            [self.stems[family](tensor[:, index : index + 1]) for index, family in enumerate(self.active_families)],
            dim=1,
        )
        gate_logits = self.gate(tokens.flatten(1))
        weights = torch.softmax(gate_logits, dim=1)
        fused = torch.sum(weights.unsqueeze(-1) * tokens, dim=1)
        logits = self.classifier(fused)
        return {
            "logits": logits,
            "gate_logits": gate_logits,
            "weights": weights,
            "tokens": tokens,
            "fused": fused,
        }


class FamilyTeacher(nn.Module):
    """One frozen source-fit encoder/head for one representation family."""

    def __init__(self, class_count: int, *, token_dim: int = 128, stem_width: int = 32):
        super().__init__()
        self.encoder = FamilyStem(token_dim, stem_width)
        self.head = nn.Linear(token_dim, class_count)

    def forward(self, family_tensor: torch.Tensor) -> torch.Tensor:
        if family_tensor.ndim != 4 or family_tensor.shape[1] != 1:
            raise ValueError("FamilyTeacher expects [batch, 1, height, width]")
        return self.head(self.encoder(family_tensor))
