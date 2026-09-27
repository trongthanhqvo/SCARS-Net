from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from .scars_net import FAMILY_ORDER, FamilyStem


def _group_count(channels: int) -> int:
    for count in (8, 4, 2, 1):
        if channels % count == 0:
            return count
    return 1


class Residual2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.skip(x))


class EarlyFusionResNet(nn.Module):
    def __init__(self, input_channels: int, class_count: int, base_width: int = 32):
        super().__init__()
        widths = (base_width, 2 * base_width, 4 * base_width)
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, widths[0], 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(widths[0]), widths[0]),
            nn.GELU(),
            Residual2D(widths[0], widths[0], 1),
            Residual2D(widths[0], widths[1], 2),
            Residual2D(widths[1], widths[2], 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(widths[2], class_count)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.network(tensor).flatten(1))


class RawIQ1DCNN(nn.Module):
    def __init__(self, class_count: int, input_channels: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, 9, padding=4, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, class_count)

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or iq.shape[1] != 2:
            raise ValueError("RawIQ1DCNN expects [batch, real_imag, samples]")
        return self.classifier(self.features(iq).flatten(1))


class MagnitudePhaseCNN(RawIQ1DCNN):
    """The input contract is [magnitude, unwrapped_phase], not real/imaginary."""


class PatchTransformer(nn.Module):
    def __init__(
        self,
        class_count: int,
        *,
        input_channels: int = 1,
        image_size: int = 16,
        patch_size: int = 4,
        embedding_dim: int = 128,
        layers: int = 4,
        heads: int = 4,
        mlp_ratio: int = 2,
    ):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.input_channels = input_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch = nn.Conv2d(
            input_channels, embedding_dim, patch_size, stride=patch_size
        )
        token_count = (image_size // patch_size) ** 2
        self.position = nn.Parameter(torch.zeros(1, token_count, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=embedding_dim * mlp_ratio,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.normalization = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, class_count)

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 4 or iq.shape[1] != self.input_channels:
            raise ValueError("PatchTransformer expects [batch,channels,height,width]")
        if tuple(iq.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError("PatchTransformer input must match its registered image size")
        tokens = self.patch(iq).flatten(2).transpose(1, 2) + self.position
        return self.classifier(self.normalization(self.encoder(tokens).mean(dim=1)))


class UniformLateFusion(nn.Module):
    def __init__(
        self,
        active_families: Iterable[str],
        class_count: int,
        *,
        token_dim: int = 128,
        stem_width: int = 32,
    ):
        super().__init__()
        active = tuple(active_families)
        if active != tuple(family for family in FAMILY_ORDER if family in active):
            raise ValueError("active families must preserve W/C/E/S order")
        self.active_families = active
        self.stems = nn.ModuleDict(
            {family: FamilyStem(token_dim, stem_width) for family in active}
        )
        self.classifier = nn.Linear(token_dim, class_count)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack(
            [self.stems[family](tensor[:, index : index + 1]) for index, family in enumerate(self.active_families)],
            dim=1,
        )
        return self.classifier(tokens.mean(dim=1))


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def capacity_match_width(
    reference_parameters: int,
    input_channels: int,
    class_count: int,
    *,
    tolerance: float = 0.05,
    candidates: range = range(8, 129),
) -> tuple[int, int, float]:
    records = []
    for width in candidates:
        count = parameter_count(EarlyFusionResNet(input_channels, class_count, width))
        records.append((abs(count / reference_parameters - 1.0), width, count))
    difference, width, count = min(records)
    if difference > tolerance:
        raise RuntimeError(
            f"No parameter-matched early-fusion width within {tolerance:.0%}; best={difference:.2%}"
        )
    return width, count, difference


def capacity_match_variants(
    reference_model: nn.Module,
    input_channels: int,
    class_count: int,
    *,
    height: int = 16,
    width: int = 16,
) -> dict[str, dict[str, float | int | bool]]:
    """Emit parameter- and MAC-nearest early-fusion variants and explicit gates."""
    import torch

    from scars.evaluation.costs import estimate_model_macs

    reference_example = torch.zeros(1, input_channels, height, width)
    reference_parameters = parameter_count(reference_model)
    reference_macs = estimate_model_macs(reference_model, reference_example)
    records = []
    for base_width in range(8, 129):
        candidate = EarlyFusionResNet(input_channels, class_count, base_width)
        parameters = parameter_count(candidate)
        macs = estimate_model_macs(candidate, reference_example)
        records.append((base_width, parameters, macs))
    parameter_best = min(records, key=lambda item: abs(item[1] / reference_parameters - 1.0))
    mac_best = min(records, key=lambda item: abs(item[2] / reference_macs - 1.0))

    def record(item: tuple[int, int, int]) -> dict[str, float | int | bool]:
        base_width, parameters, macs = item
        parameter_ratio = parameters / reference_parameters
        mac_ratio = macs / reference_macs
        return {
            "base_width": base_width,
            "parameters": parameters,
            "macs": macs,
            "parameter_ratio": parameter_ratio,
            "mac_ratio": mac_ratio,
            "parameter_gate": abs(parameter_ratio - 1.0) <= 0.05,
            "mac_gate": abs(mac_ratio - 1.0) <= 0.10,
        }

    return {
        "reference": {"parameters": reference_parameters, "macs": reference_macs},
        "parameter_matched": record(parameter_best),
        "mac_matched": record(mac_best),
    }
