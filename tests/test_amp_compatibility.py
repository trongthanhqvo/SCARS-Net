from types import SimpleNamespace
from unittest.mock import Mock

import torch

from scars.training import engine


def test_modern_scaler_uses_device_argument(monkeypatch):
    modern, legacy = Mock(), Mock()
    monkeypatch.setattr(engine, "torch", SimpleNamespace(
        amp=SimpleNamespace(GradScaler=modern),
        cuda=SimpleNamespace(amp=SimpleNamespace(GradScaler=legacy))))
    assert engine.make_grad_scaler(True) is modern.return_value
    modern.assert_called_once_with("cuda", enabled=True)
    legacy.assert_not_called()


def test_legacy_scaler_when_torch_amp_has_no_grad_scaler(monkeypatch):
    legacy = Mock()
    monkeypatch.setattr(engine, "torch", SimpleNamespace(
        amp=SimpleNamespace(),
        cuda=SimpleNamespace(amp=SimpleNamespace(GradScaler=legacy))))
    assert engine.make_grad_scaler(True) is legacy.return_value
    legacy.assert_called_once_with(enabled=True)


def test_disabled_scaler_performs_cpu_optimizer_step():
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = engine.make_grad_scaler(False)
    scaler.scale(parameter.square()).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.testing.assert_close(parameter.detach(), torch.tensor(1.6))
