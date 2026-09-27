import json
import pickle

import pytest
import torch

from scars.cli.train_source_models import _torch_load_checkpoint


def test_old_float_reader_resumes_only_explicitly_trusted_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "teacher-W.pt"
    torch.save({"temperature": 1.25, "state_dict": {"weight": torch.ones(2)}}, path)
    original = torch.load
    calls = []

    def legacy_load(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        if kwargs.get("weights_only"):
            raise pickle.UnpicklingError("WeightsUnpickler error: Unsupported operand 71")
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", legacy_load)
    with pytest.raises(pickle.UnpicklingError):
        _torch_load_checkpoint(path)
    assert calls == [True]
    with pytest.warns(RuntimeWarning, match="trusted campaign"):
        payload = _torch_load_checkpoint(path, trusted_campaign_checkpoint=True)
    assert payload["temperature"] == 1.25
    assert calls == [True, True, False]
    assert torch.equal(payload["state_dict"]["weight"], torch.ones(2))


def test_hash_mismatch_rejected_before_deserialization(tmp_path, monkeypatch):
    path = tmp_path / "teacher-W.pt"
    path.write_bytes(b"changed checkpoint")
    (tmp_path / "teacher_resume.json").write_text(json.dumps(
        {"teachers": {"W": {"path": path.name, "sha256": "incorrect"}}}
    ))
    def forbidden(*args, **kwargs):
        pytest.fail("Checkpoint must not be deserialized after a hash mismatch")
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(RuntimeError, match="hash mismatch before loading"):
        _torch_load_checkpoint(path, trusted_campaign_checkpoint=True)


def test_other_unpickling_errors_do_not_enable_fallback(tmp_path, monkeypatch):
    path = tmp_path / "teacher-W.pt"
    path.write_bytes(b"invalid checkpoint")
    calls = []
    def invalid(*args, **kwargs):
        calls.append(kwargs["weights_only"])
        raise pickle.UnpicklingError("Unsupported global")
    monkeypatch.setattr(torch, "load", invalid)
    with pytest.raises(pickle.UnpicklingError):
        _torch_load_checkpoint(path, trusted_campaign_checkpoint=True)
    assert calls == [True]
