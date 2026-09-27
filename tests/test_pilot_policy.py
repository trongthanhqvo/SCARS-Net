import copy
import json

import numpy as np
import pytest

from scars.experiment.pilot_policy import POLICY, apply_fixed_families, finite_json, is_pilot
from scars.probes.xgboost_probe import XGBoostProbe


def test_pilot_amendment_preserves_measured_selection():
    freeze = {"active_families": [], "channel_decisions": {"retain": False}}
    decisions = copy.deepcopy(freeze["channel_decisions"])
    apply_fixed_families(freeze)
    assert freeze["registered_active_families"] == []
    assert freeze["active_families"] == ["W", "C", "E", "S"]
    assert freeze["channel_decisions"] == decisions
    assert freeze["pilot_policy"] == POLICY
    assert not is_pilot({"campaign_mode": "real_confirmatory"})


def test_unavailable_pilot_diagnostics_are_null_not_zero():
    value = finite_json({"probe": float("nan"), "valid": 0.0, "nested": [float("inf")]})
    assert value == {"probe": None, "valid": 0.0, "nested": [None]}
    json.dumps(value, allow_nan=False)


def test_binary_xgboost_string_labels_safe_roundtrip(tmp_path):
    pytest.importorskip("xgboost")
    x = np.arange(80, dtype=np.float32).reshape(20, 4)
    y = np.array(["background", "uav"] * 10, dtype=object)
    probe = XGBoostProbe(seed=11, n_estimators=2).fit(x, y, x, y)
    path = tmp_path / "model.json"
    probe.save_model(path)
    restored = XGBoostProbe.load_model(path, 11)
    assert restored.classes_.tolist() == ["background", "uav"]
    np.testing.assert_allclose(probe.predict_proba(x), restored.predict_proba(x))
