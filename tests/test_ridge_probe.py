from __future__ import annotations

import numpy as np

from scars.probes.ridge import RidgeProbe


def _direct_primal(features: np.ndarray, labels: np.ndarray, l2: float) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1.0e-6)
    z = np.column_stack(((x - mean) / scale, np.ones(len(x))))
    classes = np.unique(labels)
    target = np.stack([(labels == label).astype(float) for label in classes], axis=1)
    return np.linalg.solve(z.T @ z + l2 * np.eye(z.shape[1]), z.T @ target)


def test_ridge_uses_feature_sized_primal_system_when_samples_are_large():
    rng = np.random.default_rng(24021)
    features = rng.normal(size=(200, 12))
    labels = np.arange(len(features)) % 3
    probe = RidgeProbe(l2=0.01).fit(features, labels)
    assert probe.solver_ == "primal"
    np.testing.assert_allclose(
        probe.weights_,
        _direct_primal(features, labels, probe.l2),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_ridge_dual_and_primal_closed_forms_agree_when_features_are_wide():
    rng = np.random.default_rng(24022)
    features = rng.normal(size=(8, 20))
    labels = np.arange(len(features)) % 2
    probe = RidgeProbe(l2=0.01).fit(features, labels)
    assert probe.solver_ == "dual"
    expected = _direct_primal(features, labels, probe.l2)
    np.testing.assert_allclose(probe.weights_, expected, rtol=1.0e-9, atol=1.0e-9)


def test_ridge_artifact_round_trip_preserves_predictions_and_solver():
    rng = np.random.default_rng(24023)
    features = rng.normal(size=(40, 6))
    labels = np.arange(len(features)) % 2
    fitted = RidgeProbe().fit(features, labels)
    restored = RidgeProbe.from_source_artifact(fitted.source_artifact())
    assert restored.solver_ == "primal"
    np.testing.assert_allclose(
        restored.predict_proba(features),
        fitted.predict_proba(features),
        rtol=0.0,
        atol=0.0,
    )
