from __future__ import annotations

import numpy as np
import torch

from scars.data.windowing import WindowBatch
from scars.selection.nuisance import registered_nuisance_cases
from scars.training.disk_cache import (
    ChannelSubsetArray,
    build_relation_tensor_cache,
    build_teacher_family_training_array,
    iter_relation_iq_chunks,
)
from scars.cli.train_source_models import _teacher_checkpoint_resumable
from scars.models.scars_net import FamilyTeacher
from scars.results.provenance import sha256_file


class _ToyRepresentation:
    """Small deterministic [N,W/C/E/S,H,W] stand-in for storage tests."""

    def transform(self, iq: np.ndarray) -> np.ndarray:
        iq = np.asarray(iq, dtype=np.complex64)
        base = np.stack((iq.real[:, :4], iq.imag[:, :4]), axis=1).reshape(len(iq), 1, 2, 4)
        return np.concatenate((base, base + 1, base + 2, base + 3), axis=1)


def _batch() -> WindowBatch:
    iq = np.asarray(
        [np.arange(8) + 1j * np.arange(8), np.arange(8, 16) + 2j * np.arange(8)],
        dtype=np.complex64,
    )
    return WindowBatch(
        iq=iq,
        labels=np.asarray(["background", "uav"], dtype=object),
        recording_ids=np.asarray(["r0", "r1"], dtype=object),
        domains=np.asarray(["source", "source"], dtype=object),
        starts=np.asarray([0, 0]),
        ends=np.asarray([8, 8]),
    )


def test_relation_disk_cache_preserves_historical_seeded_order(tmp_path):
    batch = _batch()
    cache = build_relation_tensor_cache(
        tmp_path / "relations", batch, _ToyRepresentation(), seed=79,
        active_indices=(0, 1, 2, 3), representation_sha256="toy", chunk_size=3,
    )
    expected_clean, expected_perturbed = [], []
    for _, _, clean, perturbed in iter_relation_iq_chunks(batch, seed=79, chunk_size=2):
        expected_clean.append(_ToyRepresentation().transform(clean))
        expected_perturbed.append(_ToyRepresentation().transform(perturbed))
    np.testing.assert_allclose(cache.clean, np.concatenate(expected_clean))
    np.testing.assert_allclose(cache.perturbed, np.concatenate(expected_perturbed))
    recordings, labels, nuisances, severities = cache.metadata(0, len(cache.clean))
    cases = registered_nuisance_cases()
    assert recordings.tolist() == ["r0"] * len(cases) + ["r1"] * len(cases)
    assert labels.tolist() == ["background"] * len(cases) + ["uav"] * len(cases)
    assert nuisances[0] == cases[0].id.split(":", 1)[0]
    assert severities[-1] == cases[-1].severity


def test_teacher_family_bank_matches_legacy_clean_then_cycle_order(tmp_path):
    batch = _batch()
    bank = build_teacher_family_training_array(
        tmp_path / "teacher", batch, _ToyRepresentation(), family_index=2, seed=701,
        chunk_size=1,
    )
    cases = registered_nuisance_cases()
    perturbed = []
    for index, window in enumerate(batch.iq):
        case = cases[index % len(cases)]
        perturbed.append(case.transform(window, np.random.default_rng(701 + index), case.severity))
    representation = _ToyRepresentation()
    expected = np.concatenate(
        (representation.transform(batch.iq)[:, 2:3], representation.transform(np.asarray(perturbed))[:, 2:3])
    )
    np.testing.assert_allclose(bank, expected)
    subset = ChannelSubsetArray(np.asarray(cache := representation.transform(batch.iq)), (1, 3))
    np.testing.assert_allclose(subset[[0, 1]], cache[[0, 1]][:, [1, 3]])


def test_legacy_teachers_are_adopted_sequentially_only_with_explicit_amendment(tmp_path):
    classes = np.asarray(["background", "uav"], dtype=object)
    checkpoint = tmp_path / "teacher-W.pt"
    torch.save(
        {
            "family": "W",
            "classes": classes.tolist(),
            "temperature": 1.0,
            "training_report": {"seed": 11},
            "state_dict": FamilyTeacher(2).state_dict(),
        },
        checkpoint,
    )
    kwargs = dict(
        fold_dir=tmp_path, fold_id="fold-00", source_freeze_sha256="freeze",
        representation_sha256="representation", family="W", family_index=0,
        active_families=("W", "C"), classes=classes, fold_index=0,
    )
    assert _teacher_checkpoint_resumable(**kwargs, allow_legacy_adoption=True)
    (tmp_path / "teacher_resume.json").write_text(
        '{"schema_version":"scars-teacher-resume-1.0","fold_id":"fold-00",'
        '"source_freeze_sha256":"freeze","representation_sha256":"representation",'
        '"active_families":["W","C"],"classes":["background","uav"],'
        '"teachers":{"W":{"sha256":"' + sha256_file(checkpoint) + '"}}}',
        encoding="utf-8",
    )
    assert _teacher_checkpoint_resumable(**kwargs, allow_legacy_adoption=True)
