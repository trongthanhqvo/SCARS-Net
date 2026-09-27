from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pytest

from scars.data.adapters.local_samples import (
    assert_real_waveform_rejected,
    read_hdf5_compound_complex,
    read_hdf5_iq_pair,
    read_interleaved_float32,
)


# Public-release portability: local data access is opt-in, independent of where
# the repository is cloned. Synthetic adapter tests require no external corpus.
ROOT = (
    Path(os.environ["SCARS_LOCAL_DATASET_ROOT"]).expanduser()
    if os.environ.get("SCARS_LOCAL_DATASET_ROOT") else None
)


@pytest.mark.skipif(ROOT is None or not ROOT.is_dir(), reason="SCARS_LOCAL_DATASET_ROOT not configured")
def test_bounded_hdf5_pair_and_compound_adapters():
    drff = next((ROOT / "DRFF-R2_2026").rglob("*.mat"))
    pair = read_hdf5_iq_pair(drff, "RF0_I", "RF0_Q", count=128)
    assert pair.shape == (128,) and pair.dtype == np.complex64
    ku = next((ROOT / "Drone_RF_Dataset_2024").rglob("*.mat"))
    compound = read_hdf5_compound_complex(ku, count=128)
    assert compound.shape == (128,) and compound.dtype == np.complex64


@pytest.mark.skipif(ROOT is None or not ROOT.is_dir(), reason="SCARS_LOCAL_DATASET_ROOT not configured")
def test_interleaved_adapter_and_real_waveform_rejection():
    rfuav = next((ROOT / "RFUAV_2025").rglob("*.iq"))
    iq = read_interleaved_float32(rfuav, count=128)
    assert iq.shape == (128,) and iq.dtype == np.complex64
    cardrf = next((ROOT / "CARDRF").rglob("*.mat"))
    with pytest.raises(ValueError, match="Real oscilloscope waveform rejected"):
        assert_real_waveform_rejected(cardrf)
