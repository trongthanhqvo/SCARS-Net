from __future__ import annotations

import numpy as np
import pytest

from scars.data.adapters.synthetic import synthetic_development_corpus
from scars.data.windowing import window_recordings


@pytest.fixture(scope="session")
def recordings():
    return synthetic_development_corpus(
        seed=17, domains=3, labels=2, recordings_per_label_domain=4, samples=1024
    )


@pytest.fixture(scope="session")
def small_windows(recordings):
    return window_recordings(recordings[:4], 256, 256)
