"""Tests for src.loader: label remapping, shapes/dtype, run concatenation."""

import mne
import numpy as np
import pytest

from src.loader import load_subject
from src.utils import edf_path, load_config

CFG = load_config()
S001R04 = edf_path(CFG, 1, 4)

pytestmark = pytest.mark.skipif(
    not S001R04.exists(), reason="data missing — run scripts/download_data.py first"
)


def test_label_mapping_matches_annotations():
    # Ground truth straight from the EDF annotations, never touching the
    # loader's event pipeline: chronological T1/T2 with T1->0, T2->1.
    raw = mne.io.read_raw_edf(S001R04, preload=False, verbose="ERROR")
    expected = [
        0 if desc == "T1" else 1
        for desc in raw.annotations.description
        if desc in ("T1", "T2")
    ]
    X, y, _ = load_subject(1, CFG, runs=[4])
    assert y.tolist() == expected
    assert X.shape[0] == len(y)


def test_shapes_and_dtype():
    X, y, ch_names = load_subject(1, CFG)
    assert X.ndim == 3
    assert X.shape[1:] == (64, 641)
    assert X.dtype == np.float32
    assert y.shape == (X.shape[0],)
    assert set(np.unique(y)) <= {0, 1}
    assert len(ch_names) == 64
    assert len(set(ch_names)) == 64
    # Standardized names: "C3", not the raw EDF's "C3.."
    assert "C3" in ch_names
    assert all(not name.endswith(".") for name in ch_names)


def test_runs_concatenate_consistently():
    runs = CFG["dataset"]["runs"]
    per_run = [load_subject(1, CFG, runs=[r]) for r in runs]
    X_all, y_all, names_all = load_subject(1, CFG)

    assert X_all.shape[0] == sum(X.shape[0] for X, _, _ in per_run)
    # Exact equality doubles as a determinism check (filtering is per-file).
    assert np.array_equal(np.concatenate([X for X, _, _ in per_run]), X_all)
    assert np.array_equal(np.concatenate([y for _, y, _ in per_run]), y_all)
    for _, _, names in per_run:
        assert names == names_all
