"""Tests for src.loader: label remapping (all runs), shapes/dtype, run
concatenation, epoch-drop alignment, and protocol guards."""

import copy

import mne
import numpy as np
import pytest

import src.loader as loader_mod
from src.loader import load_subject
from src.utils import edf_path, load_config

CFG = load_config()
RUNS = CFG["dataset"]["runs"]
DATA_PRESENT = all(edf_path(CFG, 1, run).exists() for run in RUNS)

# Applied per-test (not module-wide): the protocol-guard tests need no data.
needs_data = pytest.mark.skipif(
    not DATA_PRESENT, reason="data missing — run scripts/download_data.py first"
)


def _annotation_labels(raw):
    # Ground truth straight from EDF annotations, independent of the
    # loader's event pipeline: chronological T1/T2 with T1->0, T2->1.
    return [
        0 if desc == "T1" else 1
        for desc in raw.annotations.description
        if desc in ("T1", "T2")
    ]


@needs_data
@pytest.mark.parametrize("run", RUNS)
def test_label_mapping_matches_annotations(run):
    raw = mne.io.read_raw_edf(edf_path(CFG, 1, run), preload=False, verbose="ERROR")
    expected = _annotation_labels(raw)
    X, y, _ = load_subject(1, CFG, runs=[run])
    assert y.tolist() == expected
    assert X.shape[0] == len(y)
    assert set(y.tolist()) == {0, 1}  # both classes present in every run


@needs_data
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


@needs_data
def test_runs_concatenate_consistently():
    per_run = [load_subject(1, CFG, runs=[r]) for r in RUNS]
    X_all, y_all, names_all = load_subject(1, CFG)

    assert X_all.shape[0] == sum(X.shape[0] for X, _, _ in per_run)
    # Exact equality doubles as a determinism check (filtering is per-file).
    assert np.array_equal(np.concatenate([X for X, _, _ in per_run]), X_all)
    assert np.array_equal(np.concatenate([y for _, y, _ in per_run]), y_all)
    for _, _, names in per_run:
        assert names == names_all


@needs_data
def test_epoch_drop_keeps_labels_aligned(monkeypatch):
    # A recording that ends mid-final-trial (real cases: S034, S037, S041,
    # S064, S072-S074, S076, S102, S104) must drop that epoch AND its label.
    # Simulate by cropping S001R04 so the last cue's 4-s window overruns.
    path = edf_path(CFG, 1, 4)
    real_read = mne.io.read_raw_edf
    raw_full = real_read(path, preload=False, verbose="ERROR")
    expected_full = _annotation_labels(raw_full)
    last_cue = max(
        onset
        for onset, desc in zip(raw_full.annotations.onset, raw_full.annotations.description)
        if desc in ("T1", "T2")
    )

    def cropped_read(fname, **kwargs):
        raw = real_read(fname, **kwargs)
        return raw.crop(tmax=last_cue + 2.0)  # < tmin + 4 s of data remain

    monkeypatch.setattr(loader_mod.mne.io, "read_raw_edf", cropped_read)
    X, y, _ = load_subject(1, CFG, runs=[4])
    # The truncated final epoch is gone, and the surviving labels are still
    # the annotation sequence minus that trial — X/y did not desynchronize.
    assert len(y) == len(expected_full) - 1
    assert y.tolist() == expected_full[:-1]
    assert X.shape[0] == len(y)


@pytest.mark.parametrize("subject", CFG["dataset"]["exclude_subjects"])
def test_excluded_subject_raises(subject):
    with pytest.raises(ValueError, match="excluded"):
        load_subject(subject, CFG)


def test_label_map_mismatch_raises():
    cfg2 = copy.deepcopy(CFG)
    cfg2["dataset"]["label_map"] = {"T1": "right", "T2": "left"}
    with pytest.raises(ValueError, match="label_map"):
        load_subject(1, cfg2, runs=[4])
