"""Tests for src.splits and src.normalize: coverage, determinism, and
train-only normalization stats (leakage guard per CLAUDE.md)."""

import copy
import json

import numpy as np
import pytest

from src.normalize import apply_stats, fit_stats
from src.splits import make_splits, write_splits
from src.utils import N_SUBJECTS, load_config

CFG = load_config()
EXCLUDED = set(CFG["dataset"]["exclude_subjects"])


# ---------------------------------------------------------------- splits

def test_splits_disjoint_and_cover_all_included():
    splits = make_splits(CFG)
    train, val, test = (set(splits[k]) for k in ("train", "val", "test"))
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    included = set(range(1, N_SUBJECTS + 1)) - EXCLUDED
    assert len(included) == 106
    assert train | val | test == included


def test_excluded_subjects_appear_nowhere():
    splits = make_splits(CFG)
    for name in ("train", "val", "test"):
        assert EXCLUDED.isdisjoint(splits[name])


def test_same_seed_identical_different_seed_not():
    assert make_splits(CFG) == make_splits(copy.deepcopy(CFG))
    cfg2 = copy.deepcopy(CFG)
    cfg2["splits"]["seed"] = CFG["splits"]["seed"] + 1
    assert make_splits(cfg2) != make_splits(CFG)


def test_split_sizes_follow_ratios():
    splits = make_splits(CFG)
    sizes = (len(splits["train"]), len(splits["val"]), len(splits["test"]))
    assert sizes == (74, 16, 16)  # round(106 * [0.7, 0.15, 0.15]), remainder to test


def test_write_splits_refuses_overwrite(tmp_path):
    out = tmp_path / "splits.json"
    write_splits(CFG, out)
    payload = json.loads(out.read_text())
    assert payload["train"] == make_splits(CFG)["train"]
    assert "git_commit" in payload["provenance"]
    with pytest.raises(SystemExit):
        write_splits(CFG, out)


# ----------------------------------------------------------- normalization

def _fake_data(seed, n_trials=8, n_ch=4, n_samp=16, loc=0.0, scale=1.0):
    rng = np.random.default_rng(seed)
    return rng.normal(loc, scale, size=(n_trials, n_ch, n_samp)).astype(np.float32)


def test_fit_stats_matches_manual_per_channel():
    X = _fake_data(0, loc=3.0, scale=2.0)
    mu, sd = fit_stats(X)
    assert mu.shape == (4,)
    assert sd.shape == (4,)
    X64 = X.astype(np.float64)
    np.testing.assert_allclose(mu, X64.mean(axis=(0, 2)))
    np.testing.assert_allclose(sd, X64.std(axis=(0, 2)))


def test_val_test_cannot_influence_stats():
    X_train = _fake_data(1)
    X_val = _fake_data(2, loc=50.0, scale=9.0)
    X_test = _fake_data(3, loc=-20.0, scale=4.0)

    mu_before, sd_before = fit_stats(X_train)
    # Mutate val/test wildly; refitting on the SAME train array must be
    # bit-identical — fit_stats has no path to see anything but its input.
    X_val *= 1e6
    X_test += 1e6
    mu_after, sd_after = fit_stats(X_train)
    assert np.array_equal(mu_before, mu_after)
    assert np.array_equal(sd_before, sd_after)

    # And fit_stats IS sensitive to its input: had val data been included,
    # the stats would differ — so identical output proves val was unused.
    mu_leak, _ = fit_stats(np.concatenate([X_train, X_val]))
    assert not np.allclose(mu_leak, mu_before)


def test_apply_uses_train_stats_only():
    X_train = _fake_data(4, loc=2.0, scale=3.0)
    X_val = _fake_data(5, loc=-7.0, scale=0.5)
    mu, sd = fit_stats(X_train)

    Z_train = apply_stats(X_train, mu, sd)
    assert Z_train.dtype == np.float32
    assert Z_train.shape == X_train.shape
    Z64 = Z_train.astype(np.float64)
    np.testing.assert_allclose(Z64.mean(axis=(0, 2)), 0.0, atol=1e-6)
    np.testing.assert_allclose(Z64.std(axis=(0, 2)), 1.0, atol=1e-6)

    # Val normalized with TRAIN stats stays off-center by ~(mu_val - mu_train)/sd_train,
    # which is exactly what train-only stats should do.
    Z_val = apply_stats(X_val, mu, sd)
    assert np.abs(Z_val.astype(np.float64).mean(axis=(0, 2))).min() > 1.0


def test_fit_stats_rejects_bad_input():
    with pytest.raises(ValueError):
        fit_stats(np.zeros((4, 8)))  # not 3-D
    with pytest.raises(ValueError):
        fit_stats(np.zeros((4, 2, 8)))  # dead (zero-std) channels
    X = _fake_data(6)
    mu, sd = fit_stats(X)
    with pytest.raises(ValueError):
        apply_stats(X, mu[:2], sd[:2])  # stats/channel mismatch
