"""Tests for src.ranking and src.budget: discriminative-channel recovery,
determinism, train-only leakage guard, and budget selection."""

import numpy as np
import pytest

import json

from src.budget import (
    BUDGETS_PATH,
    RANKING_PATH,
    apply_budget,
    budget_sets,
    reduce_channels,
    top_k_channels,
    write_budgets,
)
from src.ranking import aggregate_ranking, fisher_scores, write_ranking
from src.utils import load_config

CFG = load_config()
CH = ["A", "B", "C", "D", "E", "F"]


def _fake_subject(seed, discriminative=None, n_trials=40, n_ch=6, n_samp=64):
    """Balanced 2-class trials; channels in `discriminative` get a
    class-dependent amplitude {channel: gain_ratio}."""
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n_trials // 2))
    X = rng.normal(0.0, 1.0, size=(n_trials, n_ch, n_samp))
    for ch, gain in (discriminative or {}).items():
        X[y == 1, ch, :] *= gain
    return X.astype(np.float32), y


def test_fisher_recovers_discriminative_channels():
    # Channel 2 strongly discriminative, channel 4 mildly, rest are noise.
    per_subject = [
        fisher_scores(*_fake_subject(seed, discriminative={2: 3.0, 4: 1.5}))
        for seed in range(5)
    ]
    ranked, scores = aggregate_ranking(per_subject, CH)
    assert ranked[0] == "C"
    assert ranked[1] == "E"
    assert list(scores) == sorted(scores, reverse=True)
    assert len(ranked) == len(CH)


def test_ranking_deterministic():
    a = fisher_scores(*_fake_subject(7, discriminative={1: 2.0}))
    b = fisher_scores(*_fake_subject(7, discriminative={1: 2.0}))
    assert np.array_equal(a, b)
    ranked_a, scores_a = aggregate_ranking([a], CH)
    ranked_b, scores_b = aggregate_ranking([b], CH)
    assert ranked_a == ranked_b
    assert np.array_equal(scores_a, scores_b)


def test_val_test_cannot_influence_ranking():
    train = [_fake_subject(s, discriminative={2: 3.0}) for s in range(3)]
    val = _fake_subject(99, discriminative={5: 4.0})

    before = [fisher_scores(X, y) for X, y in train]
    ranked_before, scores_before = aggregate_ranking(before, CH)
    val[0][:] *= 1e6  # mutate val wildly; train-only scores must not move
    after = [fisher_scores(X, y) for X, y in train]
    ranked_after, scores_after = aggregate_ranking(after, CH)
    assert ranked_before == ranked_after
    assert np.array_equal(scores_before, scores_after)

    # Sensitivity: had the val subject been included, the ranking would change.
    with_val = before + [fisher_scores(*_fake_subject(99, discriminative={5: 4.0}))]
    ranked_leak, _ = aggregate_ranking(with_val, CH)
    assert ranked_leak != ranked_before


def test_fisher_rejects_single_class():
    X, y = _fake_subject(0)
    with pytest.raises(ValueError, match="both classes"):
        fisher_scores(X, np.zeros_like(y))


def test_budget_selection_and_original_order():
    X, _ = _fake_subject(3)
    ranked = ["C", "E", "A", "F", "B", "D"]  # ranking order != montage order
    assert top_k_channels(ranked, 3) == ["C", "E", "A"]
    X_red, names = apply_budget(X, CH, ranked, 3, mode="reduce")
    assert X_red.shape == (X.shape[0], 3, X.shape[2])
    assert names == ["A", "C", "E"]  # original montage order preserved
    assert np.array_equal(X_red[:, 1, :], X[:, 2, :])  # "C" is montage index 2


def test_budget_full_k_is_identity():
    X, _ = _fake_subject(4)
    ranked = ["C", "E", "A", "F", "B", "D"]
    X_red, names = apply_budget(X, CH, ranked, len(CH), mode="reduce")
    assert names == CH
    assert np.array_equal(X_red, X)


def test_budget_validation():
    X, _ = _fake_subject(5)
    ranked = list(CH)
    with pytest.raises(ValueError):
        top_k_channels(ranked, 0)
    with pytest.raises(ValueError):
        top_k_channels(ranked, len(CH) + 1)
    with pytest.raises(ValueError):
        reduce_channels(X, CH, ["A", "A"])  # duplicates
    with pytest.raises(ValueError):
        reduce_channels(X, CH, ["Z"])  # unknown channel
    with pytest.raises(NotImplementedError):
        apply_budget(X, CH, ranked, 2, mode="mask")
    with pytest.raises(ValueError):
        apply_budget(X, CH, ranked, 2, mode="bogus")


def test_config_budgets_are_valid():
    budgets = CFG["budgets"]
    assert budgets == sorted(budgets)
    assert all(1 <= k <= 64 for k in budgets)
    assert budgets[-1] == 64  # full montage is always included


def test_write_ranking_refuses_overwrite(tmp_path):
    # Refusal must fire before any data is touched, so this needs no cache.
    out = tmp_path / "channel_ranking.json"
    out.write_text("{}")
    with pytest.raises(SystemExit):
        write_ranking(CFG, out)


def test_budget_sets_are_ranking_prefixes_in_both_orders():
    ranked = ["E", "B", "F", "A", "C", "D"]
    sets = budget_sets(ranked, CH, [1, 2, 4, 6])
    assert list(sets) == ["1", "2", "4", "6"]
    for k, entry in sets.items():
        k = int(k)
        assert entry["k"] == k
        assert entry["ranking_order"] == ranked[:k]  # best-first prefix
        assert set(entry["montage_order"]) == set(ranked[:k])
        # montage_order follows CH, the order reduce_channels emits data in
        assert entry["montage_order"] == [c for c in CH if c in ranked[:k]]
    assert sets["6"]["montage_order"] == CH  # full budget is the identity


def test_write_budgets_refuses_overwrite(tmp_path):
    out = tmp_path / "budgets.json"
    out.write_text("{}")
    with pytest.raises(SystemExit):
        write_budgets(CFG, RANKING_PATH, out)


@pytest.mark.skipif(
    not (BUDGETS_PATH.exists() and RANKING_PATH.exists()),
    reason="committed budgets.json / channel_ranking.json not present",
)
def test_committed_budgets_match_committed_ranking():
    """The frozen budget sets must be exactly what the frozen ranking implies,
    for every budget in config. Guards against the two files drifting apart."""
    with open(RANKING_PATH) as f:
        ranked = json.load(f)["channels"]
    with open(BUDGETS_PATH) as f:
        budgets = json.load(f)
    assert budgets["budgets"] == CFG["budgets"]
    assert budgets["reduction_mode"] == CFG["reduction_mode"]
    for k in CFG["budgets"]:
        entry = budgets["sets"][str(k)]
        assert entry["ranking_order"] == ranked[:k]
        assert sorted(entry["montage_order"]) == sorted(ranked[:k])
        assert len(set(entry["montage_order"])) == k
