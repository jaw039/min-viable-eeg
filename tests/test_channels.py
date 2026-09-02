import json

import pytest

from src.channels import (
    SENSORIMOTOR_POOL,
    ChannelDriftError,
    assert_matches_frozen_budgets,
    load_budgets,
    load_ranking,
    max_k,
    montage_order,
    select_channels,
    selection_for_run,
    sensorimotor_channels,
)


@pytest.fixture(scope="module")
def ranked():
    return load_ranking()[0]


@pytest.fixture(scope="module")
def ch_names():
    return montage_order()


def test_ranked_selection_matches_frozen_budgets_at_every_k(ch_names, ranked):
    """Prevents training on a channel set that disagrees with the frozen
    artifact -- the headline claim would describe a montage never trained."""
    sets = load_budgets()["sets"]
    for k_str, entry in sets.items():
        got = select_channels("ranked", int(k_str), ch_names, ranked)
        assert got == entry["ranking_order"], "drift at k={}".format(k_str)


def test_drift_guard_fires_on_mismatch(ch_names, ranked):
    good = select_channels("ranked", 8, ch_names, ranked)
    assert_matches_frozen_budgets(8, good)  # does not raise
    with pytest.raises(ChannelDriftError, match="disagrees with frozen"):
        assert_matches_frozen_budgets(8, list(reversed(good)))


def test_drift_guard_rejects_unknown_budget(ch_names, ranked):
    with pytest.raises(ChannelDriftError, match="no frozen set"):
        assert_matches_frozen_budgets(7, ["C4"])


def test_selection_for_run_guards_ranked_only(ch_names, ranked):
    # A random subset is not expected to match budgets.json, so the guard
    # must not fire for it.
    selection_for_run("random", 8, ch_names, ranked, seed=1)
    selection_for_run("sensorimotor", 8, ch_names, ranked)
    selection_for_run("ranked", 8, ch_names, ranked)


def test_sensorimotor_selection_stays_inside_declared_pool(ch_names, ranked):
    """A 'restricted' control that isn't restricted would silently answer a
    different question than the one the paper asks."""
    for k in (4, 6, 8, 12, 16):
        sel = select_channels("sensorimotor", k, ch_names, ranked)
        assert len(sel) == k
        assert set(sel) <= set(SENSORIMOTOR_POOL)


def test_sensorimotor_excludes_the_non_motor_channels(ch_names, ranked):
    """F7/AF7/PO7/O1 outrank C3 in the unrestricted ranking; separating them
    from motor cortex is the entire point of this arm."""
    sel = select_channels("sensorimotor", max_k("sensorimotor", ch_names), ch_names, ranked)
    for ch in ("F7", "AF7", "PO7", "O1", "Iz", "Oz"):
        assert ch not in sel


def test_sensorimotor_declared_pool_is_present_in_montage(ch_names):
    present = sensorimotor_channels(ch_names)
    assert len(present) == len(SENSORIMOTOR_POOL)
    assert set(present) == set(SENSORIMOTOR_POOL)


def test_random_is_seed_deterministic(ch_names, ranked):
    a = select_channels("random", 8, ch_names, ranked, seed=3)
    b = select_channels("random", 8, ch_names, ranked, seed=3)
    c = select_channels("random", 8, ch_names, ranked, seed=4)
    assert a == b and a != c
    assert len(set(a)) == 8


def test_random_requires_explicit_seed(ch_names, ranked):
    with pytest.raises(ValueError, match="requires an explicit seed"):
        select_channels("random", 4, ch_names, ranked)


def test_budget_bounds_are_enforced(ch_names, ranked):
    with pytest.raises(ValueError, match="outside"):
        select_channels("ranked", 0, ch_names, ranked)
    with pytest.raises(ValueError, match="outside"):
        select_channels("ranked", 65, ch_names, ranked)
    with pytest.raises(ValueError, match="outside"):
        select_channels("sensorimotor", 64, ch_names, ranked)


def test_unknown_mode_rejected(ch_names, ranked):
    with pytest.raises(ValueError, match="Unknown selection mode"):
        select_channels("greedy", 4, ch_names, ranked)


def test_full_montage_is_the_whole_montage(ch_names, ranked):
    assert sorted(select_channels("ranked", 64, ch_names, ranked)) == sorted(ch_names)
