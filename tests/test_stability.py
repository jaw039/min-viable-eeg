"""Tests for src.stability: bootstrap top-k frequency and per-subject overlap
against the shared ranking. All on synthetic per-subject score matrices."""

import numpy as np
import pytest

from src.ranking import aggregate_ranking
from src.stability import bootstrap_topk_frequency, per_subject_topk, summarize

CH = ["A", "B", "C", "D", "E", "F"]


def _scores(seed, n_subj=20):
    """Per-subject scores where B is always best and E always second."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.0, 0.1, size=(n_subj, len(CH)))
    s[:, CH.index("B")] += 1.0
    s[:, CH.index("E")] += 0.5
    return s


def test_bootstrap_dominant_channels_always_selected():
    boot = bootstrap_topk_frequency(_scores(0), CH, [1, 2, 3], n_bootstrap=50, seed=1)
    assert boot[1]["B"] == 1.0
    assert boot[2]["B"] == 1.0 and boot[2]["E"] == 1.0
    # Every resample contributes exactly k memberships.
    for k in (1, 2, 3):
        assert sum(boot[k].values()) == pytest.approx(float(k))
        assert all(0.0 <= v <= 1.0 for v in boot[k].values())


def test_bootstrap_deterministic_given_seed():
    a = bootstrap_topk_frequency(_scores(0), CH, [2, 4], n_bootstrap=30, seed=7)
    b = bootstrap_topk_frequency(_scores(0), CH, [2, 4], n_bootstrap=30, seed=7)
    assert a == b


def test_bootstrap_detects_unstable_channel():
    # A and B have the same mean but noisy per-subject scores -> top-1 is a
    # coin flip between them; the frequencies must reflect that.
    rng = np.random.default_rng(3)
    s = np.zeros((30, len(CH)))
    s[:, CH.index("A")] = rng.normal(1.0, 0.5, size=30)
    s[:, CH.index("B")] = rng.normal(1.0, 0.5, size=30)
    boot = bootstrap_topk_frequency(s, CH, [1], n_bootstrap=200, seed=0)
    assert boot[1]["A"] + boot[1]["B"] == pytest.approx(1.0)
    assert 0.15 < boot[1]["A"] < 0.85
    assert boot[1]["C"] == 0.0


def test_per_subject_frequency_and_overlap():
    scores = _scores(0)
    shared, _ = aggregate_ranking(list(scores), CH)
    ps = per_subject_topk(scores, CH, [1, 2], shared)
    assert ps[1]["frequency"]["B"] == 1.0
    assert ps[1]["overlap"]["mean"] == 1.0
    assert ps[2]["overlap"]["mean"] == 1.0  # {B, E} for every subject
    assert len(ps[2]["overlap"]["per_subject"]) == scores.shape[0]
    assert sum(ps[2]["frequency"].values()) == pytest.approx(2.0)


def test_per_subject_overlap_reflects_disagreement():
    # Subjects 0 and 1 prefer A; subject 2 prefers C so strongly that the
    # shared top-1 is C. Personal top-1s are A, A, C -> overlap mean 1/3.
    s = np.zeros((3, len(CH)))
    s[0, CH.index("A")] = 1.0
    s[1, CH.index("A")] = 1.0
    s[2, CH.index("C")] = 10.0
    shared, _ = aggregate_ranking(list(s), CH)
    assert shared[0] == "C"
    ps = per_subject_topk(s, CH, [1], shared)
    assert ps[1]["overlap"]["per_subject"] == [0.0, 0.0, 1.0]
    assert ps[1]["overlap"]["mean"] == pytest.approx(1.0 / 3.0)
    assert ps[1]["frequency"]["A"] == pytest.approx(2.0 / 3.0)


def test_validation():
    s = _scores(0)
    with pytest.raises(ValueError):
        bootstrap_topk_frequency(s, CH, [2], n_bootstrap=0, seed=0)
    with pytest.raises(ValueError):
        bootstrap_topk_frequency(s[:, :4], CH, [2], n_bootstrap=5, seed=0)
    with pytest.raises(ValueError):
        per_subject_topk(s, CH, [2], ["A", "B"])  # not a permutation of CH


def test_summarize_lists_frozen_set():
    scores = _scores(0)
    shared, _ = aggregate_ranking(list(scores), CH)
    boot = bootstrap_topk_frequency(scores, CH, [2], n_bootstrap=10, seed=0)
    ps = per_subject_topk(scores, CH, [2], shared)
    text = summarize(shared, boot, ps, [2])
    assert "k= 2" in text
    assert "B" in text and "E" in text


def test_write_stability_refuses_to_overwrite(tmp_path):
    """The frozen artifact must not be regenerated in place, like the other
    three writers. The guard runs before any data is read."""
    from src.stability import write_stability

    out = tmp_path / "stability.json"
    out.write_text("{}")
    with pytest.raises(SystemExit):
        write_stability({}, out_path=out)
    assert out.read_text() == "{}"
