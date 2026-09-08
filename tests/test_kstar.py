import pytest

from src.kstar import (
    MissingFullMontageError,
    TestSetSelectionError,
    budget_curve,
    select_kstar,
    test_report,
)


def row(k, kappa, seed=42, split="val", selection="ranked", training="scratch", **kw):
    r = {
        "budget_k": k, "kappa": kappa, "train_seed": seed, "split": split,
        "selection": selection, "training": training, "shuffle_labels": False,
    }
    r.update(kw)
    return r


def curve_rows(mapping, seed=42, split="val"):
    return [row(k, v, seed=seed, split=split) for k, v in mapping.items()]


# --- the rule the paper depends on -------------------------------------------


def test_kstar_refuses_test_rows():
    """Test-set budget selection wearing a definition as a disguise."""
    rows = curve_rows({4: 0.2, 64: 0.4}, split="test")
    with pytest.raises(TestSetSelectionError, match="selected on validation"):
        select_kstar(rows)


def test_kstar_refuses_mixed_val_and_test():
    rows = curve_rows({4: 0.2, 64: 0.4}) + curve_rows({8: 0.3}, split="test")
    with pytest.raises(TestSetSelectionError):
        select_kstar(rows)


def test_kstar_has_no_silent_fallback_when_full_montage_missing():
    """A missing kappa_full must not quietly widen the search to the largest
    budget that happens to be present."""
    rows = curve_rows({4: 0.2, 8: 0.3, 16: 0.35})  # no k=64
    with pytest.raises(MissingFullMontageError, match="No k=64 rows"):
        select_kstar(rows)


def test_kstar_rejects_nonpositive_full_montage():
    rows = curve_rows({4: 0.0, 64: 0.0})
    with pytest.raises(MissingFullMontageError, match="not positive"):
        select_kstar(rows)


def test_report_on_test_cannot_choose_kstar():
    """test_report takes k* as an argument and has no search over budgets."""
    test_rows = curve_rows({4: 0.10, 8: 0.20, 16: 0.38, 64: 0.40}, split="test")
    rep = test_report(test_rows, kstar=8)
    assert rep["kstar"] == 8
    assert rep["budgets_searched"] is None
    # k=16 scores better on test, but the report is pinned to the given k*.
    assert rep["test_kappa_at_kstar"] == 0.20


def test_report_refuses_absent_kstar_budget():
    test_rows = curve_rows({4: 0.1, 64: 0.4}, split="test")
    with pytest.raises(ValueError, match="No test rows at k\\*=8"):
        test_report(test_rows, kstar=8)


def test_report_requires_a_kstar():
    with pytest.raises(ValueError, match="kstar is required"):
        test_report(curve_rows({64: 0.4}, split="test"), kstar=None)


# --- what the selection actually reports -------------------------------------


def test_kstar_picks_the_smallest_budget_clearing_the_threshold():
    rows = curve_rows({4: 0.10, 8: 0.28, 16: 0.37, 32: 0.39, 64: 0.40})
    out = select_kstar(rows, threshold=0.90)
    assert out["kappa_full"] == 0.40
    assert out["kstar"] == 16          # 0.37 >= 0.36; 0.28 < 0.36
    assert out["selected_on"] == "val"


def test_kstar_is_none_when_nothing_clears():
    rows = curve_rows({4: 0.01, 8: 0.02, 64: 0.40})
    rows = [r for r in rows if r["budget_k"] != 64] + [row(64, 0.40)]
    out = select_kstar(rows, threshold=0.95)
    # only the full montage itself clears 0.95 * kappa_full
    assert out["kstar"] == 64


def test_kstar_per_seed_exposes_instability():
    """A headline integer that is really a seed artifact."""
    rows = (
        curve_rows({4: 0.10, 8: 0.37, 16: 0.38, 64: 0.40}, seed=1)
        + curve_rows({4: 0.10, 8: 0.20, 16: 0.37, 64: 0.40}, seed=2)
    )
    out = select_kstar(rows, threshold=0.90)
    assert out["kstar_per_seed"] == {"1": 8, "2": 16}
    assert out["stable_across_seeds"] is False
    assert "varies across training seeds" in out["verdict"] or "unstable" in out["verdict"]


def test_kstar_stable_case_says_so():
    rows = (
        curve_rows({4: 0.10, 8: 0.38, 64: 0.40}, seed=1)
        + curve_rows({4: 0.10, 8: 0.38, 64: 0.40}, seed=2)
    )
    out = select_kstar(rows, threshold=0.90, thresholds=(0.90,))
    assert out["stable_across_seeds"] is True
    assert out["verdict"].startswith("stable")


def test_threshold_sensitivity_is_reported():
    rows = curve_rows({4: 0.34, 8: 0.37, 16: 0.39, 64: 0.40})
    out = select_kstar(rows, threshold=0.90, thresholds=(0.85, 0.90, 0.95))
    assert out["threshold_sensitivity"]["0.85"] == 4   # 0.34 >= 0.34
    assert out["threshold_sensitivity"]["0.90"] == 8   # 0.37 >= 0.36
    assert out["threshold_sensitivity"]["0.95"] == 16  # 0.39 >= 0.38
    assert out["stable_across_thresholds"] is False


def test_only_ranked_scratch_rows_feed_the_curve():
    rows = curve_rows({4: 0.10, 64: 0.40})
    rows.append(row(4, 0.99, selection="random", selection_seed=0))
    rows.append(row(4, 0.99, training="distill"))
    rows.append(dict(row(4, 0.99), shuffle_labels=True))
    curve = budget_curve(rows)
    assert curve[4]["values"] == [0.1]


def test_empty_input_is_an_error():
    with pytest.raises(ValueError, match="No eligible rows"):
        select_kstar([row(4, 0.2, selection="random", selection_seed=0)])


# --- honest counts on a partial sweep ---------------------------------------


def test_kstar_reports_the_actual_full_montage_run_count():
    """Two of five planned full-montage runs must not be reported as five
    seeds. n_seeds counts seeds seen at any budget; kappa_full averages only
    the k=64 rows."""
    rows = (curve_rows({4: 0.10, 64: 0.40}, seed=42)
            + curve_rows({4: 0.10, 8: 0.30, 64: 0.42}, seed=123)
            + [row(8, 0.30, seed=456)])
    out = select_kstar(rows, threshold=0.90, planned_seeds=[42, 123, 456, 789, 101112])
    assert out["n_full_montage_runs"] == 2
    assert out["full_montage_seeds"] == [42, 123]
    assert out["n_seeds"] == 3
    assert out["provisional"] is True
    assert out["incomplete_budgets"] == {"4": 2, "8": 2, "64": 2}


def test_kstar_is_not_provisional_when_every_budget_has_every_planned_seed():
    rows = []
    for s in (1, 2):
        rows += curve_rows({4: 0.10, 64: 0.40}, seed=s)
    out = select_kstar(rows, planned_seeds=[1, 2])
    assert out["provisional"] is False
    assert out["incomplete_budgets"] == {}
    assert out["n_full_montage_runs"] == 2


def test_kstar_without_planned_seeds_still_counts_full_montage_runs():
    out = select_kstar(curve_rows({4: 0.10, 64: 0.40}))
    assert out["n_full_montage_runs"] == 1
    assert out["provisional"] is None
