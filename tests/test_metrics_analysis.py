import numpy as np
import pytest

from src.analysis import (
    arm_comparison,
    coverage_report,
    distillation_effect,
    negative_control,
    ranked_vs_random,
    resolution_floor,
    subject_heterogeneity,
)
from src.metrics import evaluate, per_subject_kappa, pooled_metrics


# ------------------------------------------------------------------ metrics


def test_pooled_metrics_perfect_and_chance():
    y = np.array([0, 1, 0, 1])
    assert pooled_metrics(y, y)["kappa"] == pytest.approx(1.0)
    assert pooled_metrics(y, 1 - y)["kappa"] == pytest.approx(-1.0)


def test_per_subject_metrics_are_preserved():
    """Averaging away the subject-level distribution hides exactly the
    bimodality this population is known for."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 1, 0, 1, 0])   # S1 perfect, S2 inverted
    subj = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    out = evaluate(y_true, y_pred, subj)
    assert out["kappa_per_subject"] == {"S001": 1.0, "S002": -1.0}
    assert out["n_subjects"] == 2
    # The mean is 0.0 and describes neither subject -- which is the point.
    assert out["kappa_macro_subject"] == pytest.approx(0.0)
    assert out["kappa_macro_subject_std"] == pytest.approx(1.0)


def test_undefined_subject_kappa_becomes_zero_not_dropped():
    # Constant predictions make kappa undefined; the subject must still count.
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 0])
    subj = np.array([7, 7, 7, 7])
    out = per_subject_kappa(y_true, y_pred, subj)
    assert out == {"S007": 0.0}


def test_both_aggregations_are_stored():
    y = np.array([0, 1, 0, 1])
    out = evaluate(y, y, np.array([1, 1, 2, 2]))
    assert "kappa" in out and "kappa_macro_subject" in out
    assert out["aggregation_primary"] == "pooled"


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        pooled_metrics([0, 1], [0])
    with pytest.raises(ValueError):
        per_subject_kappa([0, 1], [0, 1], [1])


# ------------------------------------------------------------------ analysis


def row(k, kappa, selection="ranked", split="val", seed=42, sel_seed=None,
        training="scratch", **kw):
    r = {"budget_k": k, "kappa": kappa, "selection": selection, "split": split,
         "train_seed": seed, "selection_seed": sel_seed, "training": training,
         "shuffle_labels": False}
    r.update(kw)
    return r


def test_ranked_vs_random_reports_resolution_floor():
    """p = 0.048 from 20 subsets is the floor of the design, not evidence."""
    rows = [row(8, 0.40)]
    rows += [row(8, 0.10 + i * 0.001, selection="random", sel_seed=i) for i in range(20)]
    out = ranked_vs_random(rows, 8)
    assert out["random_n"] == 20
    assert out["resolution_floor"] == pytest.approx(1 / 21, abs=1e-6)
    assert out["empirical_p"] == pytest.approx(1 / 21, abs=1e-6)
    assert out["p_is_at_resolution_floor"] is True
    assert "resolution of the design" in out["interpretation"]


def test_ranked_vs_random_not_at_floor_when_random_wins_sometimes():
    rows = [row(8, 0.20)]
    rows += [row(8, 0.10 + i * 0.02, selection="random", sel_seed=i) for i in range(20)]
    out = ranked_vs_random(rows, 8)
    assert out["p_is_at_resolution_floor"] is False
    assert out["empirical_p"] > resolution_floor(20)


def test_resolution_floor_shrinks_with_more_subsets():
    assert resolution_floor(20) > resolution_floor(50) > resolution_floor(200)


def test_arm_comparison_flags_non_motor_advantage():
    rows = [row(8, 0.40), row(8, 0.25, selection="sensorimotor")]
    out = arm_comparison(rows, 8)
    assert "non-motor" in out["reading"]


def test_arm_comparison_clean_when_arms_agree():
    rows = [row(8, 0.30), row(8, 0.295, selection="sensorimotor")]
    assert "clean" in arm_comparison(rows, 8)["reading"]


def test_distillation_effect_detects_sham_teacher_gain():
    """Soft targets regularise regardless of what the teacher knows."""
    rows = [
        row(4, 0.20),
        row(4, 0.30, training="distill"),
        row(4, 0.29, training="distill_shuffled_teacher"),
    ]
    out = distillation_effect(rows, 4)
    assert "regularisation" in out["reading"]


def test_distillation_effect_credits_real_transfer():
    rows = [
        row(4, 0.20),
        row(4, 0.30, training="distill"),
        row(4, 0.205, training="distill_shuffled_teacher"),
    ]
    assert "genuine transfer" in distillation_effect(rows, 4)["reading"]


def test_negative_control_flags_leakage():
    good = negative_control([dict(row(64, 0.01), shuffle_labels=True)])
    assert good["passes"] is True
    bad = negative_control([dict(row(64, 0.42), shuffle_labels=True)])
    assert bad["passes"] is False
    assert "LEAKAGE" in bad["note"]


def test_coverage_report_detects_incomplete_sweep():
    """Methods describing the planned sweep while Results come from the one
    that finished."""
    planned = [
        {"budget_k": 4, "selection": "ranked", "selection_seed": None,
         "training": "scratch", "shuffle_labels": False, "train_seed": 42, "split": "val"},
        {"budget_k": 8, "selection": "ranked", "selection_seed": None,
         "training": "scratch", "shuffle_labels": False, "train_seed": 42, "split": "val"},
    ]
    completed = [dict(planned[0], kappa=0.2)]
    cov = coverage_report(planned, completed)
    assert cov["complete"] is False
    assert cov["missing"] == 1
    assert cov["missing_by_kind"] == {"ranked": 1}
    assert "not the one that was" in cov["note"] or "missing" in cov["note"]


def test_coverage_report_clean_when_complete():
    planned = [{"budget_k": 4, "selection": "ranked", "selection_seed": None,
                "training": "scratch", "shuffle_labels": False,
                "train_seed": 42, "split": "val"}]
    cov = coverage_report(planned, [dict(planned[0], kappa=0.2)])
    assert cov["complete"] is True
    assert cov["note"] == "sweep complete"


def test_subject_heterogeneity_surfaces_near_chance_subjects():
    rows = [dict(row(8, 0.3), kappa_per_subject={"S001": 0.60, "S002": 0.01})]
    out = subject_heterogeneity(rows, 8)
    assert out["subjects_near_chance"] == ["S002"]
    assert out["n_subjects"] == 2
    assert "near chance" in out["note"]


# ------------------------------------------------------------------ regressions


def test_ranked_vs_random_counts_subsets_not_training_runs():
    """Repeating a subset at several training seeds draws no new subset.
    Three subsets at two seeds each must give N=3 and a floor of 1/4, not
    N=6 and 1/7."""
    rows = [row(8, 0.40, seed=42), row(8, 0.42, seed=123)]
    for i in range(3):
        for seed in (42, 123):
            rows.append(row(8, 0.10 + 0.01 * i, selection="random", sel_seed=i, seed=seed))
    out = ranked_vs_random(rows, 8)
    assert out["random_n"] == 3
    assert out["random_rows"] == 6
    assert out["resolution_floor"] == pytest.approx(1 / 4)
    assert out["empirical_p"] == pytest.approx(1 / 4)
    assert out["provisional"] is False


def test_ranked_vs_random_compares_subset_means():
    """A subset that beats ranked at one seed and loses at another counts by
    its mean, not twice."""
    rows = [row(8, 0.35, seed=42), row(8, 0.35, seed=123)]
    rows += [row(8, 0.50, selection="random", sel_seed=0, seed=42),
             row(8, 0.10, selection="random", sel_seed=0, seed=123),
             row(8, 0.20, selection="random", sel_seed=1, seed=42),
             row(8, 0.20, selection="random", sel_seed=1, seed=123)]
    out = ranked_vs_random(rows, 8)
    assert out["n_random_at_or_above_ranked"] == 0
    assert out["empirical_p"] == pytest.approx(1 / 3)
    assert out["random_max"] == pytest.approx(0.30)


def test_ranked_vs_random_flags_uneven_seed_coverage_as_provisional():
    rows = [row(8, 0.40, seed=42), row(8, 0.40, seed=123)]
    rows += [row(8, 0.10, selection="random", sel_seed=0, seed=42),
             row(8, 0.10, selection="random", sel_seed=0, seed=123),
             row(8, 0.10, selection="random", sel_seed=1, seed=42)]
    out = ranked_vs_random(rows, 8)
    assert out["provisional"] is True
    assert out["train_seeds_per_subset"] == [1, 2]
    assert "provisional" in out["interpretation"]


def test_subject_heterogeneity_uses_only_the_headline_arm():
    """Distilled students and the label-shuffle control must not be averaged
    into the scratch per-subject mean."""
    rows = [
        dict(row(64, 0.30), kappa_per_subject={"S001": 0.60, "S002": 0.40}),
        dict(row(64, 0.00, shuffle_labels=True), kappa_per_subject={"S001": 0.0, "S002": 0.0}),
        dict(row(64, 0.10, training="distill"), kappa_per_subject={"S001": 0.1, "S002": 0.1}),
    ]
    out = subject_heterogeneity(rows, 64)
    assert out["per_subject_mean_kappa"] == {"S001": 0.6, "S002": 0.4}
    assert out["n_runs"] == 1
    assert out["rows_excluded_other_arms"] == 2
    assert out["training"] == "scratch"


def test_ranked_vs_random_uses_only_training_seeds_shared_by_every_subset():
    """When one subset lacks a seed, the comparison is made on the seeds all of
    them share, and says so, rather than mixing means over different seeds."""
    rows = [row(8, 0.40, seed=42), row(8, 0.10, seed=123)]
    rows += [row(8, 0.30, selection="random", sel_seed=0, seed=42),
             row(8, 0.30, selection="random", sel_seed=0, seed=123),
             row(8, 0.50, selection="random", sel_seed=1, seed=42)]
    out = ranked_vs_random(rows, 8)
    assert out["matched_train_seeds"] == [42]
    assert out["ranked_mean"] == pytest.approx(0.40)      # seed 123 not shared, not used
    assert out["n_random_at_or_above_ranked"] == 1        # subset 1 at 0.50 on seed 42
    assert out["provisional"] is True
