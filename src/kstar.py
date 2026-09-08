"""k* selection: validation only, by construction.

    k*(tau) = min{ k : kappa_k >= tau * kappa_full }

The headline number of the paper is an integer chosen by scanning a curve. If
that scan runs over the test split, the number was selected using the data it
is reported on -- test-set selection wearing a definition as a disguise, and
the most likely single reason this study would not survive review.

So the rule is enforced by exception rather than by discipline:

* `select_kstar` raises `TestSetSelectionError` if handed a single test row.
* `test_report` takes k* as an argument. It has no search over budgets, so it
  cannot choose one.
* A missing full-montage reference raises rather than falling back to whatever
  budget happens to be present, because a quietly widened search is worse than
  a crash.

If k* moves across seeds or thresholds, the honest result is the budget curve
with confidence intervals, and `verdict` says so.
"""

import statistics
from typing import Dict, Iterable, List, Optional, Sequence

FULL_MONTAGE_K = 64
DEFAULT_THRESHOLDS = (0.85, 0.90, 0.95)


class TestSetSelectionError(RuntimeError):
    """k* selection was handed test-split data."""

    # pytest collects any Test* class it can see; this is an exception, not a suite.
    __test__ = False


class MissingFullMontageError(RuntimeError):
    """No full-montage reference, so no denominator for the ratio."""


def _eligible(rows: Iterable[Dict], metric: str) -> List[Dict]:
    """Ranked, scratch-trained, non-shuffled rows -- the budget curve itself."""
    out = []
    for r in rows:
        if r.get("selection") != "ranked":
            continue
        if r.get("training", "scratch") != "scratch":
            continue
        if r.get("shuffle_labels"):
            continue
        if metric not in r:
            continue
        out.append(r)
    return out


def _assert_validation_only(rows: Sequence[Dict]) -> None:
    offenders = sorted({r.get("split") for r in rows if r.get("split") != "val"})
    if offenders:
        raise TestSetSelectionError(
            "k* must be selected on validation. Received rows with split(s): {}. "
            "Evaluate test once, at the already-chosen k*, via test_report().".format(
                ", ".join(str(o) for o in offenders)
            )
        )


def budget_curve(
    rows: Sequence[Dict], metric: str = "kappa"
) -> Dict[int, Dict[str, float]]:
    """Mean, std and n per budget, aggregated over training seeds."""
    by_k: Dict[int, List[float]] = {}
    for r in _eligible(rows, metric):
        by_k.setdefault(int(r["budget_k"]), []).append(float(r[metric]))
    curve = {}
    for k, vals in sorted(by_k.items()):
        curve[k] = {
            "mean": round(statistics.fmean(vals), 6),
            "std": round(statistics.pstdev(vals) if len(vals) > 1 else 0.0, 6),
            "n": len(vals),
            "values": [round(v, 6) for v in vals],
        }
    return curve


def _kstar_from_curve(
    curve: Dict[int, Dict[str, float]], threshold: float, full_k: int
) -> Optional[int]:
    if full_k not in curve:
        raise MissingFullMontageError(
            "No k={} rows, so kappa_full is undefined. Run the full montage "
            "before any ratio is computed; the search is not widened to the "
            "largest available budget.".format(full_k)
        )
    kappa_full = curve[full_k]["mean"]
    if kappa_full <= 0:
        raise MissingFullMontageError(
            "kappa_full = {:.4f} is not positive; the ratio kappa_k / kappa_full "
            "is meaningless. Check the full-montage runs before proceeding.".format(
                kappa_full
            )
        )
    target = threshold * kappa_full
    for k in sorted(curve):
        if curve[k]["mean"] >= target:
            return k
    return None


def select_kstar(
    rows: Sequence[Dict],
    threshold: float = 0.90,
    metric: str = "kappa",
    full_k: int = FULL_MONTAGE_K,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    planned_seeds: Optional[Sequence[int]] = None,
    planned_budgets: Optional[Sequence[int]] = None,
) -> Dict:
    """Choose k* on validation and report how stable that choice is."""
    rows = list(rows)
    _assert_validation_only(rows)

    eligible = _eligible(rows, metric)
    if not eligible:
        raise ValueError(
            "No eligible rows: k* is read from ranked, scratch-trained "
            "validation runs."
        )

    keys = [(int(r["budget_k"]), int(r["train_seed"])) for r in eligible]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate ranked scratch condition; audit the result rows first")
    curve = budget_curve(eligible, metric)
    kstar = _kstar_from_curve(curve, threshold, full_k)

    # Per-seed k*, so a headline integer that is really a seed artifact shows up.
    per_seed: Dict[str, Optional[int]] = {}
    seeds = sorted({int(r["train_seed"]) for r in eligible})
    for s in seeds:
        seed_curve = budget_curve([r for r in eligible if int(r["train_seed"]) == s], metric)
        try:
            per_seed[str(s)] = _kstar_from_curve(seed_curve, threshold, full_k)
        except MissingFullMontageError:
            per_seed[str(s)] = None

    sensitivity = {}
    for t in thresholds:
        sensitivity["{:.2f}".format(t)] = _kstar_from_curve(curve, t, full_k)

    distinct_seed = {v for v in per_seed.values() if v is not None}
    distinct_thresh = {v for v in sensitivity.values() if v is not None}
    stable_seeds = len(distinct_seed) <= 1
    stable_thresh = len(distinct_thresh) <= 1

    if kstar is None:
        verdict = "no budget reached the threshold; report the curve"
    elif stable_seeds and stable_thresh:
        verdict = "stable across seeds and thresholds"
    elif not stable_seeds and not stable_thresh:
        verdict = (
            "unstable across both seeds and thresholds; the budget curve with "
            "confidence intervals is the result, not the integer"
        )
    elif not stable_seeds:
        verdict = (
            "k* varies across training seeds {}; report the curve and the "
            "spread, not the integer alone".format(sorted(distinct_seed))
        )
    else:
        verdict = (
            "k* varies across thresholds {}; report the sensitivity, not the "
            "integer alone".format(sorted(distinct_thresh))
        )

    # How much of the planned sweep the curve rests on. n_seeds counts seeds seen
    # at any budget; the full-montage count is what kappa_full actually averages.
    planned = sorted(int(s) for s in planned_seeds) if planned_seeds else None
    expected_budgets = sorted(set(int(k) for k in (planned_budgets or curve)))
    seeds_at_budget = {k: {int(r["train_seed"]) for r in eligible
                           if int(r["budget_k"]) == k} for k in expected_budgets}
    incomplete = ({str(k): len(seeds_at_budget[k] & set(planned))
                   for k in expected_budgets if seeds_at_budget[k] != set(planned)}
                  if planned else {})
    provisional = bool(incomplete) if planned else None
    if provisional:
        verdict = "provisional: planned ranked runs are missing; complete validation before confirmation"
        stable_seeds = None
    full_seeds = sorted({int(r["train_seed"]) for r in eligible
                         if int(r["budget_k"]) == int(full_k)})

    return {
        "kstar": kstar,
        "threshold": threshold,
        "metric": metric,
        "kappa_full": curve[full_k]["mean"],
        "kappa_full_std": curve[full_k]["std"],
        "n_seeds": len(seeds),
        "seeds": seeds,
        "n_full_montage_runs": curve[full_k]["n"],
        "full_montage_seeds": full_seeds,
        "planned_seeds": planned,
        "planned_budgets": expected_budgets if planned_budgets is not None else None,
        "incomplete_budgets": incomplete,
        "provisional": provisional,
        "budget_curve": curve,
        "kstar_per_seed": per_seed,
        "threshold_sensitivity": sensitivity,
        "stable_across_seeds": stable_seeds,
        "stable_across_thresholds": stable_thresh,
        "verdict": verdict,
        "selected_on": "val",
    }


def test_report(
    rows: Sequence[Dict], kstar: int, metric: str = "kappa", full_k: int = FULL_MONTAGE_K
) -> Dict:
    """Evaluate the test split at an ALREADY-CHOSEN k*.

    k* is a required argument, and no search over budgets happens here. That is
    the point: this function structurally cannot pick the best-looking budget.
    """
    if kstar is None:
        raise ValueError("kstar is required; choose it on validation first")

    test_rows = [r for r in rows if r.get("split") == "test"]
    if not test_rows:
        raise ValueError("No test rows supplied")

    eligible = _eligible(test_rows, metric)
    at_k = [r for r in eligible if int(r["budget_k"]) == int(kstar)]
    at_full = [r for r in eligible if int(r["budget_k"]) == int(full_k)]

    if not at_k:
        raise ValueError(
            "No test rows at k*={}. Run that single condition on test; do not "
            "substitute a nearby budget.".format(kstar)
        )
    if not at_full:
        raise MissingFullMontageError(
            "No test rows at k={}, so the retained-fraction cannot be "
            "computed on test.".format(full_k)
        )

    vals_k = [float(r[metric]) for r in at_k]
    vals_full = [float(r[metric]) for r in at_full]
    mean_k = statistics.fmean(vals_k)
    mean_full = statistics.fmean(vals_full)

    return {
        "kstar": int(kstar),
        "metric": metric,
        "test_kappa_at_kstar": round(mean_k, 6),
        "test_kappa_full": round(mean_full, 6),
        "retained_fraction": round(mean_k / mean_full, 6) if mean_full else None,
        "n_runs_at_kstar": len(vals_k),
        "n_runs_full": len(vals_full),
        "budgets_searched": None,  # by construction: none
        "note": "k* was chosen on validation; this report evaluates it once on test",
    }


# `test_report` reads naturally at the call site but starts with "test_", so
# pytest would try to collect it as a test wherever it is imported. This is the
# standard opt-out.
test_report.__test__ = False
