"""Post-sweep analysis: ranked vs random, arm comparisons, coverage.

The statistical point this module exists to protect: with N random subsets, the
smallest one-sided empirical p-value obtainable is 1/(N+1). With the default 20
subsets that floor is 0.048 -- just under the conventional 0.05. A result
reported as "p = 0.048, significant" when 0.048 is the smallest number the
design can produce is reading the resolution of the test, not evidence from it.
So every comparison reports the floor alongside the p-value and flags when the
two coincide.
"""

import statistics
from typing import Dict, Iterable, List, Optional, Sequence

from src.manifest import condition_key, missing_conditions


def _rows(rows, **eq):
    out = []
    for r in rows:
        if all(r.get(key) == val for key, val in eq.items()):
            out.append(r)
    return out


def resolution_floor(n: int) -> float:
    """Smallest one-sided empirical p-value obtainable from n draws."""
    return 1.0 / (n + 1)


def ranked_vs_random(
    rows: Sequence[Dict], budget_k: int, split: str = "val", metric: str = "kappa"
) -> Dict:
    """Compare the ranked selection against the random-subset distribution.

    The unit of the comparison is the electrode subset. Training the same
    subset at several seeds draws no new subset, so each subset is first
    averaged over its training seeds and N is the number of subsets. Counting
    rows instead would report a floor of 1/101 for a design that can only
    resolve 1/21.
    """
    ranked_rows = [
        r for r in _rows(rows, selection="ranked", split=split, budget_k=budget_k)
        if r.get("training", "scratch") == "scratch" and not r.get("shuffle_labels")
    ]
    random_rows = [
        r for r in _rows(rows, selection="random", split=split, budget_k=budget_k)
        if r.get("training", "scratch") == "scratch" and not r.get("shuffle_labels")
    ]
    if not ranked_rows:
        raise ValueError("No ranked rows at k={} on {}".format(budget_k, split))
    if not random_rows:
        raise ValueError("No random rows at k={} on {}".format(budget_k, split))

    def scores_by_seed(run_rows):
        scores = {}
        for r in run_rows:
            seed = int(r["train_seed"])
            if seed in scores:
                raise ValueError("duplicate training seed {} in one condition".format(seed))
            scores[seed] = float(r[metric])
        return scores

    ranked_scores = scores_by_seed(ranked_rows)
    by_subset: Dict[int, List[Dict]] = {}
    for r in random_rows:
        if r.get("selection_seed") is None:
            raise ValueError(
                "random row without a selection_seed cannot be attributed to a subset"
            )
        by_subset.setdefault(int(r["selection_seed"]), []).append(r)
    subset_scores = {s: scores_by_seed(rs) for s, rs in sorted(by_subset.items())}
    common_seeds = sorted(set(ranked_scores).intersection(
        *(set(scores) for scores in subset_scores.values())))
    if not common_seeds:
        raise ValueError("No common training seeds across ranked and random subsets; "
                         "wait for matching runs before comparing them")
    subset_means = [statistics.fmean(scores[s] for s in common_seeds)
                    for scores in subset_scores.values()]
    seeds_per_subset = [len(scores) for scores in subset_scores.values()]

    ranked_vals = [ranked_scores[s] for s in common_seeds]
    obs = statistics.fmean(ranked_vals)
    n = len(subset_means)
    n_ge = sum(1 for v in subset_means if v >= obs)
    p = (n_ge + 1) / (n + 1)
    floor = resolution_floor(n)
    at_floor = abs(p - floor) < 1e-12
    # Uneven seed coverage means the sweep is still running: report, do not hide.
    provisional = (
        any(set(scores) != set(common_seeds) for scores in subset_scores.values())
        or set(ranked_scores) != set(common_seeds)
    )

    interpretation = (
        "p equals the smallest value {} random subsets can produce; this is "
        "the resolution of the design, not evidence of an effect. Increase "
        "the subset count to resolve further.".format(n)
        if at_floor
        else "p = {:.4f} against a floor of {:.4f}".format(p, floor)
    )
    if provisional:
        interpretation += (
            "; provisional: using only common training seeds {}; subsets have "
            "{}-{} runs and ranked has {}".format(
                common_seeds, min(seeds_per_subset), max(seeds_per_subset), len(ranked_rows))
        )

    return {
        "budget_k": int(budget_k),
        "split": split,
        "metric": metric,
        "unit": "subset mean over training seeds",
        "ranked_mean": round(obs, 6),
        "ranked_n": len(ranked_vals),
        "matched_train_seeds": common_seeds,
        "random_mean": round(statistics.fmean(subset_means), 6),
        "random_std": round(statistics.pstdev(subset_means) if n > 1 else 0.0, 6),
        "random_min": round(min(subset_means), 6),
        "random_max": round(max(subset_means), 6),
        "random_n": n,
        "random_rows": len(random_rows),
        "train_seeds_per_subset": [min(seeds_per_subset), max(seeds_per_subset)],
        "n_random_at_or_above_ranked": n_ge,
        "empirical_p": round(p, 6),
        "resolution_floor": round(floor, 6),
        "p_is_at_resolution_floor": at_floor,
        "provisional": provisional,
        "interpretation": interpretation,
    }


def arm_comparison(
    rows: Sequence[Dict], budget_k: int, split: str = "val", metric: str = "kappa"
) -> Dict:
    """Ranked vs sensorimotor at equal k: motor ERD, or cue-correlated?"""
    out: Dict[str, object] = {"budget_k": int(budget_k), "split": split}
    for arm in ("ranked", "sensorimotor"):
        vals = [
            float(r[metric])
            for r in _rows(rows, selection=arm, split=split, budget_k=budget_k)
            if r.get("training", "scratch") == "scratch" and not r.get("shuffle_labels")
        ]
        out[arm] = {
            "mean": round(statistics.fmean(vals), 6) if vals else None,
            "std": round(statistics.pstdev(vals), 6) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
    r_mean = out["ranked"]["mean"]
    s_mean = out["sensorimotor"]["mean"]
    if r_mean is None or s_mean is None:
        out["reading"] = "incomplete: both arms need runs at this budget"
    elif abs(r_mean - s_mean) < 0.02:
        out["reading"] = (
            "arms agree within 0.02 kappa; the signal is consistent with motor "
            "ERD and the montage recommendation is clean"
        )
    elif r_mean > s_mean:
        out["reading"] = (
            "unrestricted beats the motor strip by {:.3f} kappa; part of the "
            "discriminative signal may be non-motor (gaze/attention correlates "
            "of the cue) -- a finding about the benchmark, not a defect".format(
                r_mean - s_mean
            )
        )
    else:
        out["reading"] = (
            "the motor strip beats the unrestricted ranking by {:.3f} kappa; the "
            "unrestricted ranking is picking up channels that do not "
            "generalise".format(s_mean - r_mean)
        )
    return out


def distillation_effect(
    rows: Sequence[Dict], budget_k: int, split: str = "val", metric: str = "kappa"
) -> Dict:
    """Scratch vs distilled vs distilled-from-a-teacher-that-knows-nothing.

    Soft targets regularise regardless of whether the teacher knows anything, so
    a gain over scratch only supports "knowledge transfer" if it does NOT also
    appear against the shuffled-label teacher.
    """
    def mean_of(training):
        vals = [
            float(r[metric])
            for r in _rows(rows, selection="ranked", split=split, budget_k=budget_k)
            if r.get("training", "scratch") == training and not r.get("shuffle_labels")
        ]
        return (round(statistics.fmean(vals), 6) if vals else None), len(vals)

    scratch, n_s = mean_of("scratch")
    distil, n_d = mean_of("distill")
    shuffled, n_x = mean_of("distill_shuffled_teacher")

    out = {
        "budget_k": int(budget_k),
        "split": split,
        "scratch": scratch, "n_scratch": n_s,
        "distill": distil, "n_distill": n_d,
        "distill_shuffled_teacher": shuffled, "n_shuffled": n_x,
    }
    if None in (scratch, distil):
        out["reading"] = "incomplete: needs scratch and distill runs at this budget"
        return out

    gain = distil - scratch
    out["gain_over_scratch"] = round(gain, 6)
    if shuffled is None:
        out["reading"] = (
            "gain of {:.3f} recorded, but without the shuffled-teacher control "
            "it cannot be attributed to knowledge transfer".format(gain)
        )
        return out

    sham = shuffled - scratch
    out["gain_from_sham_teacher"] = round(sham, 6)
    if gain <= 0:
        out["reading"] = "distillation did not help at this budget"
    elif sham >= gain * 0.5:
        out["reading"] = (
            "a teacher trained on shuffled labels recovers {:.0f}% of the gain; "
            "the effect is largely soft-target regularisation, not knowledge "
            "transfer".format(100 * sham / gain)
        )
    else:
        out["reading"] = (
            "gain {:.3f} over scratch, while the sham teacher gives only "
            "{:.3f}; consistent with genuine transfer".format(gain, sham)
        )
    return out


def negative_control(rows: Sequence[Dict], metric: str = "kappa") -> Dict:
    """The label-shuffle floor. Anything far from zero means leakage."""
    vals = [float(r[metric]) for r in rows if r.get("shuffle_labels")]
    if not vals:
        return {"present": False, "note": "no label-shuffle rows found"}
    mean = statistics.fmean(vals)
    ok = abs(mean) < 0.05
    return {
        "present": True,
        "mean_kappa": round(mean, 6),
        "n": len(vals),
        "passes": ok,
        "note": (
            "floor is where it should be; no evidence of leakage"
            if ok
            else "LEAKAGE SUSPECTED: shuffled labels should score near zero. "
                 "Nothing else in the sweep is interpretable until this is explained."
        ),
    }


def coverage_report(planned: Sequence[Dict], completed: Sequence[Dict]) -> Dict:
    """What was planned, what ran, and precisely what is missing."""
    missing = missing_conditions(planned, completed)
    errors = [r for r in completed if "error" in r]
    by_kind: Dict[str, int] = {}
    for r in missing:
        key = r["selection"] if not r.get("shuffle_labels") else "label-shuffle"
        if r.get("training", "scratch") != "scratch":
            key = r["training"]
        by_kind[key] = by_kind.get(key, 0) + 1
    n_planned = len(planned)
    n_done = n_planned - len(missing)
    return {
        "planned": n_planned,
        "completed": n_done,
        "missing": len(missing),
        "errored": len(errors),
        "complete": len(missing) == 0,
        "missing_by_kind": by_kind,
        "missing_examples": [
            {k: r[k] for k in ("budget_k", "selection", "training", "train_seed", "split")}
            for r in missing[:10]
        ],
        "note": (
            "sweep complete"
            if not missing
            else "Methods must describe the sweep that ran, not the one that was "
                 "planned. {} conditions are missing.".format(len(missing))
        ),
    }


def subject_heterogeneity(
    rows: Sequence[Dict], budget_k: int, split: str = "val", training: str = "scratch"
) -> Dict:
    """How much the population mean hides.

    52 of 74 training subjects share no electrode with the shared top-4, so the
    question of whether a mean describes anybody is live for this dataset.

    Only the headline arm feeds the summary: ranked, trained as `training`
    (scratch by default), labels intact. Distilled students and the
    label-shuffle control answer different questions and are excluded rather
    than averaged into the same per-subject mean.
    """
    per_subject: Dict[str, List[float]] = {}
    n_runs = excluded = 0
    for r in _rows(rows, selection="ranked", split=split, budget_k=budget_k):
        if r.get("training", "scratch") != training or r.get("shuffle_labels"):
            excluded += 1
            continue
        n_runs += 1
        for s, k in (r.get("kappa_per_subject") or {}).items():
            per_subject.setdefault(s, []).append(float(k))
    if not per_subject:
        return {"available": False, "note": "no per-subject kappa recorded"}
    means = {s: statistics.fmean(v) for s, v in sorted(per_subject.items())}
    vals = sorted(means.values())
    near_chance = [s for s, v in means.items() if v < 0.05]
    return {
        "available": True,
        "budget_k": int(budget_k),
        "split": split,
        "training": training,
        "n_runs": n_runs,
        "rows_excluded_other_arms": excluded,
        "n_subjects": len(means),
        "mean": round(statistics.fmean(vals), 6),
        "median": round(statistics.median(vals), 6),
        "min": round(vals[0], 6),
        "max": round(vals[-1], 6),
        "std": round(statistics.pstdev(vals) if len(vals) > 1 else 0.0, 6),
        "subjects_near_chance": near_chance,
        "n_near_chance": len(near_chance),
        "per_subject_mean_kappa": {s: round(v, 6) for s, v in means.items()},
        "note": (
            "{} of {} subjects sit near chance; report the distribution, not "
            "only the mean".format(len(near_chance), len(means))
            if near_chance
            else "no subject sits near chance at this budget"
        ),
    }
