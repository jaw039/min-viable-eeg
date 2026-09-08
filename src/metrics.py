"""Pooled and per-subject metrics.

Two aggregations of kappa are always stored side by side:

    pooled          one confusion matrix over all trials in the split
    macro-subject   mean of per-subject kappa, plus its spread

Which of the two belongs in the denominator of k* is a methodological choice,
and it is one that can reasonably be revisited after the sweep. Storing both,
along with the full per-subject vector, makes that revision a re-analysis
rather than a re-run.

`kappa_per_subject` is never averaged away for the same reason: BCI populations
are famously bimodal, and a subject sitting at chance regardless of montage is
a finding, not noise to be smoothed over.
"""

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


def _as_arrays(y_true, y_pred):
    return np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()


def pooled_metrics(y_true: Sequence, y_pred: Sequence) -> Dict[str, float]:
    """Metrics over all trials at once."""
    yt, yp = _as_arrays(y_true, y_pred)
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred have different lengths")
    if len(yt) == 0:
        raise ValueError("No trials to score")
    return {
        "kappa": float(cohen_kappa_score(yt, yp)),
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
    }


def per_subject_kappa(
    y_true: Sequence, y_pred: Sequence, subjects: Sequence
) -> Dict[str, float]:
    """Kappa computed independently within each subject.

    A subject whose predictions are constant, or whose labels are one class,
    yields an undefined kappa; sklearn returns nan there. That is reported as
    0.0 -- no better than chance -- rather than dropped, so the denominator of
    any later average stays the number of subjects actually evaluated.
    """
    yt, yp = _as_arrays(y_true, y_pred)
    subj = np.asarray(subjects).ravel()
    if not (len(yt) == len(yp) == len(subj)):
        raise ValueError("y_true, y_pred and subjects must be the same length")

    out: Dict[str, float] = {}
    for s in sorted(set(subj.tolist())):
        m = subj == s
        with np.errstate(invalid="ignore", divide="ignore"):
            k = cohen_kappa_score(yt[m], yp[m])
        out["S{:03d}".format(int(s))] = 0.0 if not np.isfinite(k) else float(k)
    return out


def evaluate(
    y_true: Sequence, y_pred: Sequence, subjects: Sequence
) -> Dict[str, object]:
    """Both aggregations plus the per-subject vector, ready for a result row."""
    pooled = pooled_metrics(y_true, y_pred)
    per_subject = per_subject_kappa(y_true, y_pred, subjects)
    vals = np.array(list(per_subject.values()), dtype=float)
    return {
        "kappa": round(pooled["kappa"], 6),
        "accuracy": round(pooled["accuracy"], 6),
        "macro_f1": round(pooled["macro_f1"], 6),
        "kappa_macro_subject": round(float(vals.mean()), 6),
        "kappa_macro_subject_std": round(float(vals.std(ddof=0)), 6),
        "kappa_per_subject": {k: round(v, 6) for k, v in per_subject.items()},
        "n_subjects": int(len(per_subject)),
        "aggregation_primary": "pooled",
    }
