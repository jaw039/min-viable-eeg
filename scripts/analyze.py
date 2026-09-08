"""Post-sweep analysis: k*, ranked-vs-random, arms, coverage.

    python scripts/analyze.py --results results/*.jsonl
    python scripts/analyze.py --results results/*.jsonl --emit-followup
    python scripts/analyze.py --results results/*.jsonl --test-report

Writes results/kstar_report.json. The negative control is checked first: if the
label-shuffle floor is not near zero, nothing else in the sweep is
interpretable and the script says so before printing any headline.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import (
    arm_comparison,
    coverage_report,
    distillation_effect,
    negative_control,
    ranked_vs_random,
    subject_heterogeneity,
)
from src.kstar import MissingFullMontageError, select_kstar, test_report
from src.manifest import condition_key
from src.utils import REPO_ROOT, load_config


def read_rows(patterns):
    rows = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "kstar_report.json")
    p.add_argument("--test-report", action="store_true",
                   help="evaluate test once at the k* chosen on validation")
    p.add_argument("--emit-followup", action="store_true",
                   help="print the conditions worth running next")
    return p


def main() -> None:
    args = build_parser().parse_args()

    cfg = load_config()
    threshold = args.threshold or float(cfg["eval"]["threshold"])

    rows = read_rows(args.results)
    ok = [r for r in rows if "error" not in r]
    errs = [r for r in rows if "error" in r]
    print("{} rows ({} errors)\n".format(len(rows), len(errs)))
    if not ok:
        raise SystemExit("No successful rows to analyse.")

    val = [r for r in ok if r.get("split") == "val"]
    test = [r for r in ok if r.get("split") == "test"]

    # --- negative control first -------------------------------------------
    neg = negative_control(ok)
    print("NEGATIVE CONTROL")
    if not neg["present"]:
        print("  no label-shuffle rows -- run one before trusting anything else")
    else:
        print("  shuffled-label kappa = {:.4f} over {} run(s): {}".format(
            neg["mean_kappa"], neg["n"], "PASS" if neg["passes"] else "FAIL"))
        if not neg["passes"]:
            print("  " + neg["note"])
            print("\nStopping: fix leakage before reading any other number.")
            raise SystemExit(1)
    print()

    # --- k* ----------------------------------------------------------------
    report = {"threshold": threshold, "n_rows": len(ok), "n_errors": len(errs)}
    try:
        ks = select_kstar(val, threshold=threshold,
                          thresholds=tuple(cfg["eval"]["threshold_sensitivity"]),
                          planned_seeds=cfg["sweep"]["train_seeds"])
        report["kstar"] = ks
        print("K* (validation only)")
        print("  kappa_full = {:.4f} +/- {:.4f} over {} of {} planned full-montage runs{}".format(
            ks["kappa_full"], ks["kappa_full_std"], ks["n_full_montage_runs"],
            len(ks["planned_seeds"] or []),
            "  <-- PROVISIONAL: sweep incomplete" if ks["provisional"] else ""))
        if ks["incomplete_budgets"]:
            print("  budgets short of {} planned runs: {}".format(
                len(ks["planned_seeds"]), ks["incomplete_budgets"]))
        print("  k*(tau={:.2f}) = {}".format(threshold, ks["kstar"]))
        print("  per seed        : {}".format(ks["kstar_per_seed"]))
        print("  threshold sweep : {}".format(ks["threshold_sensitivity"]))
        print("  verdict         : {}".format(ks["verdict"]))
        print("\n  budget curve")
        for k, c in ks["budget_curve"].items():
            print("    k={:<3} kappa={:.4f} +/- {:.4f}  (n={})".format(
                k, c["mean"], c["std"], c["n"]))
    except (ValueError, MissingFullMontageError) as exc:
        report["kstar_error"] = str(exc)
        print("K*: not computable yet -- {}".format(exc))
    print()

    # --- ranked vs random ---------------------------------------------------
    comparisons = {}
    budgets = sorted({int(r["budget_k"]) for r in val})
    print("RANKED vs RANDOM (validation)")
    for k in budgets:
        try:
            c = ranked_vs_random(val, k)
        except ValueError:
            continue
        comparisons[str(k)] = c
        flag = "  <-- AT RESOLUTION FLOOR" if c["p_is_at_resolution_floor"] else ""
        prov = ("  (provisional: {}-{} seeds per subset)".format(*c["train_seeds_per_subset"])
                if c.get("provisional") else "")
        print("  k={:<3} ranked={:.4f}  random={:.4f}+/-{:.4f} over {} subsets  p={:.4f} (floor {:.4f}){}{}".format(
            k, c["ranked_mean"], c["random_mean"], c["random_std"], c["random_n"],
            c["empirical_p"], c["resolution_floor"], flag, prov))
    report["ranked_vs_random"] = comparisons
    print()

    # --- arms ---------------------------------------------------------------
    arms = {}
    print("RANKED vs SENSORIMOTOR (validation)")
    for k in budgets:
        a = arm_comparison(val, k)
        if a["sensorimotor"]["n"] == 0:
            continue
        arms[str(k)] = a
        print("  k={:<3} ranked={} sensorimotor={}  {}".format(
            k, a["ranked"]["mean"], a["sensorimotor"]["mean"], a["reading"]))
    report["arm_comparison"] = arms
    print()

    # --- distillation -------------------------------------------------------
    dist = {}
    for k in budgets:
        d = distillation_effect(val, k)
        if d.get("n_distill"):
            dist[str(k)] = d
    if dist:
        print("DISTILLATION")
        for k, d in dist.items():
            print("  k={:<3} {}".format(k, d["reading"]))
        print()
    report["distillation"] = dist

    # --- heterogeneity ------------------------------------------------------
    het = {}
    for k in budgets:
        h = subject_heterogeneity(val, k)
        if h.get("available"):
            het[str(k)] = h
    report["subject_heterogeneity"] = het
    if het:
        k0 = sorted(het, key=lambda s: int(s))[-1]
        h = het[k0]
        print("SUBJECT HETEROGENEITY (k={}, ranked scratch runs only)".format(k0))
        print("  mean={:.4f} median={:.4f} range=[{:.4f}, {:.4f}] over {} subjects".format(
            h["mean"], h["median"], h["min"], h["max"], h["n_subjects"]))
        print("  {}".format(h["note"]))
        print()

    # --- coverage -----------------------------------------------------------
    manifest = REPO_ROOT / "manifests" / "manifest_val.jsonl"
    if manifest.exists():
        planned = [json.loads(l) for l in manifest.open() if l.strip()]
        cov = coverage_report(planned, ok)
        report["coverage"] = cov
        print("COVERAGE")
        print("  {}/{} planned conditions ran ({} missing, {} errored)".format(
            cov["completed"], cov["planned"], cov["missing"], cov["errored"]))
        if cov["missing_by_kind"]:
            print("  missing by kind: {}".format(cov["missing_by_kind"]))
        print("  {}".format(cov["note"]))
        print()

        if args.emit_followup and cov["missing"]:
            print("FOLLOW-UP: {} conditions still to run".format(cov["missing"]))
            for r in cov["missing_examples"]:
                print("    python scripts/run_sweep.py --budget {} --selection {} "
                      "--training {} --train-seed {} --split {}".format(
                          r["budget_k"], r["selection"], r["training"],
                          r["train_seed"], r["split"]))
            print()

    # --- test, once ---------------------------------------------------------
    if args.test_report:
        ks_val = report.get("kstar", {}).get("kstar")
        if ks_val is None:
            print("TEST REPORT: skipped -- no k* selected on validation yet")
        elif not test:
            print("TEST REPORT: skipped -- no test rows present")
        else:
            tr = test_report(test, ks_val)
            report["test_report"] = tr
            print("TEST REPORT (evaluated once, at the k* chosen on validation)")
            print("  k*={}  test kappa={:.4f}  kappa_full={:.4f}  retained={:.3f}".format(
                tr["kstar"], tr["test_kappa_at_kstar"], tr["test_kappa_full"],
                tr["retained_fraction"]))
            print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote {}".format(args.out))


if __name__ == "__main__":
    main()
