"""Reproducibility audit of sweep result rows against the frozen artifacts.

For every row in the given results files, verify that the model was trained
and evaluated under the locked protocol and as part of the written-down
experiment:

  experiment  the condition is in manifests/manifest_<split>.jsonl; its
              training seed and selection seed are the declared ones
  split       subjects evaluated == artifacts/splits.json[split], seed matches
  channels    n_channels == budget_k; both channel lists (selection order and
              montage order) are re-derived from the frozen ranking and must
              match exactly; ranked rows must equal artifacts/budgets.json
  provenance  ranking_provenance == channel_ranking.json's; config_sha256 ==
              the hash of the config.yaml in this checkout, and one value
              across all rows
  trials      n_trials == sum of per-subject counts; per-subject kappa keys
              == evaluated subjects
  errors      a failed condition fails the audit unless --allow-errors
  uniqueness  no duplicated condition
  coverage    conditions done per arm; --require-complete makes an
              incomplete sweep a failure

Usage: python scripts/audit_results.py --results 'results/*.jsonl'
Exit status 1 if any check fails.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audit import CHECKS, audit_rows
from src.channels import load_budgets, load_ranking, montage_order
from src.utils import ARTIFACTS_DIR, REPO_ROOT, config_hash, load_config


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/*.jsonl", help="glob of result files")
    ap.add_argument("--manifest", default=None,
                    help="manifest to audit against (default manifests/manifest_<split>.jsonl)")
    ap.add_argument("--allow-errors", action="store_true",
                    help="report error rows as warnings: an in-progress sweep, not a final one")
    ap.add_argument("--require-complete", action="store_true",
                    help="fail while any planned condition is missing")
    return ap


def load_jsonl(path: Path):
    return [json.loads(line) for line in open(path) if line.strip()]


def main() -> None:
    args = build_parser().parse_args()
    files = sorted(glob.glob(args.results))
    if not files:
        raise SystemExit("no results files match {!r}".format(args.results))

    rows, errors = [], []
    for f in files:
        for i, r in enumerate(load_jsonl(Path(f))):
            r["_src"] = "{}:{}".format(Path(f).name, i + 1)
            (errors if "error" in r else rows).append(r)

    cfg = load_config()
    ranked, rank_prov = load_ranking()
    budgets = load_budgets()
    ch_names = montage_order()
    splits = json.load(open(ARTIFACTS_DIR / "splits.json"))

    splits_present = sorted({r["split"] for r in rows} | {e["split"] for e in errors if e.get("split")})
    manifests = {}
    for split in splits_present:
        if args.manifest and len(splits_present) == 1:
            path = Path(args.manifest)
        else:
            path = REPO_ROOT / "manifests" / "manifest_{}.jsonl".format(split)
        if path.exists():
            manifests[split] = load_jsonl(path)

    report = audit_rows(
        rows, errors,
        manifests=manifests,
        expected_config_hash=config_hash(),
        ranked=ranked, rank_prov=rank_prov, budgets=budgets, ch_names=ch_names,
        splits=splits,
        planned_seeds=cfg["sweep"]["train_seeds"],
        n_random=int(cfg["sweep"]["n_random_subsets"]),
        allow_errors=args.allow_errors,
        require_complete=args.require_complete,
    )

    print("audit of {} file(s): {} result rows, {} error rows".format(len(files), len(rows), len(errors)))
    print("config_sha256 : {} (expected {})".format(report["config_hashes"], report["expected_config_hash"]))
    print("ranking       : commit {} config {}".format(rank_prov["git_commit"][:7], rank_prov["config_hash"]))
    print("splits        : {} (seed {}), evaluated split(s) {}".format(
        splits["strategy"], splits["seed"], splits_present))
    print("manifests     : {}".format({s: len(m) for s, m in manifests.items()} or "none found"))
    print("shuffle rows  : {}".format(sum(bool(r.get("shuffle_labels")) for r in rows)))
    print()
    for c in CHECKS:
        n = len(report["fails"].get(c, []))
        print("  {:<24} {}".format(c, "PASS" if n == 0 else "FAIL ({} rows)".format(n)))
    for c, items in report["fails"].items():
        print("\n  details for {}:".format(c))
        for src, detail in items[:10]:
            print("    {}  {}".format(src, detail))
        if len(items) > 10:
            print("    ... {} more".format(len(items) - 10))
    if report["warns"]:
        print("\nwarnings (not protocol violations):")
        for c, items in report["warns"].items():
            print("  {:<24} {} rows".format(c, len(items)))
    for split, cov in report["coverage"].items():
        print("\ncoverage ({}): {}/{} conditions done, {}".format(
            split, cov["done"], cov["planned"], "COMPLETE" if cov["complete"] else "INCOMPLETE"))
        for arm, (done, planned) in cov["by_arm"].items():
            print("  {:<40} {:>3}/{:<3}".format(arm, done, planned))

    print("\nRESULT: {}  (coverage {})".format(
        "PASS" if report["ok"] else "FAIL", "complete" if report["complete"] else "INCOMPLETE"))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
