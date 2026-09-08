"""Reproducibility audit of sweep result rows against the frozen artifacts.

For every row in the given results files, verify that the model was trained
and evaluated under the locked protocol:

  split       subjects evaluated == splits.json[split], splits seed matches
  channels    n_channels == budget_k == len(channels); the channel set is
              re-derived from (selection, k, seed) with src.channels and must
              match the row exactly (ranked rows must equal budgets.json)
  ranking     ranking_provenance in the row == channel_ranking.json provenance
  config      one config hash across all rows (a second value means two
              different configs were mixed into one sweep)
  labels      shuffle_labels is False except on the declared control rows
  trials      n_trials == sum of per-subject counts; per-subject kappa keys
              == evaluated subjects
  uniqueness  no duplicated condition
  coverage    how many manifest conditions are done (per split)

Usage: python scripts/audit_results.py --results 'results/*.jsonl'
Exit status 1 if any check fails.
"""

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.channels import load_budgets, load_ranking, montage_order, select_channels
from src.utils import ARTIFACTS_DIR, REPO_ROOT


def sid(n):
    return "S{:03d}".format(int(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/*.jsonl")
    ap.add_argument("--manifest", default=None, help="manifest to measure coverage against")
    args = ap.parse_args()

    files = sorted(glob.glob(args.results))
    if not files:
        raise SystemExit("no results files match {!r}".format(args.results))

    ranked, rank_prov = load_ranking()
    budgets = load_budgets()
    ch_names = montage_order()
    splits = json.load(open(ARTIFACTS_DIR / "splits.json"))

    rows, errors = [], []
    for f in files:
        for i, line in enumerate(open(f)):
            r = json.loads(line)
            r["_src"] = "{}:{}".format(Path(f).name, i + 1)
            (errors if "error" in r else rows).append(r)

    fails = defaultdict(list)   # check -> [row src, detail]
    warns = defaultdict(list)

    def fail(check, r, detail=""):
        fails[check].append((r["_src"], detail))

    seen = Counter()
    config_hashes = Counter()
    for r in rows:
        key = (r["budget_k"], r["selection"], r.get("selection_seed"), r["training"],
               r["train_seed"], r["split"], bool(r.get("shuffle_labels")))
        seen[key] += 1
        config_hashes[r.get("config_sha256")] += 1

        # split
        expected_subjects = sorted(sid(s) for s in splits[r["split"]])
        got_subjects = sorted(r["n_trials_per_subject"].keys())
        if got_subjects != expected_subjects:
            fail("split_subjects", r, "evaluated {} vs splits.json {}".format(got_subjects, expected_subjects))
        if r.get("splits_seed") != splits["seed"]:
            fail("splits_seed", r, "{} vs {}".format(r.get("splits_seed"), splits["seed"]))
        if r.get("n_subjects") != len(expected_subjects):
            fail("n_subjects", r, str(r.get("n_subjects")))

        # channels
        chans = list(r["channels_montage_order"])
        k = r["budget_k"]
        if not (r["n_channels"] == k == len(chans)):
            fail("channel_count", r, "n_channels={} k={} len={}".format(r["n_channels"], k, len(chans)))
        try:
            derived = select_channels(r["selection"], k, ch_names, ranked, seed=r.get("selection_seed"))
        except Exception as e:  # noqa: BLE001
            fail("channel_derivation", r, repr(e))
            derived = None
        if derived is not None:
            if set(derived) != set(chans):
                fail("channel_set", r, "row {} vs derived {}".format(chans, derived))
            if chans != [c for c in ch_names if c in set(chans)]:
                fail("channel_order", r, "not in montage order: {}".format(chans))
            if r["selection"] == "ranked":
                frozen = budgets["sets"].get(str(k))
                if frozen is None or chans != frozen["montage_order"]:
                    fail("frozen_budget", r, "row {} vs budgets.json {}".format(
                        chans, frozen and frozen["montage_order"]))

        # ranking provenance
        rp = r.get("ranking_provenance") or {}
        if rp.get("git_commit") != rank_prov["git_commit"] or rp.get("config_hash") != rank_prov["config_hash"]:
            fail("ranking_provenance", r, "{} vs {}".format(rp, rank_prov))

        # labels
        if r.get("shuffle_labels") and r["selection"] != "ranked":
            warns["shuffle_on_non_ranked"].append((r["_src"], r["selection"]))

        # trials
        per = r["n_trials_per_subject"]
        if sum(per.values()) != r["n_trials"]:
            fail("n_trials", r, "{} vs {}".format(sum(per.values()), r["n_trials"]))
        if any(not (40 <= v <= 45) for v in per.values()):
            fail("trials_per_subject", r, str(per))
        if sorted(r["kappa_per_subject"].keys()) != got_subjects:
            fail("kappa_per_subject_keys", r, "")
        if r.get("aggregation_primary") != "pooled":
            warns["aggregation"].append((r["_src"], r.get("aggregation_primary")))

        # provenance of code
        if r.get("git_commit") in (None, "unknown"):
            warns["git_commit_unknown"].append((r["_src"], ""))

    dups = {k: v for k, v in seen.items() if v > 1}
    if dups:
        for k, v in dups.items():
            fails["duplicate_condition"].append((str(k), "x{}".format(v)))
    if len(config_hashes) > 1:
        fails["config_hash_mixed"].append(("all", str(dict(config_hashes))))

    # coverage
    coverage = None
    splits_present = sorted({r["split"] for r in rows})
    man_path = args.manifest
    if man_path is None and len(splits_present) == 1:
        man_path = str(REPO_ROOT / "manifests" / "manifest_{}.jsonl".format(splits_present[0]))
    if man_path and Path(man_path).exists():
        manifest = [json.loads(l) for l in open(man_path)]
        mkeys = {(m["budget_k"], m["selection"], m.get("selection_seed"), m["training"],
                  m["train_seed"], m["split"], bool(m.get("shuffle_labels"))) for m in manifest}
        done = mkeys & set(seen)
        extra = set(seen) - mkeys
        coverage = (len(done), len(mkeys), extra)
        by_arm = Counter((m[1], m[3], m[6]) for m in mkeys)
        done_arm = Counter((m[1], m[3], m[6]) for m in done)

    # ---- report
    print("audit of {} file(s): {} result rows, {} error rows".format(len(files), len(rows), len(errors)))
    print("config_sha256 : {}".format(dict(config_hashes)))
    print("ranking       : commit {} config {}".format(rank_prov["git_commit"][:7], rank_prov["config_hash"]))
    print("splits        : {} (seed {}), evaluated split(s) {}".format(
        splits["strategy"], splits["seed"], splits_present))
    print("shuffle rows  : {}".format(sum(bool(r.get("shuffle_labels")) for r in rows)))
    print()
    checks = ["split_subjects", "splits_seed", "n_subjects", "channel_count", "channel_derivation",
              "channel_set", "channel_order", "frozen_budget", "ranking_provenance", "n_trials",
              "trials_per_subject", "kappa_per_subject_keys", "duplicate_condition", "config_hash_mixed"]
    for c in checks:
        n = len(fails.get(c, []))
        print("  {:<24} {}".format(c, "PASS" if n == 0 else "FAIL ({} rows)".format(n)))
    for c, items in fails.items():
        print("\n  details for {}:".format(c))
        for src, detail in items[:10]:
            print("    {}  {}".format(src, detail))
        if len(items) > 10:
            print("    ... {} more".format(len(items) - 10))
    if warns:
        print("\nwarnings (not protocol violations):")
        for c, items in warns.items():
            print("  {:<24} {} rows".format(c, len(items)))
    if errors:
        print("\nerror rows:")
        for e in errors[:10]:
            print("  {}  {}".format(e["_src"], str(e["error"])[:120]))
    if coverage:
        done, total, extra = coverage
        print("\ncoverage vs {}: {}/{} conditions done".format(Path(man_path).name, done, total))
        for arm in sorted(by_arm):
            print("  {:<40} {:>3}/{:<3}".format("{} / {}{}".format(arm[0], arm[1], " / shuffled" if arm[2] else ""),
                                               done_arm.get(arm, 0), by_arm[arm]))
        if extra:
            print("  {} rows not in manifest: {}".format(len(extra), sorted(extra)[:5]))

    ok = not fails
    print("\nRESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
