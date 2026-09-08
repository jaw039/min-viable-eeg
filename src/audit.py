"""Reproducibility audit of result rows against the frozen artifacts and the manifest.

Every check answers one question: was this row produced under the locked
protocol, as part of the experiment that was written down before it ran?
A file passes only if every row passes and nothing failed silently. Partial
coverage is reported explicitly and, on request, is itself a failure, so an
incomplete sweep can be worked with but never mistaken for a complete one.
"""

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence

from src.channels import select_channels

CHECKS = (
    "error_rows",
    "manifest_missing",
    "not_in_manifest",
    "seed_declared",
    "split_subjects",
    "splits_seed",
    "n_subjects",
    "channel_count",
    "channel_derivation",
    "channel_set",
    "channel_order",
    "channels_field",
    "frozen_budget",
    "ranking_provenance",
    "config_hash_expected",
    "config_hash_mixed",
    "n_trials",
    "trials_per_subject",
    "kappa_per_subject_keys",
    "duplicate_condition",
    "incomplete_coverage",
)


def condition_key(r: Dict) -> tuple:
    return (
        int(r["budget_k"]), r["selection"], r.get("selection_seed"),
        r.get("training", "scratch"), int(r["train_seed"]), r["split"],
        bool(r.get("shuffle_labels")),
    )


def sid(n) -> str:
    return "S{:03d}".format(int(n))


def audit_rows(
    rows: Sequence[Dict],
    error_rows: Sequence[Dict],
    *,
    manifests: Dict[str, Sequence[Dict]],
    expected_config_hash: str,
    ranked: Sequence[str],
    rank_prov: Dict,
    budgets: Dict,
    ch_names: Sequence[str],
    splits: Dict,
    planned_seeds: Iterable[int],
    n_random: int,
    allow_errors: bool = False,
    require_complete: bool = False,
) -> Dict:
    """Audit result rows. Returns a report; `ok` is True only if no check failed."""
    fails: Dict[str, List] = defaultdict(list)
    warns: Dict[str, List] = defaultdict(list)

    def fail(check, src, detail=""):
        fails[check].append((src, detail))

    def warn(check, src, detail=""):
        warns[check].append((src, detail))

    # A failed condition is not a result. It is reported as a failure unless the
    # caller explicitly accepts an in-progress sweep, and even then it is listed.
    for e in error_rows:
        (warn if allow_errors else fail)("error_rows", e.get("_src", "?"), str(e.get("error"))[:120])

    manifest_keys = {split: {condition_key(m) for m in man} for split, man in manifests.items()}
    planned = {int(s) for s in planned_seeds}
    seen: Counter = Counter()
    hashes: Counter = Counter()

    for r in rows:
        src = r.get("_src", "?")
        key = condition_key(r)
        seen[key] += 1
        hashes[r.get("config_sha256")] += 1
        split = r["split"]

        # -- membership in the written-down experiment
        if split not in manifest_keys:
            fail("manifest_missing", src, "no manifest for split {!r}".format(split))
        elif key not in manifest_keys[split]:
            fail("not_in_manifest", src, str(key))
        if int(r["train_seed"]) not in planned:
            fail("seed_declared", src, "train_seed {} not in {}".format(r["train_seed"], sorted(planned)))
        if r["selection"] == "random":
            ss = r.get("selection_seed")
            if ss is None or not 0 <= int(ss) < int(n_random):
                fail("seed_declared", src, "selection_seed {} outside 0..{}".format(ss, int(n_random) - 1))
        elif r.get("selection_seed") is not None:
            fail("seed_declared", src, "selection_seed set on a {} row".format(r["selection"]))

        # -- split
        expected_subjects = sorted(sid(s) for s in splits[split])
        got_subjects = sorted(r["n_trials_per_subject"].keys())
        if got_subjects != expected_subjects:
            fail("split_subjects", src, "evaluated {} vs splits.json {}".format(got_subjects, expected_subjects))
        if r.get("splits_seed") != splits["seed"]:
            fail("splits_seed", src, "{} vs {}".format(r.get("splits_seed"), splits["seed"]))
        if r.get("n_subjects") != len(expected_subjects):
            fail("n_subjects", src, str(r.get("n_subjects")))

        # -- channels: both representations, re-derived from the frozen ranking
        chans = list(r["channels"])
        chans_m = list(r["channels_montage_order"])
        k = int(r["budget_k"])
        if not (r["n_channels"] == k == len(chans) == len(chans_m)):
            fail("channel_count", src, "n_channels={} k={} len(channels)={} len(montage)={}".format(
                r["n_channels"], k, len(chans), len(chans_m)))
        derived = None
        try:
            derived = select_channels(r["selection"], k, ch_names, ranked, seed=r.get("selection_seed"))
        except Exception as exc:  # noqa: BLE001
            fail("channel_derivation", src, repr(exc))
        if derived is not None:
            if set(derived) != set(chans_m):
                fail("channel_set", src, "montage {} vs derived {}".format(chans_m, derived))
            if chans_m != [c for c in ch_names if c in set(chans_m)]:
                fail("channel_order", src, "not in montage order: {}".format(chans_m))
            if chans != derived:
                fail("channels_field", src, "channels {} vs derived {}".format(chans, derived))
            if r["selection"] == "ranked":
                frozen = budgets["sets"].get(str(k))
                if (frozen is None or chans_m != frozen["montage_order"]
                        or chans != frozen["ranking_order"]):
                    fail("frozen_budget", src, "row {} vs budgets.json {}".format(
                        chans, frozen and frozen["ranking_order"]))

        # -- provenance
        rp = r.get("ranking_provenance") or {}
        if (rp.get("git_commit") != rank_prov["git_commit"]
                or rp.get("config_hash") != rank_prov["config_hash"]):
            fail("ranking_provenance", src, "{} vs {}".format(rp, rank_prov))
        if r.get("config_sha256") != expected_config_hash:
            fail("config_hash_expected", src, "{} vs expected {}".format(
                r.get("config_sha256"), expected_config_hash))
        if r.get("git_commit") in (None, "unknown"):
            warn("git_commit_unknown", src, "")

        # -- labels and trials
        if r.get("shuffle_labels") and r["selection"] != "ranked":
            warn("shuffle_on_non_ranked", src, r["selection"])
        per = r["n_trials_per_subject"]
        if sum(per.values()) != r["n_trials"]:
            fail("n_trials", src, "{} vs {}".format(sum(per.values()), r["n_trials"]))
        if any(not 40 <= v <= 45 for v in per.values()):
            fail("trials_per_subject", src, str(per))
        if sorted(r["kappa_per_subject"].keys()) != got_subjects:
            fail("kappa_per_subject_keys", src, "")
        if r.get("aggregation_primary") != "pooled":
            warn("aggregation", src, str(r.get("aggregation_primary")))

    for key, n in seen.items():
        if n > 1:
            fail("duplicate_condition", str(key), "x{}".format(n))
    if len(hashes) > 1:
        fail("config_hash_mixed", "all", str(dict(hashes)))

    # -- coverage, per split, against the manifest that defines the experiment
    coverage: Dict[str, Dict] = {}
    for split, mkeys in manifest_keys.items():
        done = {k for k in seen if k[5] == split} & mkeys
        by_arm = Counter((m[1], m[3], m[6]) for m in mkeys)
        done_arm = Counter((m[1], m[3], m[6]) for m in done)
        coverage[split] = {
            "done": len(done),
            "planned": len(mkeys),
            "complete": len(done) == len(mkeys),
            "by_arm": {
                "{} / {}{}".format(a[0], a[1], " / shuffled" if a[2] else ""): [done_arm.get(a, 0), by_arm[a]]
                for a in sorted(by_arm)
            },
        }
    complete = bool(coverage) and all(c["complete"] for c in coverage.values())
    if require_complete and not complete:
        for split, c in coverage.items():
            if not c["complete"]:
                fail("incomplete_coverage", split, "{}/{} conditions".format(c["done"], c["planned"]))
        if not coverage:
            fail("incomplete_coverage", "all", "no manifest to measure coverage against")

    return {
        "n_rows": len(rows),
        "n_error_rows": len(error_rows),
        "config_hashes": dict(hashes),
        "expected_config_hash": expected_config_hash,
        "fails": dict(fails),
        "warns": dict(warns),
        "coverage": coverage,
        "complete": complete,
        "ok": not fails,
    }
