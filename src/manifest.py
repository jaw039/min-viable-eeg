"""The condition matrix, and deterministic sharding across teammates.

The manifest is the experiment. Generating it once, before anything runs, is
what makes "we ran the sweep we described" checkable afterwards rather than a
claim resting on memory.

Two properties are deliberate:

* **Validation by default.** Test rows are produced only for an explicitly
  named budget. Sweeping test across every budget and reporting the smallest
  one that clears the threshold is test-set selection wearing a definition as a
  disguise.
* **Controls are rows.** The random baseline, the sensorimotor arm, the
  shuffled-teacher control and the label-shuffle floor are all in the default
  matrix, so dropping one is a visible edit rather than something that quietly
  never happened.

Sharding is round-robin over a stable ordering, so `--shard-id 0..3
--num-shards 4` covers the matrix exactly once with no coordination between
teammates.
"""

from typing import Dict, List, Optional, Sequence

from src.channels import max_k

# Budgets small enough that which channels you pick plausibly matters.
LOW_BUDGETS = (4, 6, 8)


def condition_key(row: Dict) -> tuple:
    """Identity of a condition, for resume and coverage checks."""
    return (
        int(row["budget_k"]),
        row["selection"],
        row.get("selection_seed"),
        row.get("training", "scratch"),
        bool(row.get("shuffle_labels", False)),
        int(row["train_seed"]),
        row["split"],
    )


def build_manifest(
    budgets: Sequence[int],
    ch_names: Sequence[str],
    train_seeds: Sequence[int],
    n_random: int = 20,
    split: str = "val",
    include_distillation: bool = True,
) -> List[Dict]:
    rows: List[Dict] = []

    def add(k, selection, train_seed, selection_seed=None, training="scratch",
            shuffle_labels=False):
        rows.append({
            "budget_k": int(k),
            "selection": selection,
            "selection_seed": selection_seed,
            "training": training,
            "shuffle_labels": shuffle_labels,
            "train_seed": int(train_seed),
            "split": split,
        })

    smax = max_k("sensorimotor", ch_names)

    for seed in train_seeds:
        # Headline arm, every budget. k=64 is the full-montage reference every
        # ratio in the paper divides by.
        for k in budgets:
            add(k, "ranked", seed)

        # Control arm. A "random" subset of the whole montage is the montage,
        # so skip it there.
        for k in budgets:
            if k < len(ch_names):
                for i in range(n_random):
                    add(k, "random", seed, selection_seed=i)

        # Mechanism arm: motor ERD or cue-correlated activity?
        for k in budgets:
            if k <= smax:
                add(k, "sensorimotor", seed)

        if include_distillation:
            for k in budgets:
                if k in LOW_BUDGETS:
                    add(k, "ranked", seed, training="distill")
                    add(k, "ranked", seed, training="distill_shuffled_teacher")

    # Negative control: one run establishes the floor.
    add(max(budgets), "ranked", train_seeds[0], shuffle_labels=True)
    return rows


def shard(rows: Sequence[Dict], shard_id: int, num_shards: int) -> List[Dict]:
    """Round-robin slice. Every row lands in exactly one shard."""
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= shard_id < num_shards:
        raise ValueError("shard_id must be in 0..{}".format(num_shards - 1))
    return [r for i, r in enumerate(rows) if i % num_shards == shard_id]


def summarise(rows: Sequence[Dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        if r.get("shuffle_labels"):
            key = "label-shuffle control"
        elif r.get("training", "scratch") != "scratch":
            key = r["training"]
        else:
            key = r["selection"]
        out[key] = out.get(key, 0) + 1
    return out


def missing_conditions(
    planned: Sequence[Dict], completed: Sequence[Dict]
) -> List[Dict]:
    """Planned conditions with no completed row -- the coverage gap.

    Methods that describe the planned sweep while Results come from the one
    that finished is the failure this exists to make visible.
    """
    done = {condition_key(r) for r in completed}
    return [r for r in planned if condition_key(r) not in done]
