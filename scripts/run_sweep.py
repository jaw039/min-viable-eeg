"""The experiment CLI.

    python scripts/run_sweep.py --smoke-test
    python scripts/run_sweep.py --budget 64 --selection ranked --train-seed 42 --split val
    python scripts/run_sweep.py --write-manifest
    python scripts/run_sweep.py --manifest --shard-id 0 --num-shards 4

Shard runs are resumable: a condition already present in the output file is
skipped, so a Kaggle session hitting its 12-hour limit costs nothing. A
condition that raises is recorded as an error row and the sweep continues --
one bad condition should not cost you the other hundred.

Output goes to /kaggle/working/results when that directory exists, otherwise to
./results. Nothing is hard-coded to a user directory.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.channels import montage_order
from src.manifest import build_manifest, condition_key, shard, summarise
from src.runner import run_condition
from src.utils import REPO_ROOT, load_config

MANIFEST_DIR = REPO_ROOT / "manifests"


def results_dir() -> Path:
    kaggle = Path("/kaggle/working")
    base = kaggle if kaggle.exists() else REPO_ROOT
    d = base / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def manifest_path(split: str) -> Path:
    return MANIFEST_DIR / "manifest_{}.jsonl".format(split)


def read_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.open():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def write_manifest(args, cfg) -> None:
    ch_names = montage_order()
    budgets = [args.budget] if args.budget else list(cfg["budgets"])
    seeds = args.train_seeds or list(cfg["sweep"]["train_seeds"])
    n_random = args.n_random if args.n_random is not None else int(cfg["sweep"]["n_random_subsets"])

    if args.split == "test" and args.budget is None:
        raise SystemExit(
            "--write-manifest --split test requires --budget: confirm the single\n"
            "chosen k* on test. Sweeping test across budgets is test-set selection."
        )

    rows = build_manifest(
        budgets=budgets,
        ch_names=ch_names,
        train_seeds=seeds,
        n_random=n_random,
        split=args.split,
        include_distillation=not args.no_distillation,
    )
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = manifest_path(args.split)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print("{} conditions -> {}".format(len(rows), path))
    for kind, n in sorted(summarise(rows).items()):
        print("  {:<26} {}".format(kind, n))
    if args.num_shards > 1:
        per = len(rows) // args.num_shards
        print("  --num-shards {} gives ~{} conditions each".format(args.num_shards, per))


def run_one(args, cfg) -> None:
    row = run_condition(
        budget_k=args.budget,
        selection=args.selection,
        split=args.split,
        training=args.training,
        selection_seed=args.selection_seed,
        train_seed=args.train_seed,
        shuffle_labels=args.shuffle_labels,
        cfg=cfg,
        teacher_cache=results_dir().parent / "teachers",
    )
    out = results_dir() / "single.jsonl"
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))
    print("\nappended to {}".format(out))


def run_shard(args, cfg) -> None:
    path = manifest_path(args.split)
    if not path.exists():
        raise SystemExit(
            "No manifest at {}. Generate it first:\n"
            "    python scripts/run_sweep.py --write-manifest".format(path)
        )
    planned = read_jsonl(path)
    mine = shard(planned, args.shard_id, args.num_shards)

    out = results_dir() / "shard_{}_of_{}_{}.jsonl".format(
        args.shard_id, args.num_shards, args.split
    )
    done = {condition_key(r) for r in read_jsonl(out) if "error" not in r}
    todo = [c for c in mine if condition_key(c) not in done]

    print("shard {}/{}: {} conditions, {} already done, {} to run".format(
        args.shard_id, args.num_shards, len(mine), len(done), len(todo)))
    if not todo:
        print("nothing to do")
        return

    teacher_cache = results_dir().parent / "teachers"
    started = time.time()
    with out.open("a") as f:
        for i, c in enumerate(todo, 1):
            label = "k={:<3}{:<14}{:<26}seed={}".format(
                c["budget_k"], c["selection"], c.get("training", "scratch"), c["train_seed"]
            )
            try:
                row = run_condition(
                    budget_k=c["budget_k"],
                    selection=c["selection"],
                    split=c["split"],
                    training=c.get("training", "scratch"),
                    selection_seed=c.get("selection_seed"),
                    train_seed=c["train_seed"],
                    shuffle_labels=c.get("shuffle_labels", False),
                    cfg=cfg,
                    teacher_cache=teacher_cache,
                )
                f.write(json.dumps(row) + "\n")
                f.flush()
                print("[{}/{}] {} kappa={:.3f} ({}s)".format(
                    i, len(todo), label, row["kappa"], row["runtime_sec"]))
            except Exception as exc:  # noqa: BLE001 - one bad row must not stop the sweep
                f.write(json.dumps(dict(c, error="{}: {}".format(type(exc).__name__, exc))) + "\n")
                f.flush()
                print("[{}/{}] {} FAILED {}: {}".format(
                    i, len(todo), label, type(exc).__name__, exc))

    print("\ndone in {:.1f} min -> {}".format((time.time() - started) / 60, out))


def smoke_test(cfg) -> None:
    """One cheap condition, to prove the path works before spending quota."""
    cfg = json.loads(json.dumps(cfg))  # deep copy
    cfg["training"]["max_epochs"] = 2
    cfg["training"]["patience"] = 2
    print("smoke test: k=8 ranked, val, 2 epochs\n")
    t0 = time.time()
    row = run_condition(budget_k=8, selection="ranked", split="val", cfg=cfg)
    print(json.dumps(
        {k: row[k] for k in ("budget_k", "selection", "channels", "kappa",
                             "kappa_macro_subject", "n_trials", "n_subjects",
                             "device", "runtime_sec")},
        indent=2,
    ))
    print("\nOK in {:.1f}s. Multiply by your shard size before planning.".format(
        time.time() - t0))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--write-manifest", action="store_true")
    p.add_argument("--manifest", action="store_true", help="run a shard of the manifest")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--selection", choices=("ranked", "random", "sensorimotor"),
                   default="ranked")
    p.add_argument("--selection-seed", type=int, default=None)
    p.add_argument("--training",
                   choices=("scratch", "distill", "distill_shuffled_teacher"),
                   default="scratch")
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--train-seeds", type=int, nargs="+", default=None)
    p.add_argument("--n-random", type=int, default=None)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--shuffle-labels", action="store_true",
                   help="negative control: expect kappa near zero")
    p.add_argument("--no-distillation", action="store_true")
    args = p.parse_args()

    cfg = load_config()

    if args.smoke_test:
        smoke_test(cfg)
    elif args.write_manifest:
        write_manifest(args, cfg)
    elif args.manifest:
        run_shard(args, cfg)
    elif args.budget is not None:
        run_one(args, cfg)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
