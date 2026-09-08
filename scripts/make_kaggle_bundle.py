"""Build the two zip files the Kaggle sweep runs from.

    dist/mve-code/code.zip               git archive of HEAD + COMMIT, files at the zip root
    dist/mve-eegmmidb-cache/cache.zip    processed/S###/{X,y}.npy and
                                         {splits,channel_ranking,budgets}.json at the root

Kaggle extracts an uploaded zip into a folder named after the file (code.zip
becomes code/), so these mount at /kaggle/input/datasets/<owner>/mve-code/code
and /kaggle/input/datasets/<owner>/mve-eegmmidb-cache/cache, which is what
notebooks/kaggle_sweep.ipynb expects. Keep the file names: a zip called
anything else lands in a differently named folder. A dataset-metadata.json is
written next to each zip so the Kaggle CLI can upload the folder as a private
dataset; when uploading a new version in the browser, remove the previous
code.zip from the version so the new one replaces it.

Usage:
    python scripts/make_kaggle_bundle.py --code            # refuses a dirty tree
    python scripts/make_kaggle_bundle.py --cache           # needs data/processed
    kaggle datasets create -p dist/mve-code                # first upload
    kaggle datasets version -p dist/mve-code -m "commit <sha>"   # new version
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ARTIFACTS_DIR, REPO_ROOT

CACHE_ARTIFACTS = ("splits.json", "channel_ranking.json", "budgets.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metadata(folder: Path, slug: str, owner: str) -> None:
    (folder / "dataset-metadata.json").write_text(json.dumps({
        "title": slug,
        "id": "{}/{}".format(owner, slug),
        "licenses": [{"name": "other"}],
    }, indent=2) + "\n")


def build_code(out: Path, owner: str, allow_dirty: bool) -> Path:
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True)
    if status.strip() and not allow_dirty:
        raise SystemExit("working tree is dirty; commit first so the snapshot equals a commit "
                         "(or pass --allow-dirty for a smoke test)")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    folder = out / "mve-code"
    folder.mkdir(parents=True, exist_ok=True)
    zip_path = folder / "code.zip"
    # COMMIT goes inside the zip too: the snapshot has no .git, so the runner
    # stamps rows from this file (src.utils.get_git_commit).
    commit_file = folder / "COMMIT"
    commit_file.write_text(head + "\n")
    subprocess.check_call(["git", "archive", "--format=zip", "--add-file=" + str(commit_file),
                           "-o", str(zip_path), "HEAD"], cwd=REPO_ROOT)
    metadata(folder, "mve-code", owner)
    print("code   : {}  commit {}".format(zip_path, head))
    return zip_path


def build_cache(out: Path, owner: str) -> Path:
    env = os.environ.get("MVE_DATA_ROOT")
    processed = Path(env) if env else REPO_ROOT / "data" / "processed"
    subjects = sorted(p for p in processed.glob("S*") if (p / "X.npy").exists())
    if not subjects:
        raise SystemExit("no cached subjects under {}; run scripts/cache_preprocessed.py".format(processed))
    folder = out / "mve-eegmmidb-cache"
    folder.mkdir(parents=True, exist_ok=True)
    zip_path = folder / "cache.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for name in CACHE_ARTIFACTS:
            z.write(ARTIFACTS_DIR / name, name)
        for extra in processed.glob("*.jsonl"):
            z.write(extra, "processed/{}".format(extra.name))
        for s in subjects:
            for f in ("X.npy", "y.npy"):
                z.write(s / f, "processed/{}/{}".format(s.name, f))
    metadata(folder, "mve-eegmmidb-cache", owner)
    print("cache  : {}  {} subjects".format(zip_path, len(subjects)))
    return zip_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", action="store_true")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--owner", default="jackiewang2323", help="Kaggle username that owns the datasets")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "dist")
    args = ap.parse_args()
    if not (args.code or args.cache):
        ap.error("pass --code, --cache or both")
    built = []
    if args.code:
        built.append(build_code(args.out, args.owner, args.allow_dirty))
    if args.cache:
        built.append(build_cache(args.out, args.owner))
    for p in built:
        print("{:<8} {:>8.1f} MB  sha256:{}".format(p.name, p.stat().st_size / 1e6, sha256(p)[:16]))


if __name__ == "__main__":
    main()
