"""Download PhysioNet EEGMMIDB runs 4, 8, 12 for the given subjects.

Usage: python scripts/download_data.py --subjects 1 2 3 4 5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mne.datasets import eegbci

from src.utils import data_root, load_config, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects", nargs="+", type=int, default=list(range(1, 6)),
        help="Subject numbers to download (default: 1 2 3 4 5)",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to config.yaml (default: repo root config.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    runs = config["dataset"]["runs"]
    excluded = set(config["dataset"]["exclude_subjects"])
    data_dir = data_root(config)
    data_dir.mkdir(parents=True, exist_ok=True)

    downloaded = {}
    for subject in args.subjects:
        if subject in excluded:
            print("SKIP S{:03d} (excluded by protocol: not 160 Hz)".format(subject))
            continue
        # update_path=False keeps the global MNE config untouched.
        paths = eegbci.load_data(subject, runs, path=str(data_dir), update_path=False)
        downloaded[subject] = [str(p) for p in paths]
        print("S{:03d}: {} files".format(subject, len(paths)))

    log_entry = provenance(args.config)
    log_entry.update({"subjects": sorted(downloaded), "runs": runs, "files": downloaded})
    log_path = data_dir / "download_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print("Logged provenance to {}".format(log_path))


if __name__ == "__main__":
    main()
