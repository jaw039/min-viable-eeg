"""Inventory of downloaded EEGMMIDB EDF files.

Run as: python -m src.inventory

Writes data/inventory.csv with per-file sampling rate, channel count, and
annotation counts, and prints a WARNING for any file deviating from the
locked protocol (160 Hz, 64 channels).
"""

import csv
import re
from collections import Counter

import mne

from src.utils import data_root, load_config, provenance

# Locked protocol (CLAUDE.md): EEGMMIDB is 160 Hz, 64 channels.
EXPECTED_SFREQ = 160.0
EXPECTED_N_CHANNELS = 64

FILENAME_RE = re.compile(r"S(\d{3})R(\d{2})\.edf$")
FIELDNAMES = ["subject", "run", "sfreq", "n_channels", "n_T0", "n_T1", "n_T2"]


def main() -> None:
    config = load_config()
    edf_root = data_root(config) / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0"
    edf_files = sorted(edf_root.glob("S*/S*R*.edf"))
    if not edf_files:
        raise SystemExit(
            "No EDF files under {} — run scripts/download_data.py first".format(edf_root)
        )

    rows = []
    for path in edf_files:
        match = FILENAME_RE.search(path.name)
        if match is None:
            continue
        subject, run = int(match.group(1)), int(match.group(2))
        # Header-only read; annotations are parsed without loading signal data.
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        sfreq = raw.info["sfreq"]
        n_channels = len(raw.ch_names)
        counts = Counter(raw.annotations.description)
        if sfreq != EXPECTED_SFREQ:
            print("WARNING: {} sfreq={} != {}".format(path.name, sfreq, EXPECTED_SFREQ))
        if n_channels != EXPECTED_N_CHANNELS:
            print(
                "WARNING: {} n_channels={} != {}".format(
                    path.name, n_channels, EXPECTED_N_CHANNELS
                )
            )
        rows.append({
            "subject": subject,
            "run": run,
            "sfreq": sfreq,
            "n_channels": n_channels,
            "n_T0": counts.get("T0", 0),
            "n_T1": counts.get("T1", 0),
            "n_T2": counts.get("T2", 0),
        })

    meta = provenance()
    out_path = data_root(config) / "inventory.csv"
    with open(out_path, "w", newline="") as f:
        # Comment header keeps provenance in the output; pandas reads it with comment="#".
        f.write("# git_commit: {}\n".format(meta["git_commit"]))
        f.write("# config_hash: {}\n".format(meta["config_hash"]))
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["subject"], r["run"])))
    print("Wrote {} rows to {}".format(len(rows), out_path))


if __name__ == "__main__":
    main()
