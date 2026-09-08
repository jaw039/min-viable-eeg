# Frozen protocol artifacts

Generated once from the training subjects, committed, and never regenerated
in place: each writer refuses to overwrite an existing file. Every downstream
step reads these files rather than recomputing them, so the sweep, the audit
and the paper tables cannot drift from one another.

| File | Contents | Writer | Provenance commit |
| --- | --- | --- | --- |
| `splits.json` | Subject-wise 74/16/16 split, seed 42; excludes S088, S092, S100 | `python -m src.splits` | `55a17a3` |
| `channel_ranking.json` | All 64 electrodes ranked by Fisher score of log band-power (8–30 Hz), averaged over training subjects | `python -m src.ranking` | `3c7a2af` |
| `budgets.json` | Exact electrode set for k ∈ {4, 6, 8, 12, 16, 32, 64}, in ranking order and in montage order | `python -m src.budget` | `b1be5db` |
| `stability.json` | Top-k inclusion frequency over 200 bootstrap resamples of the training subjects (seed 42) and per-subject overlap with the shared ranking | `python -m src.stability` | `b8b88b9` |

sha256 (first 16): `splits` `b5b370b649920612`, `channel_ranking`
`2a78bb38ef66d85c`, `budgets` `2a344260d4d05b58`, `stability`
`4d7c7a95a98800b6`.

Each file carries a `provenance` block: git commit, config hash and UTC
timestamp. All four record config hash `f31815c177f9`, the hash of
`config.yaml` before training hyperparameters were appended; the protocol
sections have not changed since (`protocol_hash() == e9f48c5483a2`), which
`tests/test_protocol_integrity.py` verifies.

`budgets.json` also records the montage order of the data arrays, so the
experiment path can map electrode names to array rows without MNE or the raw
recordings.

Changing any of these is a protocol change: it invalidates every result row
and needs sign-off. To verify them instead, follow Level 1 of
[docs/reproducing.md](../docs/reproducing.md).
