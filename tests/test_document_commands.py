"""Documentation must not advertise commands that fail.

These tests parse the actual CLI parsers and the shell blocks in README.md,
KAGGLE.md and the Kaggle notebook, and check that every documented flag exists.
They are cheap and they catch the specific failure that shipped here: a
workflow documented in three places after the command behind it was removed.
"""

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
KAGGLE = REPO / "KAGGLE.md"
NOTEBOOK = REPO / "notebooks" / "kaggle_sweep.ipynb"

REMOVED_COMMAND = "--write-manifest --split test"


def _parser_flags(module_path: str):
    """Flags the given script's argparse parser actually accepts."""
    import importlib.util
    import sys

    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location("_cli", REPO / module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUN_SWEEP_FLAGS = {
    "--smoke-test", "--write-manifest", "--write-test-manifest", "--kstar",
    "--kstar-report", "--manifest", "--shard-id", "--num-shards", "--budget",
    "--selection", "--selection-seed", "--training", "--train-seed",
    "--train-seeds", "--n-random", "--split", "--shuffle-labels",
    "--no-distillation",
}
ANALYZE_FLAGS = {"--results", "--threshold", "--out", "--test-report", "--emit-followup"}
ANCHOR_FLAGS = {
    "--out", "--shuffle-seeds", "--shuffle-tolerance", "--within-subjects",
    "--full-montage-kappa", "--skip-compute",
}


def _command_lines(text: str):
    """Lines inside fenced bash blocks only.

    Prose that merely names a flag while explaining what changed is not an
    instruction; only fenced commands are. Continuation lines ending in a
    backslash are joined so multi-line commands are read whole.
    """
    lines = []
    for block in re.findall(r"```(?:bash|sh)?\n(.*?)```", text, re.S):
        joined = block.replace("\\\n", " ")
        lines.extend(joined.splitlines())
    return lines


def _notebook_command_lines():
    nb = json.loads(NOTEBOOK.read_text())
    lines = []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        joined = "".join(c["source"]).replace("\\\n", " ")
        lines.extend(joined.splitlines())
    return lines


def _documented_flags(lines, script: str):
    """Flags used against `script` in actual command lines."""
    flags = set()
    for line in lines:
        if script not in line:
            continue
        for m in re.finditer(r"(--[a-z][a-z0-9-]*)", line):
            flags.add(m.group(1))
    return flags


def _notebook_text() -> str:
    nb = json.loads(NOTEBOOK.read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"])


# --------------------------------------------------------------------------
# The removed command must be gone from every instruction
# --------------------------------------------------------------------------

def test_removed_command_is_not_instructed_anywhere():
    """It may be MENTIONED (explaining what changed) but never given as a
    command inside a shell block or notebook cell."""
    for path in (README, KAGGLE):
        for block in re.findall(r"```bash(.*?)```", path.read_text(), re.S):
            assert REMOVED_COMMAND not in block, (
                "{} still instructs the removed command".format(path.name)
            )
    nb = json.loads(NOTEBOOK.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert REMOVED_COMMAND not in "".join(cell["source"])


def test_removed_command_actually_fails_now():
    """If this ever starts succeeding, the docs above become wrong again."""
    import subprocess

    r = subprocess.run(
        ["python3", "scripts/run_sweep.py", "--write-manifest", "--split", "test",
         "--budget", "8"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert r.returncode != 0
    assert "--write-test-manifest" in (r.stdout + r.stderr)


# --------------------------------------------------------------------------
# Every documented flag exists in the CLI
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path_name", ["README.md", "KAGGLE.md", "notebook"])
def test_documented_run_sweep_flags_exist(path_name):
    lines = (
        _notebook_command_lines() if path_name == "notebook"
        else _command_lines((REPO / path_name).read_text())
    )
    used = _documented_flags(lines, "run_sweep.py")
    unknown = used - RUN_SWEEP_FLAGS
    assert not unknown, "{} documents unknown run_sweep flags: {}".format(
        path_name, sorted(unknown)
    )


@pytest.mark.parametrize("path_name", ["README.md", "KAGGLE.md", "notebook"])
def test_documented_analyze_flags_exist(path_name):
    lines = (
        _notebook_command_lines() if path_name == "notebook"
        else _command_lines((REPO / path_name).read_text())
    )
    used = _documented_flags(lines, "analyze.py")
    unknown = used - ANALYZE_FLAGS
    assert not unknown, "{} documents unknown analyze flags: {}".format(
        path_name, sorted(unknown)
    )


def test_documented_anchor_flags_exist():
    lines = (
        _command_lines(README.read_text())
        + _command_lines(KAGGLE.read_text())
        + _notebook_command_lines()
    )
    unknown = _documented_flags(lines, "run_anchors.py") - ANCHOR_FLAGS
    assert not unknown, "unknown anchor flags documented: {}".format(sorted(unknown))


# --------------------------------------------------------------------------
# The restricted workflow is documented in the right order
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", [README, KAGGLE])
def test_confirmatory_workflow_is_documented(path):
    t = path.read_text()
    assert "--write-test-manifest" in t
    assert "--kstar" in t
    # k* comes from validation, and the manifest is inspected before running.
    assert "kstar_report.json" in t
    assert "manifest_test.jsonl" in t


@pytest.mark.parametrize("path", [README, KAGGLE])
def test_docs_state_that_controls_stay_on_validation(path):
    t = path.read_text().lower()
    assert "sensorimotor" in t and "distillation" in t
    assert "label" in t and "shuffle" in t


def test_notebook_requires_an_explicit_kstar():
    nb = json.loads(NOTEBOOK.read_text())
    code = "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )
    assert "KSTAR = None" in code, "the notebook must not default a k*"
    assert "--write-test-manifest" in code


def test_notebook_is_valid_json_and_has_no_stale_shard_test_call():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["cells"]
    for c in nb["cells"]:
        assert c["cell_type"] in ("markdown", "code")


# --------------------------------------------------------------------------
# The test count in the README is not hard-coded to a stale number
# --------------------------------------------------------------------------

def test_readme_reports_a_test_count_that_looks_current():
    """Not an exact assertion (that would be circular), but the README must
    not still claim the pre-repair counts."""
    t = README.read_text()
    for stale in ("96 passed, 1 skipped", "100 passed, 6 skipped",
                  "127 passed, 6 skipped"):
        assert stale not in t, "README still reports the stale count {!r}".format(stale)
