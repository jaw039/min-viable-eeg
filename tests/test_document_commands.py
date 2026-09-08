"""Documentation must not advertise commands that fail.

These tests read the real CLI parsers and the shell blocks in the README, the
docs and the Kaggle notebook, and check that every documented flag exists.
They catch the failure that shipped here once: a workflow documented in three
places after the command behind it had changed.
"""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
KAGGLE = REPO / "docs" / "kaggle.md"
REPRODUCING = REPO / "docs" / "reproducing.md"
RESULTS_README = REPO / "results" / "README.md"
NOTEBOOK = REPO / "notebooks" / "kaggle_sweep.ipynb"
DOCS = {
    "README.md": README,
    "docs/kaggle.md": KAGGLE,
    "docs/reproducing.md": REPRODUCING,
    "results/README.md": RESULTS_README,
}

# Writing a test manifest through the generic sweep writer produced a matrix
# with no 64-channel reference, so the retained fraction on test could not be
# computed. The command is refused and must not be instructed anywhere.
REMOVED_COMMAND = "--write-manifest --split test"


def _cli_flags(script: str):
    """Flags the script's argparse parser actually accepts."""
    spec = importlib.util.spec_from_file_location(
        "_cli_" + script.replace(".", "_"), REPO / "scripts" / script
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {opt for action in mod.build_parser()._actions for opt in action.option_strings}


RUN_SWEEP_FLAGS = _cli_flags("run_sweep.py")
ANALYZE_FLAGS = _cli_flags("analyze.py")
AUDIT_FLAGS = _cli_flags("audit_results.py")


def _command_lines(text: str):
    """Lines inside fenced bash blocks only.

    Prose that names a flag while explaining a change is not an instruction;
    only fenced commands are. Backslash continuations are joined.
    """
    lines = []
    for block in re.findall(r"```(?:bash|sh)?\n(.*?)```", text, re.S):
        lines.extend(block.replace("\\\n", " ").splitlines())
    return lines


def _notebook_command_lines():
    nb = json.loads(NOTEBOOK.read_text())
    lines = []
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            lines.extend("".join(c["source"]).replace("\\\n", " ").splitlines())
    return lines


def _lines_for(name: str):
    return _notebook_command_lines() if name == "notebook" else _command_lines(DOCS[name].read_text())


def _documented_flags(lines, script: str):
    """Flags used against `script` in actual command lines."""
    flags = set()
    for line in lines:
        if script in line:
            flags.update(m.group(1) for m in re.finditer(r"(--[a-z][a-z0-9-]*)", line))
    return flags


# --------------------------------------------------------------------------
# The CLI offers the confirmatory workflow, and the removed command is gone
# --------------------------------------------------------------------------

def test_cli_has_the_confirmatory_flags():
    assert {"--write-test-manifest", "--kstar", "--kstar-report", "--results-dir"} <= RUN_SWEEP_FLAGS
    assert {"--test-report", "--out"} <= ANALYZE_FLAGS
    assert {"--require-complete", "--allow-errors"} <= AUDIT_FLAGS


def test_removed_command_is_not_instructed_anywhere():
    """It may be mentioned while explaining the change, never given as a
    command inside a shell block or a notebook cell."""
    for name in DOCS:
        for line in _lines_for(name):
            assert REMOVED_COMMAND not in line, "{} still instructs the removed command".format(name)
    for line in _notebook_command_lines():
        assert REMOVED_COMMAND not in line


def test_removed_command_actually_fails_now():
    """If this ever starts succeeding, the docs above become wrong again."""
    r = subprocess.run(
        [sys.executable, "scripts/run_sweep.py", "--write-manifest", "--split", "test",
         "--budget", "8"],
        cwd=REPO, capture_output=True, text=True, timeout=180,
    )
    assert r.returncode != 0
    assert "--write-test-manifest" in (r.stdout + r.stderr)


# --------------------------------------------------------------------------
# Every documented flag exists in the CLI
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(DOCS) + ["notebook"])
def test_documented_run_sweep_flags_exist(name):
    unknown = _documented_flags(_lines_for(name), "run_sweep.py") - RUN_SWEEP_FLAGS
    assert not unknown, "{} documents unknown run_sweep flags: {}".format(name, sorted(unknown))


@pytest.mark.parametrize("name", sorted(DOCS) + ["notebook"])
def test_documented_analyze_flags_exist(name):
    unknown = _documented_flags(_lines_for(name), "analyze.py") - ANALYZE_FLAGS
    assert not unknown, "{} documents unknown analyze flags: {}".format(name, sorted(unknown))


@pytest.mark.parametrize("name", sorted(DOCS) + ["notebook"])
def test_documented_audit_flags_exist(name):
    unknown = _documented_flags(_lines_for(name), "audit_results.py") - AUDIT_FLAGS
    assert not unknown, "{} documents unknown audit flags: {}".format(name, sorted(unknown))


# --------------------------------------------------------------------------
# The restricted workflow is documented, in the right order
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", [README, KAGGLE, REPRODUCING])
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
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "KSTAR = None" in code, "the notebook must not default a k*"
    assert "--write-test-manifest" in code


def test_notebook_is_valid_json_and_has_only_known_cell_types():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["cells"]
    for c in nb["cells"]:
        assert c["cell_type"] in ("markdown", "code")


def test_readme_does_not_hard_code_a_test_count():
    """A count goes stale with the next test added; the README says what the
    skips are instead."""
    assert not re.search(r"\d+ passed, \d+ skipped", README.read_text())
