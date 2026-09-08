"""The confirmatory test manifest is written only from a complete, passing
validation report, and never silently replaced by a different one.

These guard the rule the paper depends on: k* is chosen on validation, once,
and the test split is evaluated at that k* and nothing else.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from src.utils import REPO_ROOT, load_config

CFG = load_config()


def _run_sweep():
    spec = importlib.util.spec_from_file_location("_run_sweep_cli", REPO_ROOT / "scripts" / "run_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _report(**over):
    r = {
        "n_errors": 0,
        "coverage": {"complete": True},
        "negative_control": {"passes": True},
        "kstar": {"kstar": 32, "selected_on": "val", "provisional": False},
    }
    for key, val in over.items():
        if isinstance(val, dict) and isinstance(r.get(key), dict):
            r[key].update(val)
        else:
            r[key] = val
    return r


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    mod = _run_sweep()
    monkeypatch.setattr(mod, "MANIFEST_DIR", tmp_path)
    return mod


def _args(report_path=None, kstar=None):
    return argparse.Namespace(kstar=kstar, kstar_report=report_path)


def _write(tmp_path, report):
    p = tmp_path / "kstar_report.json"
    p.write_text(json.dumps(report))
    return p


def test_manifest_is_written_from_a_complete_passing_report(sweep, tmp_path):
    sweep.write_test_manifest(_args(_write(tmp_path, _report())), CFG)
    rows = [json.loads(l) for l in (tmp_path / "manifest_test.jsonl").open()]
    assert {r["budget_k"] for r in rows} == {32, 64}
    assert len(rows) == 2 * len(CFG["sweep"]["train_seeds"])


@pytest.mark.parametrize("bad", [
    {"coverage": {"complete": False}},
    {"n_errors": 3},
    {"negative_control": {"passes": False}},
    {"kstar": {"provisional": True}},
    {"kstar": {"selected_on": "test"}},
])
def test_incomplete_or_failing_report_is_refused(sweep, tmp_path, bad):
    with pytest.raises(SystemExit):
        sweep.write_test_manifest(_args(_write(tmp_path, _report(**bad))), CFG)
    assert not (tmp_path / "manifest_test.jsonl").exists()


def test_a_report_without_the_negative_control_section_is_refused(sweep, tmp_path):
    r = _report()
    del r["negative_control"]
    with pytest.raises(SystemExit):
        sweep.write_test_manifest(_args(_write(tmp_path, r)), CFG)


def test_typed_kstar_must_agree_with_the_report(sweep, tmp_path):
    with pytest.raises(SystemExit, match="disagrees"):
        sweep.write_test_manifest(_args(_write(tmp_path, _report()), kstar=16), CFG)


def test_existing_manifest_with_a_different_selection_is_not_overwritten(sweep, tmp_path):
    sweep.write_test_manifest(_args(kstar=16), CFG)
    before = (tmp_path / "manifest_test.jsonl").read_text()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        sweep.write_test_manifest(_args(_write(tmp_path, _report())), CFG)
    assert (tmp_path / "manifest_test.jsonl").read_text() == before


def test_matching_manifest_is_left_alone(sweep, tmp_path, capsys):
    sweep.write_test_manifest(_args(kstar=32), CFG)
    before = (tmp_path / "manifest_test.jsonl").read_text()
    sweep.write_test_manifest(_args(_write(tmp_path, _report())), CFG)
    assert (tmp_path / "manifest_test.jsonl").read_text() == before
    assert "already matches" in capsys.readouterr().out


def test_cli_without_arguments_prints_help_instead_of_crashing():
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "scripts/run_sweep.py"], cwd=REPO_ROOT,
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0
    assert "--write-test-manifest" in r.stdout


def test_teacher_identity_includes_a_source_hash():
    from src.checkpoints import IDENTITY_FIELDS
    from src.provenance import source_identity

    assert "source_sha256" in IDENTITY_FIELDS
    h = source_identity()
    assert len(h) == 64 and h == source_identity()
