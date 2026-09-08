"""Result rows stamp the commit outside a git checkout, from the bundle's COMMIT file.

Kaggle copies the unpacked code zip, which has no .git, so every row of the
validation sweep says git_commit "unknown". The bundle script now archives a
COMMIT file at the zip root and src.utils.get_git_commit reads it.
"""
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from src import utils

REPO = Path(__file__).resolve().parents[1]
HASH = "8f913201411caf80f97ce4ba530bc6271400c069"


@pytest.fixture
def snapshot(monkeypatch, tmp_path):
    """A code tree with no .git, as Kaggle's copy of the code dataset is."""
    monkeypatch.setattr(utils, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "no-such-repo"))
    return tmp_path


def test_commit_file_supplies_the_hash(snapshot):
    (snapshot / "COMMIT").write_text(HASH + "\n")
    assert utils.get_git_commit() == HASH


def test_no_git_and_no_commit_file_is_unknown(snapshot):
    assert utils.get_git_commit() == "unknown"


def test_malformed_commit_file_is_unknown(snapshot):
    (snapshot / "COMMIT").write_text("latest main\n")
    assert utils.get_git_commit() == "unknown"


def test_checkout_still_uses_git(monkeypatch):
    monkeypatch.delenv("GIT_DIR", raising=False)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    assert utils.get_git_commit().startswith(head)


def test_bundle_zip_carries_commit_file(tmp_path):
    """The notebook copies only the unpacked zip, so COMMIT must be inside it."""
    subprocess.run([sys.executable, "scripts/make_kaggle_bundle.py", "--code",
                    "--out", str(tmp_path), "--allow-dirty"],
                   cwd=REPO, check=True, capture_output=True, text=True, timeout=120)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    with zipfile.ZipFile(tmp_path / "mve-code" / "code.zip") as z:
        names = z.namelist()
        assert "COMMIT" in names
        assert z.read("COMMIT").decode().strip() == head
        assert "src/utils.py" in names, "files sit at the zip root, not under a prefix"
