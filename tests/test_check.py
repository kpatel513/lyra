from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from lyra.check import run_check


def test_check_without_claude() -> None:
    with patch("shutil.which", return_value=None):
        report = run_check(repo=None)
    assert any(i.name == "Python" for i in report.items)
    assert any(i.name == "Claude Code CLI" and i.ok is False for i in report.items)


def test_check_repo_write_access(tmp_path: Path) -> None:
    with patch("shutil.which", return_value=None):
        report = run_check(repo=tmp_path)
    assert any(i.name == "Repo path" and i.ok for i in report.items)
    assert any(i.name == "Repo write access" and i.ok for i in report.items)


