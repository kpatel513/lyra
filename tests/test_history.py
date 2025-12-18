from __future__ import annotations

from pathlib import Path

import pytest

from lyra.history import create_history_entry, finalize_history_entry, undo_history


def test_history_undo_restores_modified_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    f = repo / "train.py"
    f.write_text("a=1\n", encoding="utf-8")

    entry = create_history_entry(repo=repo, command="test")

    # Modify file
    f.write_text("a=2\n", encoding="utf-8")
    finalize_history_entry(entry)

    # Undo should restore backed-up version
    undo_history(repo=repo, run_id=entry.run_id, force=True)
    assert f.read_text(encoding="utf-8") == "a=1\n"


def test_history_undo_refuses_when_diverged_without_force(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    f = repo / "train.py"
    f.write_text("a=1\n", encoding="utf-8")

    entry = create_history_entry(repo=repo, command="test")
    f.write_text("a=2\n", encoding="utf-8")
    finalize_history_entry(entry)

    # Diverge after finalize
    f.write_text("a=3\n", encoding="utf-8")

    with pytest.raises(RuntimeError):
        undo_history(repo=repo, run_id=entry.run_id, force=False)


