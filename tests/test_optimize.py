from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from lyra.optimize import optimize_repo


def test_optimize_repo_dry_run_profiles_only(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('x')\n", encoding="utf-8")

    # Avoid actually running subprocess; core.run_safe_profile uses subprocess.run
    with patch("subprocess.run") as run:
        run.return_value.returncode = 0
        run.return_value.stdout = ""
        run.return_value.stderr = ""

        report = optimize_repo(
            repo=repo,
            training_script="train.py",
            max_steps=3,
            apply=False,
            project_root=tmp_path,
        )

    assert report.applied is False
    assert "profile" in report.before
    assert report.after is None


