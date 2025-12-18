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
            plan=False,
            yes=False,
            project_root=tmp_path,
        )

    assert report.mode == "dry-run"
    assert "profile" in report.before
    assert report.after is None


def test_optimize_repo_plan_runs_analysis_only(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('x')\n", encoding="utf-8")

    called = {"analyze": 0, "optimize": 0}

    def fake_run_prompt_in_repo(*, prompt_name, output_path, **kwargs):
        if prompt_name == "lyraAnalyze":
            called["analyze"] += 1
        if prompt_name == "lyraOptimize":
            called["optimize"] += 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("ok\n", encoding="utf-8")

    import lyra.optimize as opt

    monkeypatch.setattr(opt, "_run_prompt_in_repo", fake_run_prompt_in_repo)

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
            plan=True,
            yes=False,
            project_root=tmp_path,
        )

    assert report.mode == "plan"
    assert called["analyze"] == 1
    assert called["optimize"] == 0
    assert report.after is None
    assert report.analysis_output is not None
    assert report.optimize_output is None


