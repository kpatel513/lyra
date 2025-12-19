from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from lyra.setup_env import detect_repo_deps, setup_venv


def test_detect_repo_deps(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("x\n", encoding="utf-8")
    deps = detect_repo_deps(tmp_path)
    assert "requirements.txt" in deps


def test_setup_venv_creates_venv_and_skips_install(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    venv_dir = repo / ".venv"

    with patch("subprocess.run") as run:
        # venv creation succeeds
        run.return_value.returncode = 0
        run.return_value.stdout = ""
        run.return_value.stderr = ""

        res = setup_venv(
            repo=repo,
            venv_dir=venv_dir,
            python_executable="python",
            install=False,
            requirements_file=None,
            verify=False,
        )

    assert res.kind == "venv"
    assert res.env_path == venv_dir.resolve()


def test_setup_venv_installs_editable_when_pyproject_present(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\nversion='0.0.0'\n", encoding="utf-8")
    venv_dir = repo / ".venv"

    calls = []

    def fake_run(cmd, cwd=None, text=None, capture_output=None):
        calls.append(cmd)
        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()

    with patch("subprocess.run", side_effect=fake_run):
        res = setup_venv(
            repo=repo,
            venv_dir=venv_dir,
            python_executable="python",
            install=True,
            requirements_file=None,
            verify=False,
        )

    assert res.installed is True
    # ensure editable install attempted
    assert any("-e" in c for c in calls if isinstance(c, list))

