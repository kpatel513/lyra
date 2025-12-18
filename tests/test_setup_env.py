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
        )

    assert res.kind == "venv"
    assert res.env_path == venv_dir.resolve()

