from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from lyra.safe_profile import prepare_isolated_run
from lyra.core import run_safe_profile


def test_prepare_isolated_run_copies_repo_and_writes_sitecustomize(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('x')\n", encoding="utf-8")

    iso = prepare_isolated_run(repo=repo, training_script=repo / "train.py")
    assert iso.isolated_repo.exists()
    assert iso.isolated_script.exists()
    assert iso.sitecustomize_path.exists()
    text = iso.sitecustomize_path.read_text(encoding="utf-8")
    assert "LYRA_MAX_STEPS" in text


def test_run_safe_profile_isolated_uses_isolated_cwd(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('x')\n", encoding="utf-8")

    with patch("subprocess.run") as run:
        run.return_value.returncode = 0
        run.return_value.stdout = ""
        run.return_value.stderr = ""

        result = run_safe_profile(repo, training_script="train.py", isolated=True)

    assert result.run_dir is not None

