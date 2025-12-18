from __future__ import annotations

from pathlib import Path

from lyra.core import analyze_repo, run_safe_profile, summarize_repo


def _write(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def test_summarize_repo_detects_training_script(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    _write(repo / "train.py", "print('hello')\n")
    _write(repo / "utils.py", "def x():\n    return 1\n")

    summary = summarize_repo(repo)
    assert summary.python_files == 2
    assert summary.total_lines >= 3
    assert any(p.name == "train.py" for p in summary.training_scripts)


def test_analyze_repo_detects_mixed_precision_and_ddp(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    _write(
        repo / "train_model.py",
        """
import torch
from torch.nn.parallel import DistributedDataParallel

scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    pass

model = DistributedDataParallel(torch.nn.Linear(1, 1))
""".lstrip(),
    )

    report = analyze_repo(repo, engine="ast")
    kinds = {f.kind for f in report.findings}
    assert "mixed_precision_grad_scaler" in kinds
    assert "mixed_precision_autocast" in kinds
    assert "ddp_torch" in kinds


def test_analyze_repo_scan_all(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    # No training-named file, so this would be missed without scan_all.
    _write(
        repo / "entrypoint.py",
        "from torch.nn.parallel import DistributedDataParallel\n",
    )
    report = analyze_repo(repo, scan_all_python_files=True, engine="ast")
    assert any(f.kind == "ddp_torch" for f in report.findings)


def test_analyze_repo_engine_string(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    _write(repo / "train.py", "with torch.cuda.amp.autocast():\n    pass\n")
    report = analyze_repo(repo, engine="string")
    assert any(f.engine == "string" for f in report.findings)


def test_run_safe_profile_writes_log_and_returns_code(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    # Minimal script that exits successfully and prints env vars.
    _write(
        repo / "train.py",
        """
import os
print("SAFE", os.environ.get("LYRA_SAFE_PROFILE"))
print("MAX_STEPS", os.environ.get("LYRA_MAX_STEPS"))
print("DISABLE_SAVING", os.environ.get("LYRA_DISABLE_SAVING"))
""".lstrip(),
    )

    result = run_safe_profile(repo, training_script="train.py", max_steps=7, isolated=False)
    assert result.return_code == 0
    assert result.log_file.exists()
    text = result.log_file.read_text(encoding="utf-8")
    assert "SAFE 1" in text
    assert "MAX_STEPS 7" in text
    assert "DISABLE_SAVING 1" in text


