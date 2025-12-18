from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SetupResult:
    repo: Path
    kind: str  # "venv" or "conda"
    env_path: Optional[Path]
    env_name: Optional[str]
    installed: bool
    details: str

    def format_human(self) -> str:
        lines = [
            "Lyra environment setup",
            f"- repo: {self.repo}",
            f"- type: {self.kind}",
        ]
        if self.env_path:
            lines.append(f"- path: {self.env_path}")
        if self.env_name:
            lines.append(f"- name: {self.env_name}")
        lines.append(f"- dependencies installed: {self.installed}")
        lines.append("")
        lines.append(self.details)
        return "\n".join(lines) + "\n"


def detect_repo_deps(repo: Path) -> dict[str, Path]:
    """
    Detect common dependency spec files in a training repo.
    """
    repo = repo.expanduser().resolve()
    found: dict[str, Path] = {}
    for name in ("requirements.txt", "pyproject.toml", "environment.yml", "environment.yaml"):
        p = repo / name
        if p.exists():
            found[name] = p
    return found


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)


def setup_venv(
    *,
    repo: Path,
    venv_dir: Path,
    python_executable: str,
    install: bool,
    requirements_file: Optional[Path],
) -> SetupResult:
    repo = repo.expanduser().resolve()
    venv_dir = venv_dir.expanduser().resolve()
    venv_python = venv_dir / "bin" / "python"
    venv_pip = venv_dir / "bin" / "pip"

    if not venv_dir.exists():
        proc = _run([python_executable, "-m", "venv", str(venv_dir)], cwd=repo)
        if proc.returncode != 0:
            return SetupResult(
                repo=repo,
                kind="venv",
                env_path=venv_dir,
                env_name=None,
                installed=False,
                details=f"Failed to create venv:\n{proc.stderr or proc.stdout}",
            )

    installed = False
    if install:
        # pip upgrade first
        proc = _run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], cwd=repo)
        if proc.returncode != 0:
            return SetupResult(
                repo=repo,
                kind="venv",
                env_path=venv_dir,
                env_name=None,
                installed=False,
                details=f"Venv created, but pip upgrade failed:\n{proc.stderr or proc.stdout}",
            )

        if requirements_file and requirements_file.exists():
            proc = _run([str(venv_pip), "install", "-r", str(requirements_file)], cwd=repo)
            if proc.returncode != 0:
                return SetupResult(
                    repo=repo,
                    kind="venv",
                    env_path=venv_dir,
                    env_name=None,
                    installed=False,
                    details=f"Venv created, but installing requirements failed:\n{proc.stderr or proc.stdout}",
                )
            installed = True

    details = (
        "Activate:\n"
        f"  source {venv_dir}/bin/activate\n"
        "\n"
        "Verify:\n"
        "  lyra check\n"
    )
    if requirements_file is None:
        details += "\nNote: no requirements file was selected for installation.\n"

    return SetupResult(
        repo=repo,
        kind="venv",
        env_path=venv_dir,
        env_name=None,
        installed=installed,
        details=details,
    )


def setup_conda(
    *,
    repo: Path,
    env_name: str,
    env_file: Path,
    install: bool,
    conda_executable: str = "conda",
) -> SetupResult:
    repo = repo.expanduser().resolve()
    env_file = env_file.expanduser().resolve()

    if shutil.which(conda_executable) is None:
        return SetupResult(
            repo=repo,
            kind="conda",
            env_path=None,
            env_name=env_name,
            installed=False,
            details=f"`{conda_executable}` not found in PATH.",
        )

    if not install:
        return SetupResult(
            repo=repo,
            kind="conda",
            env_path=None,
            env_name=env_name,
            installed=False,
            details=(
                "Detected conda + environment.yml, but install was skipped.\n"
                "To create env manually:\n"
                f"  conda env create -n {env_name} -f {env_file}\n"
                f"  conda activate {env_name}\n"
            ),
        )

    proc = _run([conda_executable, "env", "create", "-n", env_name, "-f", str(env_file)], cwd=repo)
    if proc.returncode != 0:
        return SetupResult(
            repo=repo,
            kind="conda",
            env_path=None,
            env_name=env_name,
            installed=False,
            details=f"Conda env create failed:\n{proc.stderr or proc.stdout}",
        )

    return SetupResult(
        repo=repo,
        kind="conda",
        env_path=None,
        env_name=env_name,
        installed=True,
        details=(
            "Activate:\n"
            f"  conda activate {env_name}\n"
            "\n"
            "Verify:\n"
            "  lyra check\n"
        ),
    )


