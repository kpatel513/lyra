from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class SetupResult:
    repo: Path
    kind: str  # "venv" or "conda"
    env_path: Optional[Path]
    env_name: Optional[str]
    installed: bool
    verified: bool
    verify_report: Optional[Dict[str, Any]]
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
        if self.verify_report is not None:
            lines.append(f"- verify: {'ok' if self.verified else 'failed'}")
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


def _verify_env(python_exe: Path, *, cwd: Path) -> Dict[str, Any]:
    """
    Best-effort import checks inside the created environment.
    This does not install anything; it only reports what works.
    """
    snippet = r"""
import json
out = {"python": None, "torch": None, "numpy": None}
import sys
out["python"] = sys.version.split()[0]
try:
    import torch
    out["torch"] = {"version": getattr(torch, "__version__", None), "cuda_available": torch.cuda.is_available()}
except Exception as e:
    out["torch"] = {"error": str(e)}
try:
    import numpy as np
    out["numpy"] = {"version": getattr(np, "__version__", None)}
except Exception as e:
    out["numpy"] = {"error": str(e)}
print(json.dumps(out))
"""
    proc = _run([str(python_exe), "-c", snippet], cwd=cwd)
    report: Dict[str, Any] = {"return_code": proc.returncode, "raw": (proc.stdout or "").strip()}
    if proc.returncode == 0:
        try:
            import json as _json

            report["parsed"] = _json.loads(report["raw"] or "{}")
        except Exception:
            report["parsed"] = None
    else:
        report["stderr"] = proc.stderr
    return report


def setup_venv(
    *,
    repo: Path,
    venv_dir: Path,
    python_executable: str,
    install: bool,
    requirements_file: Optional[Path],
    editable: bool = True,
    verify: bool = False,
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
                verified=False,
                verify_report=None,
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
                verified=False,
                verify_report=None,
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
                    verified=False,
                    verify_report=None,
                    details=f"Venv created, but installing requirements failed:\n{proc.stderr or proc.stdout}",
                )
            installed = True
        else:
            # Try installing from pyproject.toml if present (common for modern repos).
            pyproject = repo / "pyproject.toml"
            if pyproject.exists():
                cmd = [str(venv_python), "-m", "pip", "install"]
                if editable:
                    cmd.append("-e")
                cmd.append(str(repo))
                proc = _run(cmd, cwd=repo)
                if proc.returncode != 0:
                    return SetupResult(
                        repo=repo,
                        kind="venv",
                        env_path=venv_dir,
                        env_name=None,
                        installed=False,
                        verified=False,
                        verify_report=None,
                        details=(
                            "Venv created, but installing from pyproject.toml failed.\n"
                            "If this repo is not a Python package, create a requirements.txt and rerun.\n\n"
                            f"{proc.stderr or proc.stdout}"
                        ),
                    )
                installed = True

    verify_report = None
    verified = False
    if verify:
        verify_report = _verify_env(venv_python, cwd=repo)
        parsed = verify_report.get("parsed") if isinstance(verify_report, dict) else None
        # Consider "verified" if python ran and torch import did not error (if present).
        if isinstance(parsed, dict):
            torch_info = parsed.get("torch")
            if isinstance(torch_info, dict) and "error" not in torch_info:
                verified = True
            else:
                verified = False
        else:
            verified = False

    details = (
        "Activate:\n"
        f"  source {venv_dir}/bin/activate\n"
        "\n"
        "Verify:\n"
        "  lyra check\n"
    )
    if requirements_file is None:
        details += "\nNote: no requirements file was selected for installation.\n"
    if verify_report is not None:
        details += "\nEnvironment verify report (raw JSON):\n" + (verify_report.get("raw", "") if isinstance(verify_report, dict) else "") + "\n"

    return SetupResult(
        repo=repo,
        kind="venv",
        env_path=venv_dir,
        env_name=None,
        installed=installed,
        verified=verified,
        verify_report=verify_report,
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
            verified=False,
            verify_report=None,
            details=f"`{conda_executable}` not found in PATH.",
        )

    if not install:
        return SetupResult(
            repo=repo,
            kind="conda",
            env_path=None,
            env_name=env_name,
            installed=False,
            verified=False,
            verify_report=None,
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
            verified=False,
            verify_report=None,
            details=f"Conda env create failed:\n{proc.stderr or proc.stdout}",
        )

    return SetupResult(
        repo=repo,
        kind="conda",
        env_path=None,
        env_name=env_name,
        installed=True,
        verified=False,
        verify_report=None,
        details=(
            "Activate:\n"
            f"  conda activate {env_name}\n"
            "\n"
            "Verify:\n"
            "  lyra check\n"
        ),
    )


