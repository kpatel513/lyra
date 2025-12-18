from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from ..setup_env import detect_repo_deps, setup_conda, setup_venv
from .common import resolve_path


@dataclass(frozen=True)
class SetupArgs:
    repo_path: str
    environment_name: str | None
    prefer: str
    python_executable: str | None
    venv_dir: str | None
    skip_install: bool
    requirements: str | None


def cmd_setup(args: SetupArgs) -> int:
    repo = resolve_path(args.repo_path)
    if not repo.exists():
        print(f"Error: repo_path does not exist: {repo}", file=sys.stderr)
        return 2

    deps = detect_repo_deps(repo)
    env_name = args.environment_name or f"lyra-{repo.name}"

    if args.prefer == "conda":
        env_file = deps.get("environment.yml") or deps.get("environment.yaml")
        if env_file:
            result = setup_conda(
                repo=repo,
                env_name=env_name,
                env_file=env_file,
                install=not args.skip_install,
            )
            print(result.format_human(), end="")
            return 0 if result.installed or args.skip_install else 2
        print("No environment.yml found; falling back to venv.", file=sys.stderr)

    python_exe = args.python_executable or sys.executable
    venv_dir = resolve_path(args.venv_dir) if args.venv_dir else (repo / ".venv").resolve()
    requirements_file = (
        resolve_path(args.requirements)
        if args.requirements
        else deps.get("requirements.txt")
    )

    result = setup_venv(
        repo=repo,
        venv_dir=venv_dir,
        python_executable=python_exe,
        install=not args.skip_install,
        requirements_file=requirements_file,
    )
    print(result.format_human(), end="")
    return 0 if result.env_path else 2


