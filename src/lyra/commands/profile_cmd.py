from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from ..core import run_safe_profile
from ..metrics import parse_profile_log
from .common import resolve_path


@dataclass(frozen=True)
class ProfileArgs:
    repo_path: str
    training_script: str | None
    max_steps: int
    python_executable: str | None
    isolated: bool
    runs_root: str | None
    output_format: str


def cmd_profile(args: ProfileArgs) -> int:
    repo = resolve_path(args.repo_path)
    if not repo.exists():
        print(f"Error: repo_path does not exist: {repo}", file=sys.stderr)
        return 2

    try:
        result = run_safe_profile(
            root=repo,
            training_script=args.training_script,
            max_steps=args.max_steps,
            python_executable=args.python_executable,
            isolated=args.isolated,
            runs_root=resolve_path(args.runs_root) if args.runs_root else None,
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if args.output_format == "json":
        metrics = parse_profile_log(result.log_file).to_dict()
        payload = {"profile": result.to_dict(), "metrics": metrics}
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(result.format_human(), end="")

    return result.return_code


