from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..check import run_check
from .common import resolve_path


@dataclass(frozen=True)
class CheckArgs:
    repo: str | None
    output_format: str


def cmd_check(args: CheckArgs) -> int:
    repo_path = resolve_path(args.repo) if args.repo else None
    report = run_check(repo=repo_path)
    if args.output_format == "json":
        payload = {"ok": report.ok, "items": [i.__dict__ for i in report.items]}
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(report.format_human(), end="")
    return 0 if report.ok else 2


