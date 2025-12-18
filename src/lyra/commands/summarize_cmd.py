from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from ..core import summarize_repo
from .common import resolve_path, write_output_if_requested


@dataclass(frozen=True)
class SummarizeArgs:
    repo_path: str
    output: str | None
    output_format: str


def cmd_summarize(args: SummarizeArgs) -> int:
    repo = resolve_path(args.repo_path)
    if not repo.exists():
        print(f"Error: repo_path does not exist: {repo}", file=sys.stderr)
        return 2

    summary = summarize_repo(repo)
    if args.output_format == "json":
        text = json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n"
        print(text, end="")
        write_output_if_requested(text, args.output)
    else:
        text = summary.format_human()
        print(text, end="" if text.endswith("\n") else "\n")
        write_output_if_requested(text, args.output)
    return 0


