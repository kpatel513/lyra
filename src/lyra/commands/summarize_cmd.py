from __future__ import annotations

import argparse
import json
import sys

from ..core import summarize_repo
from .common import resolve_path, write_output_if_requested


def cmd_summarize(args: argparse.Namespace) -> int:
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


