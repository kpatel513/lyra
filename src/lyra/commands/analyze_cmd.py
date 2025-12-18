from __future__ import annotations

import argparse
import json
import sys

from ..core import analyze_repo
from .common import resolve_path, write_output_if_requested


def cmd_analyze(args: argparse.Namespace) -> int:
    repo = resolve_path(args.repo_path)
    if not repo.exists():
        print(f"Error: repo_path does not exist: {repo}", file=sys.stderr)
        return 2

    report = analyze_repo(repo, scan_all_python_files=args.scan_all, engine=args.engine)
    if args.output_format == "json":
        text = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
        print(text, end="")
        write_output_if_requested(text, args.output)
    else:
        text = report.format_human()
        print(text, end="" if text.endswith("\n") else "\n")
        write_output_if_requested(text, args.output)
    return 0


