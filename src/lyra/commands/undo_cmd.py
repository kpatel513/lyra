from __future__ import annotations

import sys
import argparse
import json

from ..history import list_history, undo_history
from .common import resolve_path


def cmd_undo_list(args: argparse.Namespace) -> int:
    repo = resolve_path(args.repo)
    items = list_history(repo)
    print(json.dumps(items, indent=2, sort_keys=True))
    return 0


def cmd_undo_last(args: argparse.Namespace) -> int:
    repo = resolve_path(args.repo)
    items = list_history(repo)
    if not items:
        print("No history entries found.", file=sys.stderr)
        return 2
    run_id = items[0].get("run_id")
    if not run_id:
        print("History metadata missing run_id.", file=sys.stderr)
        return 2
    try:
        summary = undo_history(repo=repo, run_id=run_id, force=args.force)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    print(f"Reverted snapshot: {run_id}")
    print(
        f"Restored: {len(summary['restored'])} | Removed: {len(summary['removed'])} | Skipped (no backup): {len(summary['skipped_no_backup'])}"
    )
    return 0


def cmd_undo_apply(args: argparse.Namespace) -> int:
    repo = resolve_path(args.repo)
    try:
        summary = undo_history(repo=repo, run_id=args.run_id, force=args.force)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    print(f"Reverted snapshot: {args.run_id}")
    print(
        f"Restored: {len(summary['restored'])} | Removed: {len(summary['removed'])} | Skipped (no backup): {len(summary['skipped_no_backup'])}"
    )
    return 0


