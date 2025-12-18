from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from ..optimize import optimize_repo
from .common import resolve_path, write_output_if_requested


@dataclass(frozen=True)
class OptimizeArgs:
    repo_path: str
    training_script: str | None
    max_steps: int
    apply: bool
    plan: bool
    yes: bool
    output_format: str
    output: str | None


def cmd_optimize(args: OptimizeArgs, *, project_root: Path) -> int:
    repo = resolve_path(args.repo_path)
    if not repo.exists():
        print(f"Error: repo_path does not exist: {repo}", file=sys.stderr)
        return 2

    try:
        report = optimize_repo(
            repo=repo,
            training_script=args.training_script,
            max_steps=args.max_steps,
            apply=args.apply,
            plan=args.plan,
            yes=args.yes,
            project_root=project_root,
        )
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if args.output_format == "json":
        text = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
        print(text, end="")
        write_output_if_requested(text, args.output)
        return 0

    lines = [
        "Lyra optimize",
        f"- repo: {report.repo}",
        f"- mode: {report.mode}",
        f"- before log: {report.before['profile']['log_file']}",
    ]
    if report.after:
        lines.append(f"- after log: {report.after['profile']['log_file']}")
    if report.diff:
        d = report.diff
        lines.append(
            f"- duration_s: {d['duration_s']['before']} -> {d['duration_s']['after']} (Δ {d['duration_s']['delta']})"
        )
        lines.append(f"- exit_reason: {d['exit_reason']['before']} -> {d['exit_reason']['after']}")
    if report.analysis_output:
        lines.append(f"- analysis output: {report.analysis_output}")
    if report.optimize_output:
        lines.append(f"- optimize output: {report.optimize_output}")
    if report.history_run_id:
        lines.append(f"- undo id: {report.history_run_id}")

    text = "\n".join(lines) + "\n"
    print(text, end="")
    write_output_if_requested(text, args.output)
    return 0


