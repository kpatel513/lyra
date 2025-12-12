"""
Lyra command-line interface.

This module mirrors the existing shell-based Lyra commands:
- lyra-summarize
- lyra-analyze
- lyra-profile
- lyra-setup

For now, the commands provide structured argument parsing and
human-friendly output. The deeper "agent" behavior that uses Claude
and project-specific prompts can be layered in behind these entry
points without changing the public CLI surface.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .core import analyze_repo, run_safe_profile, summarize_repo


def _add_common_repo_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "repo_path",
        type=str,
        help="Path to the ML/AI training repository to analyze/profile.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lyra",
        description=(
            "Lyra – AI agent for analyzing and optimizing ML training code "
            "before it runs on production clusters."
        ),
    )

    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show Lyra version and exit.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="subcommands",
        metavar="<command>",
        required=True,
    )

    # lyra summarize
    summarize = subparsers.add_parser(
        "summarize",
        help="Summarize an ML repository for mixed precision and sharding implementations.",
    )
    _add_common_repo_argument(summarize)

    # lyra analyze
    analyze = subparsers.add_parser(
        "analyze",
        help="Analyze training pipelines for performance bottlenecks.",
    )
    _add_common_repo_argument(analyze)

    # lyra profile
    profile = subparsers.add_parser(
        "profile",
        help="Safely profile training code to generate performance traces.",
    )
    _add_common_repo_argument(profile)
    profile.add_argument(
        "training_script",
        type=str,
        nargs="?",
        help="Optional training script to run (e.g. train.py). If omitted, Lyra will auto-detect later.",
    )
    profile.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum number of training steps to allow during safe profiling (default: 100).",
    )

    # lyra setup
    setup = subparsers.add_parser(
        "setup",
        help="Create an isolated environment suitable for safe profiling.",
    )
    _add_common_repo_argument(setup)
    setup.add_argument(
        "environment_name",
        type=str,
        nargs="?",
        help="Optional name for the environment. If omitted, Lyra will generate one.",
    )

    return parser


def _resolve_repo_path(raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    return path


def cmd_summarize(args: argparse.Namespace) -> int:
    repo = _resolve_repo_path(args.repo_path)
    summary = summarize_repo(repo)
    print(summary.format_human())
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    repo = _resolve_repo_path(args.repo_path)
    report = analyze_repo(repo)
    print(report.format_human())
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    repo = _resolve_repo_path(args.repo_path)
    result = run_safe_profile(
        root=repo,
        training_script=args.training_script,
        max_steps=args.max_steps,
    )
    print(result.format_human())
    return result.return_code


def cmd_setup(args: argparse.Namespace) -> int:
    repo = _resolve_repo_path(args.repo_path)
    env_name = args.environment_name or "<auto-generate>"
    print(
        f"🎵 Lyra (setup)\n"
        f"- Repository: {repo}\n"
        f"- Environment name: {env_name}\n"
        "\n"
        "This is the Python CLI scaffold. Environment detection, creation,\n"
        "and dependency installation will be implemented here."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``lyra`` command."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "summarize":
        return cmd_summarize(args)
    if args.command == "analyze":
        return cmd_analyze(args)
    if args.command == "profile":
        return cmd_profile(args)
    if args.command == "setup":
        return cmd_setup(args)

    parser.error(f"Unknown command: {args.command!r}")
    return 2


def main_summarize() -> int:
    """Entry point for the ``lyra-summarize`` script."""
    return main(["summarize", *sys.argv[1:]])


def main_analyze() -> int:
    """Entry point for the ``lyra-analyze`` script."""
    return main(["analyze", *sys.argv[1:]])


def main_profile() -> int:
    """Entry point for the ``lyra-profile`` script."""
    return main(["profile", *sys.argv[1:]])


def main_setup() -> int:
    """Entry point for the ``lyra-setup`` script."""
    return main(["setup", *sys.argv[1:]])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


