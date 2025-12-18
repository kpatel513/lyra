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
from .llm import ClaudeCodeRunner
from .prompts import load_prompt, resolve_prompt


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
    summarize.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the report text to.",
    )

    # lyra analyze
    analyze = subparsers.add_parser(
        "analyze",
        help="Analyze training pipelines for performance bottlenecks.",
    )
    _add_common_repo_argument(analyze)
    analyze.add_argument(
        "--scan-all",
        action="store_true",
        help="Scan all Python files, not only likely training entrypoints.",
    )
    analyze.add_argument(
        "--engine",
        choices=["ast", "string"],
        default="ast",
        help="Analysis engine: AST-based (lower false positives) or legacy string scan.",
    )
    analyze.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the report text to.",
    )

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
    profile.add_argument(
        "--python",
        dest="python_executable",
        type=str,
        default=None,
        help="Override the Python executable used to run the training script.",
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

    # lyra llm
    llm = subparsers.add_parser(
        "llm",
        help="Run Claude Code prompts (commands/*.md) non-interactively.",
    )
    llm_sub = llm.add_subparsers(dest="llm_command", metavar="<llm_command>", required=True)

    llm_run = llm_sub.add_parser(
        "run",
        help="Run a prompt by name (from commands/) or by path.",
    )
    llm_run.add_argument(
        "prompt",
        type=str,
        help="Prompt name (e.g. lyraAnalyze) or path to a .md prompt file.",
    )
    llm_run.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Repository/workspace to run Claude Code in (sets cwd so tools can access the repo).",
    )
    llm_run.add_argument(
        "--arguments",
        type=str,
        default="",
        help="Value substituted for $ARGUMENTS in the prompt template.",
    )
    llm_run.add_argument(
        "--training-script",
        type=str,
        default="",
        help="Value substituted for $TRAINING_SCRIPT in the prompt template.",
    )
    llm_run.add_argument(
        "--output-format",
        choices=["text", "json", "stream-json"],
        default="text",
        help="Claude output format (passed to --output-format).",
    )
    llm_run.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write Claude output to.",
    )

    return parser


def _resolve_repo_path(raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    return path


def _write_output_if_requested(text: str, output: str | None) -> None:
    if not output:
        return
    out_path = Path(output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote report to: {out_path}")


def cmd_summarize(args: argparse.Namespace) -> int:
    repo = _resolve_repo_path(args.repo_path)
    if not repo.exists():
        print(f"Error: repo_path does not exist: {repo}", file=sys.stderr)
        return 2

    summary = summarize_repo(repo)
    text = summary.format_human()
    print(text)
    _write_output_if_requested(text, args.output)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    repo = _resolve_repo_path(args.repo_path)
    if not repo.exists():
        print(f"Error: repo_path does not exist: {repo}", file=sys.stderr)
        return 2

    report = analyze_repo(repo, scan_all_python_files=args.scan_all, engine=args.engine)
    text = report.format_human()
    print(text)
    _write_output_if_requested(text, args.output)
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    repo = _resolve_repo_path(args.repo_path)
    if not repo.exists():
        print(f"Error: repo_path does not exist: {repo}", file=sys.stderr)
        return 2

    try:
        result = run_safe_profile(
            root=repo,
            training_script=args.training_script,
            max_steps=args.max_steps,
            python_executable=args.python_executable,
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    print(result.format_human())
    return result.return_code


def cmd_llm_run(args: argparse.Namespace) -> int:
    repo = _resolve_repo_path(args.repo)
    if not repo.exists():
        print(f"Error: --repo does not exist: {repo}", file=sys.stderr)
        return 2

    # Resolve and render prompt
    project_root = Path(__file__).resolve().parents[2]
    prompt_path = resolve_prompt(project_root, args.prompt)
    if not prompt_path.exists():
        print(f"Error: prompt not found: {prompt_path}", file=sys.stderr)
        return 2

    spec = load_prompt(prompt_path)
    rendered = spec.render(
        {
            "ARGUMENTS": args.arguments,
            "TRAINING_SCRIPT": args.training_script,
        }
    ).strip()

    runner = ClaudeCodeRunner()
    if not runner.is_available():
        print("Error: Claude Code CLI not found (expected `claude` in PATH).", file=sys.stderr)
        return 2

    result = runner.run(
        prompt=rendered,
        cwd=repo,
        extra_args=spec.cli_args,
        output_format=args.output_format,
    )

    # Surface stderr if any (Claude sometimes uses stderr for warnings)
    if result.stderr.strip():
        print(result.stderr, file=sys.stderr)

    print(result.stdout)
    _write_output_if_requested(result.stdout, args.output)
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
    if args.command == "llm":
        if args.llm_command == "run":
            return cmd_llm_run(args)
        parser.error(f"Unknown llm command: {args.llm_command!r}")
        return 2

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


