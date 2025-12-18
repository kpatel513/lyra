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
from .commands.analyze_cmd import AnalyzeArgs, cmd_analyze
from .commands.check_cmd import CheckArgs, cmd_check
from .commands.llm_cmd import (
    LlmAnalyzeArgs,
    LlmOptimizeArgs,
    LlmProfileArgs,
    LlmRunArgs,
    cmd_llm_analyze,
    cmd_llm_optimize,
    cmd_llm_profile,
    cmd_llm_run,
)
from .commands.optimize_cmd import OptimizeArgs, cmd_optimize
from .commands.profile_cmd import ProfileArgs, cmd_profile
from .commands.setup_cmd import SetupArgs, cmd_setup
from .commands.summarize_cmd import SummarizeArgs, cmd_summarize
from .commands.undo_cmd import (
    UndoApplyArgs,
    UndoLastArgs,
    UndoListArgs,
    cmd_undo_apply,
    cmd_undo_last,
    cmd_undo_list,
)


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
    summarize.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format for stdout (default: text).",
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
    analyze.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format for stdout (default: text).",
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
    profile.add_argument(
        "--isolated",
        action="store_true",
        default=True,
        help="Run profiling in an isolated copy under .lyra/runs (default: enabled).",
    )
    profile.add_argument(
        "--no-isolated",
        action="store_false",
        dest="isolated",
        help="Run profiling in-place (not recommended).",
    )
    profile.add_argument(
        "--runs-root",
        type=str,
        default=None,
        help="Override where isolated runs are created (default: <repo>/.lyra/runs).",
    )
    profile.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format for stdout (default: text).",
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
    setup.add_argument(
        "--prefer",
        choices=["venv", "conda"],
        default="venv",
        help="Preferred environment type to create (default: venv).",
    )
    setup.add_argument(
        "--python",
        dest="python_executable",
        type=str,
        default=None,
        help="Python executable to use for venv creation (default: current interpreter).",
    )
    setup.add_argument(
        "--venv-dir",
        type=str,
        default=None,
        help="Where to create the venv (default: <repo>/.venv).",
    )
    setup.add_argument(
        "--skip-install",
        action="store_true",
        help="Create the environment but do not install dependencies.",
    )
    setup.add_argument(
        "--requirements",
        type=str,
        default=None,
        help="Explicit requirements file to install (default: <repo>/requirements.txt if present).",
    )

    # lyra check
    check = subparsers.add_parser(
        "check",
        help="Check Lyra prerequisites (Python/Claude) and optional repo write access.",
    )
    check.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Optional repo/workspace path to validate write access for .lyra/ outputs.",
    )
    check.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format for stdout (default: text).",
    )

    # NOTE: intentionally no legacy alias.

    # lyra llm
    llm = subparsers.add_parser(
        "llm",
        help="Run Claude Code prompts (commands/*.md) non-interactively.",
    )
    llm_sub = llm.add_subparsers(dest="llm_command", metavar="<llm_command>", required=True)

    llm_analyze = llm_sub.add_parser(
        "analyze",
        help="Convenience wrapper for commands/lyraAnalyze.md (analyze a profiler report).",
    )
    llm_analyze.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Repository/workspace to run Claude Code in (sets cwd so tools can access the repo).",
    )
    llm_analyze.add_argument(
        "--profile-file",
        type=str,
        required=True,
        help="Path to the profiling report file (substituted into $ARGUMENTS).",
    )
    llm_analyze.add_argument(
        "--output-format",
        choices=["text", "json", "stream-json"],
        default="text",
        help="Claude output format (passed to --output-format).",
    )
    llm_analyze.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write Claude output to.",
    )

    llm_profile = llm_sub.add_parser(
        "profile",
        help="Convenience wrapper for commands/lyraProfile.md (patch training script for profiling).",
    )
    llm_profile.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Repository/workspace to run Claude Code in (sets cwd so tools can access the repo).",
    )
    llm_profile.add_argument(
        "--training-script",
        type=str,
        required=True,
        help="Training script path (substituted into $TRAINING_SCRIPT and, if needed, $ARGUMENTS).",
    )
    llm_profile.add_argument(
        "--arguments",
        type=str,
        default="",
        help="Optional value substituted for $ARGUMENTS. If omitted, defaults to --training-script.",
    )
    llm_profile.add_argument(
        "--output-format",
        choices=["text", "json", "stream-json"],
        default="text",
        help="Claude output format (passed to --output-format).",
    )
    llm_profile.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write Claude output to.",
    )

    llm_optimize = llm_sub.add_parser(
        "optimize",
        help="Convenience wrapper for commands/lyraOptimize.md (apply suggestions from an analysis doc).",
    )
    llm_optimize.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Repository/workspace to run Claude Code in (sets cwd so tools can access the repo).",
    )
    llm_optimize.add_argument(
        "--analysis-file",
        type=str,
        required=True,
        help="Path to the analysis document (substituted into $ARGUMENTS).",
    )
    llm_optimize.add_argument(
        "--output-format",
        choices=["text", "json", "stream-json"],
        default="text",
        help="Claude output format (passed to --output-format).",
    )
    llm_optimize.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write Claude output to.",
    )

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

    # lyra optimize
    optimize = subparsers.add_parser(
        "optimize",
        help="Orchestrate: profile -> (optional LLM) optimize -> re-profile.",
    )
    _add_common_repo_argument(optimize)
    optimize.add_argument(
        "training_script",
        type=str,
        nargs="?",
        help="Optional training script to run (default: auto-detect).",
    )
    optimize.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Max optimizer steps for each profile run (default: 100).",
    )
    optimize.add_argument(
        "--apply",
        action="store_true",
        help="Actually run Claude analyze+optimize prompts (may modify repo). Default is dry-run (profile only).",
    )
    optimize.add_argument(
        "--yes",
        action="store_true",
        help="Acknowledge and proceed with applying edits when using --apply.",
    )
    optimize.add_argument(
        "--plan",
        action="store_true",
        help="Run profile + Claude analysis only (no code changes, no re-profile).",
    )
    optimize.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format for stdout (default: text).",
    )
    optimize.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the report output to.",
    )

    # lyra undo
    undo = subparsers.add_parser(
        "undo",
        help="Revert repo changes made by `lyra optimize --apply` using .lyra/history snapshots.",
    )
    undo_sub = undo.add_subparsers(dest="undo_command", metavar="<undo_command>", required=True)

    undo_list = undo_sub.add_parser("list", help="List available undo snapshots.")
    undo_list.add_argument("--repo", type=str, required=True, help="Target repo containing .lyra/history.")

    undo_last = undo_sub.add_parser("last", help="Undo the most recent snapshot.")
    undo_last.add_argument("--repo", type=str, required=True, help="Target repo containing .lyra/history.")
    undo_last.add_argument("--force", action="store_true", help="Overwrite even if files changed since run.")

    undo_apply = undo_sub.add_parser("apply", help="Undo a specific snapshot by run-id.")
    undo_apply.add_argument("--repo", type=str, required=True, help="Target repo containing .lyra/history.")
    undo_apply.add_argument("run_id", type=str, help="History run id (directory name under .lyra/history).")
    undo_apply.add_argument("--force", action="store_true", help="Overwrite even if files changed since run.")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``lyra`` command."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]

    if args.command == "summarize":
        return cmd_summarize(SummarizeArgs(args.repo_path, args.output, args.output_format))
    if args.command == "analyze":
        return cmd_analyze(
            AnalyzeArgs(args.repo_path, args.scan_all, args.engine, args.output, args.output_format)
        )
    if args.command == "profile":
        return cmd_profile(
            ProfileArgs(
                repo_path=args.repo_path,
                training_script=args.training_script,
                max_steps=args.max_steps,
                python_executable=args.python_executable,
                isolated=args.isolated,
                runs_root=args.runs_root,
                output_format=args.output_format,
            )
        )
    if args.command == "setup":
        return cmd_setup(
            SetupArgs(
                repo_path=args.repo_path,
                environment_name=args.environment_name,
                prefer=args.prefer,
                python_executable=args.python_executable,
                venv_dir=args.venv_dir,
                skip_install=args.skip_install,
                requirements=args.requirements,
            )
        )
    if args.command == "check":
        return cmd_check(CheckArgs(args.repo, args.output_format))
    if args.command == "optimize":
        return cmd_optimize(
            OptimizeArgs(
                repo_path=args.repo_path,
                training_script=args.training_script,
                max_steps=args.max_steps,
                apply=args.apply,
                plan=args.plan,
                yes=args.yes,
                output_format=args.output_format,
                output=args.output,
            ),
            project_root=project_root,
        )
    if args.command == "undo":
        if args.undo_command == "list":
            return cmd_undo_list(UndoListArgs(args.repo))
        if args.undo_command == "last":
            return cmd_undo_last(UndoLastArgs(args.repo, args.force))
        if args.undo_command == "apply":
            return cmd_undo_apply(UndoApplyArgs(args.repo, args.run_id, args.force))
        parser.error(f"Unknown undo command: {args.undo_command!r}")
        return 2
    if args.command == "llm":
        if args.llm_command == "run":
            return cmd_llm_run(
                LlmRunArgs(
                    prompt=args.prompt,
                    repo=args.repo,
                    arguments=args.arguments,
                    training_script=args.training_script,
                    output_format=args.output_format,
                    output=args.output,
                ),
                project_root=project_root,
            )
        if args.llm_command == "analyze":
            return cmd_llm_analyze(
                LlmAnalyzeArgs(
                    repo=args.repo,
                    profile_file=args.profile_file,
                    output_format=args.output_format,
                    output=args.output,
                ),
                project_root=project_root,
            )
        if args.llm_command == "profile":
            return cmd_llm_profile(
                LlmProfileArgs(
                    repo=args.repo,
                    training_script=args.training_script,
                    arguments=args.arguments,
                    output_format=args.output_format,
                    output=args.output,
                ),
                project_root=project_root,
            )
        if args.llm_command == "optimize":
            return cmd_llm_optimize(
                LlmOptimizeArgs(
                    repo=args.repo,
                    analysis_file=args.analysis_file,
                    output_format=args.output_format,
                    output=args.output,
                ),
                project_root=project_root,
            )
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


