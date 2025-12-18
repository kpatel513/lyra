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
import json
import sys
from pathlib import Path

from . import __version__
from .core import analyze_repo, run_safe_profile, summarize_repo
from .check import run_check
from .llm import ClaudeCodeRunner
from .metrics import parse_profile_log
from .optimize import optimize_repo
from .prompts import load_prompt, resolve_prompt
from .setup_env import detect_repo_deps, setup_conda, setup_venv


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
    if args.output_format == "json":
        payload = summary.to_dict()
        text = json.dumps(payload, indent=2, sort_keys=True)
        print(text)
        _write_output_if_requested(text + "\n", args.output)
    else:
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
    if args.output_format == "json":
        payload = report.to_dict()
        text = json.dumps(payload, indent=2, sort_keys=True)
        print(text)
        _write_output_if_requested(text + "\n", args.output)
    else:
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
            isolated=args.isolated,
            runs_root=Path(args.runs_root).expanduser().resolve() if args.runs_root else None,
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if args.output_format == "json":
        metrics = parse_profile_log(result.log_file).to_dict()
        payload = {"profile": result.to_dict(), "metrics": metrics}
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(result.format_human())
    return result.return_code


def _llm_execute(
    *,
    repo: Path,
    prompt: str,
    arguments: str,
    training_script: str,
    output_format: str,
    output: str | None,
) -> int:
    if not repo.exists():
        print(f"Error: --repo does not exist: {repo}", file=sys.stderr)
        return 2

    # Resolve and render prompt
    project_root = Path(__file__).resolve().parents[2]
    prompt_path = resolve_prompt(project_root, prompt)
    if not prompt_path.exists():
        print(f"Error: prompt not found: {prompt_path}", file=sys.stderr)
        return 2

    spec = load_prompt(prompt_path)
    rendered = spec.render(
        {
            "ARGUMENTS": arguments,
            "TRAINING_SCRIPT": training_script,
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
        output_format=output_format,
    )

    # Surface stderr if any (Claude sometimes uses stderr for warnings)
    if result.stderr.strip():
        print(result.stderr, file=sys.stderr)

    print(result.stdout)
    _write_output_if_requested(result.stdout, output)
    return result.return_code


def cmd_llm_run(args: argparse.Namespace) -> int:
    return _llm_execute(
        repo=_resolve_repo_path(args.repo),
        prompt=args.prompt,
        arguments=args.arguments,
        training_script=args.training_script,
        output_format=args.output_format,
        output=args.output,
    )


def cmd_llm_analyze(args: argparse.Namespace) -> int:
    return _llm_execute(
        repo=_resolve_repo_path(args.repo),
        prompt="lyraAnalyze",
        arguments=args.profile_file,
        training_script="",
        output_format=args.output_format,
        output=args.output,
    )


def cmd_llm_profile(args: argparse.Namespace) -> int:
    arguments = args.arguments or args.training_script
    return _llm_execute(
        repo=_resolve_repo_path(args.repo),
        prompt="lyraProfile",
        arguments=arguments,
        training_script=args.training_script,
        output_format=args.output_format,
        output=args.output,
    )


def cmd_llm_optimize(args: argparse.Namespace) -> int:
    return _llm_execute(
        repo=_resolve_repo_path(args.repo),
        prompt="lyraOptimize",
        arguments=args.analysis_file,
        training_script="",
        output_format=args.output_format,
        output=args.output,
    )


def cmd_setup(args: argparse.Namespace) -> int:
    repo = _resolve_repo_path(args.repo_path)
    if not repo.exists():
        print(f"Error: repo_path does not exist: {repo}", file=sys.stderr)
        return 2

    deps = detect_repo_deps(repo)
    env_name = args.environment_name or f"lyra-{repo.name}"

    if args.prefer == "conda":
        env_file = deps.get("environment.yml") or deps.get("environment.yaml")
        if not env_file:
            print("No environment.yml found; falling back to venv.", file=sys.stderr)
        else:
            result = setup_conda(
                repo=repo,
                env_name=env_name,
                env_file=env_file,
                install=not args.skip_install,
            )
            print(result.format_human())
            return 0 if result.installed or args.skip_install else 2

    python_exe = args.python_executable or sys.executable
    venv_dir = Path(args.venv_dir).expanduser() if args.venv_dir else (repo / ".venv")
    requirements_file = (
        Path(args.requirements).expanduser().resolve()
        if args.requirements
        else deps.get("requirements.txt")
    )

    result = setup_venv(
        repo=repo,
        venv_dir=venv_dir,
        python_executable=python_exe,
        install=not args.skip_install,
        requirements_file=requirements_file,
    )
    print(result.format_human())
    return 0 if result.env_path else 2


def cmd_check(args: argparse.Namespace) -> int:
    repo = _resolve_repo_path(args.repo) if args.repo else None
    report = run_check(repo=repo)
    if args.output_format == "json":
        payload = {
            "ok": report.ok,
            "items": [i.__dict__ for i in report.items],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(report.format_human())
    return 0 if report.ok else 2


def cmd_optimize(args: argparse.Namespace) -> int:
    repo = _resolve_repo_path(args.repo_path)
    if not repo.exists():
        print(f"Error: repo_path does not exist: {repo}", file=sys.stderr)
        return 2

    project_root = Path(__file__).resolve().parents[2]
    try:
        report = optimize_repo(
            repo=repo,
            training_script=args.training_script,
            max_steps=args.max_steps,
            apply=args.apply,
            plan=args.plan,
            project_root=project_root,
        )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if args.output_format == "json":
        text = json.dumps(report.to_dict(), indent=2, sort_keys=True)
        print(text)
        _write_output_if_requested(text + "\n", args.output)
    else:
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
            lines.append(
                f"- exit_reason: {d['exit_reason']['before']} -> {d['exit_reason']['after']}"
            )
        if report.analysis_output:
            lines.append(f"- analysis output: {report.analysis_output}")
        if report.optimize_output:
            lines.append(f"- optimize output: {report.optimize_output}")
        text = "\n".join(lines) + "\n"
        print(text)
        _write_output_if_requested(text, args.output)

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
    if args.command == "check":
        return cmd_check(args)
    if args.command == "optimize":
        return cmd_optimize(args)
    if args.command == "llm":
        if args.llm_command == "run":
            return cmd_llm_run(args)
        if args.llm_command == "analyze":
            return cmd_llm_analyze(args)
        if args.llm_command == "profile":
            return cmd_llm_profile(args)
        if args.llm_command == "optimize":
            return cmd_llm_optimize(args)
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


