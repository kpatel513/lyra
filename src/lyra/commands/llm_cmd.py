from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from ..llm import ClaudeCodeRunner
from ..prompts import load_prompt, resolve_prompt
from .common import resolve_path, write_output_if_requested


def _build_user_llm_args(args: argparse.Namespace) -> list[str]:
    extra: list[str] = []
    if getattr(args, "dangerously_skip_permissions", False):
        extra.append("--dangerously-skip-permissions")
    model = getattr(args, "model", None)
    if model:
        extra.extend(["--model", model])
    pm = getattr(args, "permission_mode", None)
    if pm:
        extra.extend(["--permission-mode", pm])
    allowed = getattr(args, "allowed_tools", None) or []
    for item in allowed:
        extra.extend(["--allowed-tools", item])
    disallowed = getattr(args, "disallowed_tools", None) or []
    for item in disallowed:
        extra.extend(["--disallowed-tools", item])
    return extra


def _llm_execute(
    *,
    repo: Path,
    prompt: str,
    arguments: str,
    training_script: str,
    output_format: str,
    output: Optional[str],
    project_root: Path,
    user_extra_args: Optional[list[str]] = None,
) -> int:
    if not repo.exists():
        print(f"Error: --repo does not exist: {repo}", file=sys.stderr)
        return 2

    prompt_path = resolve_prompt(project_root, prompt)
    if not prompt_path.exists():
        print(f"Error: prompt not found: {prompt_path}", file=sys.stderr)
        return 2

    spec = load_prompt(prompt_path)
    rendered = spec.render({"ARGUMENTS": arguments, "TRAINING_SCRIPT": training_script}).strip()

    runner = ClaudeCodeRunner()
    if not runner.is_available():
        print("Error: Claude Code CLI not found (expected `claude` in PATH).", file=sys.stderr)
        return 2

    merged_args = list(spec.cli_args)
    if user_extra_args:
        merged_args.extend(user_extra_args)

    result = runner.run(
        prompt=rendered,
        cwd=repo,
        extra_args=merged_args,
        output_format=output_format,
    )

    if result.stderr.strip():
        print(result.stderr, file=sys.stderr)

    print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    write_output_if_requested(result.stdout, output)
    return result.return_code


def cmd_llm_run(args: argparse.Namespace, *, project_root: Path) -> int:
    return _llm_execute(
        repo=resolve_path(args.repo),
        prompt=args.prompt,
        arguments=args.arguments,
        training_script=args.training_script,
        output_format=args.output_format,
        output=args.output,
        project_root=project_root,
        user_extra_args=_build_user_llm_args(args),
    )


def cmd_llm_analyze(args: argparse.Namespace, *, project_root: Path) -> int:
    return _llm_execute(
        repo=resolve_path(args.repo),
        prompt="lyraAnalyze",
        arguments=args.profile_file,
        training_script="",
        output_format=args.output_format,
        output=args.output,
        project_root=project_root,
        user_extra_args=_build_user_llm_args(args),
    )


def cmd_llm_profile(args: argparse.Namespace, *, project_root: Path) -> int:
    arguments = args.arguments or args.training_script
    return _llm_execute(
        repo=resolve_path(args.repo),
        prompt="lyraProfile",
        arguments=arguments,
        training_script=args.training_script,
        output_format=args.output_format,
        output=args.output,
        project_root=project_root,
        user_extra_args=_build_user_llm_args(args),
    )


def cmd_llm_optimize(args: argparse.Namespace, *, project_root: Path) -> int:
    return _llm_execute(
        repo=resolve_path(args.repo),
        prompt="lyraOptimize",
        arguments=args.analysis_file,
        training_script="",
        output_format=args.output_format,
        output=args.output,
        project_root=project_root,
        user_extra_args=_build_user_llm_args(args),
    )


