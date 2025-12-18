from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from ..llm import ClaudeCodeRunner
from ..prompts import load_prompt, resolve_prompt
from .common import resolve_path, write_output_if_requested


@dataclass(frozen=True)
class LlmRunArgs:
    prompt: str
    repo: str
    arguments: str
    training_script: str
    output_format: str
    output: str | None


def _llm_execute(
    *,
    repo: Path,
    prompt: str,
    arguments: str,
    training_script: str,
    output_format: str,
    output: str | None,
    project_root: Path,
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

    result = runner.run(
        prompt=rendered,
        cwd=repo,
        extra_args=spec.cli_args,
        output_format=output_format,
    )

    if result.stderr.strip():
        print(result.stderr, file=sys.stderr)

    print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    write_output_if_requested(result.stdout, output)
    return result.return_code


def cmd_llm_run(args: LlmRunArgs, *, project_root: Path) -> int:
    return _llm_execute(
        repo=resolve_path(args.repo),
        prompt=args.prompt,
        arguments=args.arguments,
        training_script=args.training_script,
        output_format=args.output_format,
        output=args.output,
        project_root=project_root,
    )


@dataclass(frozen=True)
class LlmAnalyzeArgs:
    repo: str
    profile_file: str
    output_format: str
    output: str | None


def cmd_llm_analyze(args: LlmAnalyzeArgs, *, project_root: Path) -> int:
    return _llm_execute(
        repo=resolve_path(args.repo),
        prompt="lyraAnalyze",
        arguments=args.profile_file,
        training_script="",
        output_format=args.output_format,
        output=args.output,
        project_root=project_root,
    )


@dataclass(frozen=True)
class LlmProfileArgs:
    repo: str
    training_script: str
    arguments: str
    output_format: str
    output: str | None


def cmd_llm_profile(args: LlmProfileArgs, *, project_root: Path) -> int:
    arguments = args.arguments or args.training_script
    return _llm_execute(
        repo=resolve_path(args.repo),
        prompt="lyraProfile",
        arguments=arguments,
        training_script=args.training_script,
        output_format=args.output_format,
        output=args.output,
        project_root=project_root,
    )


@dataclass(frozen=True)
class LlmOptimizeArgs:
    repo: str
    analysis_file: str
    output_format: str
    output: str | None


def cmd_llm_optimize(args: LlmOptimizeArgs, *, project_root: Path) -> int:
    return _llm_execute(
        repo=resolve_path(args.repo),
        prompt="lyraOptimize",
        arguments=args.analysis_file,
        training_script="",
        output_format=args.output_format,
        output=args.output,
        project_root=project_root,
    )


