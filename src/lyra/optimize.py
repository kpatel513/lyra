from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .core import run_safe_profile
from .llm import ClaudeCodeRunner
from .metrics import parse_profile_log
from .prompts import load_prompt, resolve_prompt


@dataclass(frozen=True)
class OptimizeReport:
    repo: Path
    applied: bool
    before: dict
    after: Optional[dict]
    analysis_output: Optional[Path]
    optimize_output: Optional[Path]

    def to_dict(self) -> dict:
        return {
            "repo": str(self.repo),
            "applied": self.applied,
            "before": self.before,
            "after": self.after,
            "analysis_output": str(self.analysis_output) if self.analysis_output else None,
            "optimize_output": str(self.optimize_output) if self.optimize_output else None,
        }


def _run_prompt_in_repo(
    *,
    repo: Path,
    project_root: Path,
    prompt_name: str,
    arguments: str,
    training_script: str,
    output_path: Path,
    output_format: str = "text",
) -> None:
    prompt_path = resolve_prompt(project_root, prompt_name)
    spec = load_prompt(prompt_path)
    rendered = spec.render({"ARGUMENTS": arguments, "TRAINING_SCRIPT": training_script}).strip()

    runner = ClaudeCodeRunner()
    if not runner.is_available():
        raise RuntimeError("Claude Code CLI not found (`claude` not in PATH).")

    res = runner.run(
        prompt=rendered,
        cwd=repo,
        extra_args=spec.cli_args,
        output_format=output_format,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(res.stdout, encoding="utf-8")
    if res.return_code != 0:
        raise RuntimeError(f"Claude returned non-zero exit code {res.return_code}. See {output_path}.")


def optimize_repo(
    *,
    repo: Path,
    training_script: Optional[str],
    max_steps: int,
    apply: bool,
    project_root: Path,
) -> OptimizeReport:
    """
    Minimal orchestration:
    - profile (before)
    - (optional) Claude analyze + optimize
    - profile (after)

    `apply=False` is safe: it only profiles and returns a report scaffold.
    """
    repo = repo.expanduser().resolve()

    before_res = run_safe_profile(
        root=repo,
        training_script=training_script,
        max_steps=max_steps,
        isolated=True,
    )
    before_metrics = parse_profile_log(before_res.log_file).to_dict()
    before = {"profile": before_res.to_dict(), "metrics": before_metrics}

    analysis_output = None
    optimize_output = None
    after = None

    if apply:
        # Store outputs in the repo so they can be inspected later.
        out_dir = repo / ".lyra" / "optimize"
        analysis_output = out_dir / "analysis.txt"
        optimize_output = out_dir / "optimize.txt"

        # Use existing prompt assets. We pass the profile log as $ARGUMENTS to analyze.
        _run_prompt_in_repo(
            repo=repo,
            project_root=project_root,
            prompt_name="lyraAnalyze",
            arguments=str(before_res.log_file),
            training_script="",
            output_path=analysis_output,
            output_format="text",
        )

        _run_prompt_in_repo(
            repo=repo,
            project_root=project_root,
            prompt_name="lyraOptimize",
            arguments=str(analysis_output),
            training_script="",
            output_path=optimize_output,
            output_format="text",
        )

        after_res = run_safe_profile(
            root=repo,
            training_script=training_script,
            max_steps=max_steps,
            isolated=True,
        )
        after_metrics = parse_profile_log(after_res.log_file).to_dict()
        after = {"profile": after_res.to_dict(), "metrics": after_metrics}

    return OptimizeReport(
        repo=repo,
        applied=apply,
        before=before,
        after=after,
        analysis_output=analysis_output,
        optimize_output=optimize_output,
    )


