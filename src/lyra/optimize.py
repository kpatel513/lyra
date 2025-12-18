from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .core import run_safe_profile
from .history import create_history_entry, finalize_history_entry
from .llm import ClaudeCodeRunner
from .metrics import parse_profile_log
from .prompts import load_prompt, resolve_prompt


@dataclass(frozen=True)
class OptimizeReport:
    repo: Path
    mode: str  # "dry-run" | "plan" | "apply"
    before: dict
    after: Optional[dict]
    diff: Optional[dict]
    analysis_output: Optional[Path]
    optimize_output: Optional[Path]
    history_run_id: Optional[str]

    def to_dict(self) -> dict:
        return {
            "repo": str(self.repo),
            "mode": self.mode,
            "before": self.before,
            "after": self.after,
            "diff": self.diff,
            "analysis_output": str(self.analysis_output) if self.analysis_output else None,
            "optimize_output": str(self.optimize_output) if self.optimize_output else None,
            "history_run_id": self.history_run_id,
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
    plan: bool,
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

    if apply and plan:
        raise ValueError("Only one of apply/plan can be true.")

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
    diff = None
    history_run_id: Optional[str] = None

    if apply or plan:
        # Store outputs in the repo so they can be inspected later.
        out_dir = repo / ".lyra" / "optimize"
        analysis_output = out_dir / "analysis.txt"

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

        if apply:
            hist = create_history_entry(repo=repo, command="lyra optimize --apply")
            history_run_id = hist.run_id

            optimize_output = out_dir / "optimize.txt"
            _run_prompt_in_repo(
                repo=repo,
                project_root=project_root,
                prompt_name="lyraOptimize",
                arguments=str(analysis_output),
                training_script="",
                output_path=optimize_output,
                output_format="text",
            )

            finalize_history_entry(hist)

            after_res = run_safe_profile(
                root=repo,
                training_script=training_script,
                max_steps=max_steps,
                isolated=True,
            )
            after_metrics = parse_profile_log(after_res.log_file).to_dict()
            after = {"profile": after_res.to_dict(), "metrics": after_metrics}

            diff = _diff_before_after(before, after)

    return OptimizeReport(
        repo=repo,
        mode=("apply" if apply else ("plan" if plan else "dry-run")),
        before=before,
        after=after,
        diff=diff,
        analysis_output=analysis_output,
        optimize_output=optimize_output,
        history_run_id=history_run_id,
    )


def _diff_before_after(before: dict, after: dict) -> dict:
    """
    Compute a small, stable before/after diff for consumption by humans or tooling.
    """
    bprof = before.get("profile", {})
    aprof = after.get("profile", {})
    bmet = before.get("metrics", {})
    amet = after.get("metrics", {})

    bdur = bprof.get("duration_s")
    adur = aprof.get("duration_s")
    duration_delta = (adur - bdur) if isinstance(bdur, (int, float)) and isinstance(adur, (int, float)) else None

    return {
        "duration_s": {
            "before": bdur,
            "after": adur,
            "delta": duration_delta,
        },
        "exit_reason": {
            "before": bmet.get("exit_reason"),
            "after": amet.get("exit_reason"),
        },
        "max_steps": {
            "before": bmet.get("max_steps"),
            "after": amet.get("max_steps"),
        },
        "saving_disabled": {
            "before": bmet.get("saving_disabled"),
            "after": amet.get("saving_disabled"),
        },
    }


