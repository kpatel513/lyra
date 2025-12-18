from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PromptSpec:
    path: Path
    cli_args: List[str]
    template: str

    def render(self, variables: Optional[Dict[str, str]] = None) -> str:
        text = self.template
        if variables:
            for key, value in variables.items():
                text = text.replace(f"${key}", value)
        return text


def commands_dir(repo_root: Path) -> Path:
    return repo_root / "commands"


def load_prompt(prompt_path: Path) -> PromptSpec:
    """
    Load a prompt markdown file.

    Supports an optional first line of the form:
      args: --dangerously-skip-permissions --output-format json

    Everything after that line is treated as the prompt template.
    """
    prompt_path = prompt_path.expanduser().resolve()
    raw = prompt_path.read_text(encoding="utf-8", errors="ignore")
    lines = raw.splitlines()

    cli_args: List[str] = []
    start_idx = 0
    if lines and lines[0].strip().startswith("args:"):
        # split on whitespace after "args:"
        args_part = lines[0].split("args:", 1)[1].strip()
        cli_args = args_part.split() if args_part else []
        start_idx = 1

    template = "\n".join(lines[start_idx:]).lstrip("\n")
    return PromptSpec(path=prompt_path, cli_args=cli_args, template=template)


def resolve_prompt(repo_root: Path, name_or_path: str) -> Path:
    """
    Resolve either:
    - a direct path to a prompt file, or
    - a short name like "lyraAnalyze" / "lyraAnalyze.md" within commands/
    """
    p = Path(name_or_path).expanduser()
    if p.exists():
        return p.resolve()

    name = name_or_path
    if not name.endswith(".md"):
        name = f"{name}.md"
    candidate = commands_dir(repo_root) / name
    return candidate.resolve()


