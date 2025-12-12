"""
Core utilities for inspecting ML/AI training repositories.

This is intentionally lightweight for now and can be expanded into a
full agent-backed analysis module as Lyra evolves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


PYTHON_EXTENSIONS = {".py", ".pyw"}


@dataclass
class RepoSummary:
    root: Path
    python_files: int
    total_lines: int
    training_scripts: List[Path]

    def format_human(self) -> str:
        rel = lambda p: p.relative_to(self.root) if p.is_relative_to(self.root) else p
        training_list = "\n".join(f"  - {rel(p)}" for p in self.training_scripts) or "  (none detected)"

        return (
            "🎵 Lyra Repository Summary\n"
            f"✨ Root: {self.root}\n"
            f"📄 Python files: {self.python_files}\n"
            f"📏 Total lines of Python: {self.total_lines}\n"
            "🎯 Potential training entrypoints:\n"
            f"{training_list}"
        )


def _iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        # Skip common virtualenv and build dirs
        if any(part in {".venv", "venv", "__pycache__", "build", "dist"} for part in path.parts):
            continue
        yield path


def summarize_repo(root: Path) -> RepoSummary:
    """
    Produce a lightweight structural summary of a repository:
    - counts Python files
    - counts total lines
    - heuristically detects likely training scripts
    """
    root = root.resolve()
    python_files = list(_iter_python_files(root))

    total_lines = 0
    for f in python_files:
        try:
            with f.open("r", encoding="utf-8", errors="ignore") as fh:
                total_lines += sum(1 for _ in fh)
        except OSError:
            continue

    training_candidates: List[Path] = []
    for f in python_files:
        name = f.name.lower()
        if any(token in name for token in ("train", "fit", "finetune", "fine_tune", "downstream")):
            training_candidates.append(f)

    return RepoSummary(
        root=root,
        python_files=len(python_files),
        total_lines=total_lines,
        training_scripts=sorted(training_candidates),
    )


