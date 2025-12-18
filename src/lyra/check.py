from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class CheckItem:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class CheckReport:
    items: List[CheckItem]

    @property
    def ok(self) -> bool:
        return all(i.ok for i in self.items)

    def format_human(self) -> str:
        lines = ["🎵 Lyra Check", ""]
        for i in self.items:
            prefix = "✅" if i.ok else "❌"
            lines.append(f"{prefix} {i.name}: {i.detail}")
        lines.append("")
        lines.append("Status: ✅ healthy" if self.ok else "Status: ⚠️ needs attention")
        return "\n".join(lines) + "\n"


def run_check(*, repo: Optional[Path] = None, claude_executable: str = "claude") -> CheckReport:
    items: List[CheckItem] = []

    # Python version check
    py_ok = sys.version_info >= (3, 9)
    items.append(
        CheckItem(
            name="Python",
            ok=py_ok,
            detail=f"{sys.version.split()[0]} (required: >= 3.9)",
        )
    )

    # Claude Code availability
    claude_path = shutil.which(claude_executable)
    if claude_path is None:
        items.append(
            CheckItem(
                name="Claude Code CLI",
                ok=False,
                detail="`claude` not found in PATH (required only for `lyra llm ...`).",
            )
        )
    else:
        # Best-effort version probe (should be safe / non-interactive).
        try:
            proc = subprocess.run(
                [claude_executable, "--version"],
                capture_output=True,
                text=True,
            )
            ver = (proc.stdout or proc.stderr).strip() or "unknown"
            items.append(
                CheckItem(
                    name="Claude Code CLI",
                    ok=(proc.returncode == 0),
                    detail=f"found at {claude_path} ({ver})",
                )
            )
        except OSError as e:
            items.append(
                CheckItem(
                    name="Claude Code CLI",
                    ok=False,
                    detail=f"found at {claude_path} but failed to execute: {e}",
                )
            )

    # Repo checks (optional)
    if repo is not None:
        repo = repo.expanduser().resolve()
        if not repo.exists():
            items.append(
                CheckItem(
                    name="Repo path",
                    ok=False,
                    detail=f"does not exist: {repo}",
                )
            )
        else:
            items.append(
                CheckItem(
                    name="Repo path",
                    ok=True,
                    detail=str(repo),
                )
            )

            # Check we can create .lyra directory (logs, profile output, etc).
            lyra_dir = repo / ".lyra"
            try:
                lyra_dir.mkdir(parents=True, exist_ok=True)
                probe = lyra_dir / ".write_test"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink(missing_ok=True)
                items.append(
                    CheckItem(
                        name="Repo write access",
                        ok=True,
                        detail=f"can write to {lyra_dir}",
                    )
                )
            except OSError as e:
                items.append(
                    CheckItem(
                        name="Repo write access",
                        ok=False,
                        detail=f"cannot write to {lyra_dir}: {e}",
                    )
                )

    return CheckReport(items=items)


