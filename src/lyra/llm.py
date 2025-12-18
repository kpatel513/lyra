from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class LlmResult:
    command: List[str]
    stdout: str
    stderr: str
    return_code: int


class ClaudeCodeRunner:
    """
    Minimal non-interactive wrapper around the Claude Code CLI.

    Uses:
      claude -p --output-format <fmt> "<prompt>"
    """

    def __init__(self, *, claude_executable: str = "claude") -> None:
        self.claude_executable = claude_executable

    def is_available(self) -> bool:
        return shutil.which(self.claude_executable) is not None

    def run(
        self,
        *,
        prompt: str,
        cwd: Optional[Path] = None,
        extra_args: Optional[List[str]] = None,
        output_format: str = "text",
    ) -> LlmResult:
        if output_format not in {"text", "json", "stream-json"}:
            raise ValueError("output_format must be 'text', 'json', or 'stream-json'")

        cmd: List[str] = [self.claude_executable, "-p", "--output-format", output_format]
        if extra_args:
            cmd.extend(extra_args)
        cmd.append(prompt)

        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
        )

        return LlmResult(
            command=cmd,
            stdout=proc.stdout,
            stderr=proc.stderr,
            return_code=proc.returncode,
        )


