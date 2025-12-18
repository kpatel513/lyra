from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_path(raw: str) -> Path:
    return Path(raw).expanduser().resolve()


def write_output_if_requested(text: str, output: Optional[str]) -> None:
    if not output:
        return
    out_path = Path(output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote report to: {out_path}")


