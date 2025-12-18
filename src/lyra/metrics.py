from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ProfileMetrics:
    """
    Best-effort metrics derived from Lyra profiling output/logs.

    For pure-PyTorch runs, we currently infer:
    - whether we exited due to LYRA_MAX_STEPS
    - whether saving was disabled (torch.save patched)
    """

    exit_reason: Optional[str]  # e.g. "max_steps"
    max_steps: Optional[int]
    saving_disabled: Optional[bool]

    def to_dict(self) -> dict:
        return {
            "exit_reason": self.exit_reason,
            "max_steps": self.max_steps,
            "saving_disabled": self.saving_disabled,
        }


def parse_profile_log(log_file: Path) -> ProfileMetrics:
    """
    Parse Lyra profile logs for safe-profile markers emitted by sitecustomize.py.
    """
    try:
        text = log_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ProfileMetrics(exit_reason=None, max_steps=None, saving_disabled=None)

    exit_reason = None
    max_steps = None
    saving_disabled = None

    # Markers emitted by src/lyra/safe_profile.py sitecustomize:
    # [lyra-safe-profile] Reached LYRA_MAX_STEPS=123. Exiting.
    for line in text.splitlines():
        if "[lyra-safe-profile]" not in line:
            continue
        if "Reached LYRA_MAX_STEPS=" in line:
            exit_reason = "max_steps"
            try:
                rhs = line.split("Reached LYRA_MAX_STEPS=", 1)[1]
                num = ""
                for ch in rhs:
                    if ch.isdigit():
                        num += ch
                    else:
                        break
                max_steps = int(num) if num else None
            except Exception:
                max_steps = None
        if "torch.save disabled" in line or "torch.jit.save disabled" in line:
            saving_disabled = True

    return ProfileMetrics(exit_reason=exit_reason, max_steps=max_steps, saving_disabled=saving_disabled)


