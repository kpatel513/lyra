from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


SAFE_PROFILE_SITECUSTOMIZE = r"""
import os
import sys


def _truthy(v: str | None) -> bool:
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


LYRA_SAFE = _truthy(os.environ.get("LYRA_SAFE_PROFILE"))
if not LYRA_SAFE:
    # Do nothing unless explicitly enabled.
    raise SystemExit(0)


_MAX_STEPS_RAW = os.environ.get("LYRA_MAX_STEPS", "100")
try:
    LYRA_MAX_STEPS = int(_MAX_STEPS_RAW)
except Exception:
    LYRA_MAX_STEPS = 100

LYRA_DISABLE_SAVING = _truthy(os.environ.get("LYRA_DISABLE_SAVING"))

_warned_save = False


def _warn_once(msg: str) -> None:
    global _warned_save
    if _warned_save:
        return
    _warned_save = True
    print(f"[lyra-safe-profile] {msg}", file=sys.stderr)


def _patch_torch() -> None:
    # Import lazily; many repos will import torch.
    try:
        import torch  # type: ignore
    except Exception:
        return

    # 1) Hard cap optimizer steps by wrapping torch.optim.Optimizer.step
    try:
        import torch.optim  # type: ignore
    except Exception:
        torch_optim = None
    else:
        torch_optim = torch.optim

    if torch_optim is not None and hasattr(torch_optim, "Optimizer"):
        Opt = torch_optim.Optimizer
        if not hasattr(Opt, "_lyra_patched_step"):
            original_step = Opt.step
            _counter = {"steps": 0}

            def step(self, *args, **kwargs):  # type: ignore[no-redef]
                out = original_step(self, *args, **kwargs)
                _counter["steps"] += 1
                if _counter["steps"] >= LYRA_MAX_STEPS:
                    _warn_once(f"Reached LYRA_MAX_STEPS={LYRA_MAX_STEPS}. Exiting.")
                    raise SystemExit(0)
                return out

            Opt.step = step  # type: ignore[assignment]
            Opt._lyra_patched_step = True  # type: ignore[attr-defined]

    # 2) Disable common saving APIs (best-effort)
    if LYRA_DISABLE_SAVING and hasattr(torch, "save"):
        if not hasattr(torch.save, "_lyra_disabled"):  # type: ignore[attr-defined]
            original_save = torch.save

            def save(*args, **kwargs):  # type: ignore[no-redef]
                _warn_once("torch.save disabled (LYRA_DISABLE_SAVING=1).")
                return None

            save._lyra_disabled = True  # type: ignore[attr-defined]
            torch.save = save  # type: ignore[assignment]
            torch._lyra_original_save = original_save  # type: ignore[attr-defined]

    # Some projects use torch.jit.save
    try:
        jit = torch.jit  # type: ignore[attr-defined]
    except Exception:
        jit = None
    if LYRA_DISABLE_SAVING and jit is not None and hasattr(jit, "save"):
        original_jit_save = jit.save

        def jit_save(*args, **kwargs):  # type: ignore[no-redef]
            _warn_once("torch.jit.save disabled (LYRA_DISABLE_SAVING=1).")
            return None

        jit.save = jit_save  # type: ignore[assignment]
        jit._lyra_original_save = original_jit_save  # type: ignore[attr-defined]


# Apply immediately (and be resilient if torch is imported later).
_patch_torch()

"""


def _ignore_copytree(_: str, names: list[str]) -> set[str]:
    ignore = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "build",
        "dist",
        ".pytest_cache",
        ".ruff_cache",
        ".lyra",
    }
    return {n for n in names if n in ignore}


@dataclass(frozen=True)
class IsolatedRun:
    original_repo: Path
    run_dir: Path
    isolated_repo: Path
    isolated_script: Path
    sitecustomize_path: Path


def prepare_isolated_run(
    *,
    repo: Path,
    training_script: Path,
    runs_root: Optional[Path] = None,
) -> IsolatedRun:
    repo = repo.expanduser().resolve()
    training_script = training_script.expanduser().resolve()

    rel_script = training_script.relative_to(repo)

    if runs_root is None:
        runs_root = repo / ".lyra" / "runs"

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_dir = (runs_root / timestamp).resolve()
    isolated_repo = run_dir / "repo"
    isolated_repo.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(repo, isolated_repo, dirs_exist_ok=False, ignore=_ignore_copytree)

    isolated_script = (isolated_repo / rel_script).resolve()
    if not isolated_script.exists():
        raise FileNotFoundError(f"Script not found in isolated copy: {isolated_script}")

    sitecustomize_path = isolated_repo / "sitecustomize.py"
    sitecustomize_path.write_text(SAFE_PROFILE_SITECUSTOMIZE.strip() + "\n", encoding="utf-8")

    return IsolatedRun(
        original_repo=repo,
        run_dir=run_dir,
        isolated_repo=isolated_repo,
        isolated_script=isolated_script,
        sitecustomize_path=sitecustomize_path,
    )


