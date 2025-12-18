"""
Core utilities for inspecting ML/AI training repositories.

This is intentionally lightweight for now and can be expanded into a
full agent-backed analysis module as Lyra evolves.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


PYTHON_EXTENSIONS = {".py", ".pyw"}


@dataclass
class RepoSummary:
    root: Path
    python_files: int
    total_lines: int
    training_scripts: List[Path]

    def format_human(self) -> str:
        rel = (
            lambda p: p.relative_to(self.root)
            if p.is_relative_to(self.root)
            else p
        )
        training_list = (
            "\n".join(f"  - {rel(p)}" for p in self.training_scripts)
            or "  (none detected)"
        )

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


# --- Deeper analysis: look for optimization / distributed patterns ----------------


@dataclass
class AnalysisFinding:
    kind: str
    file: Path
    line_number: int
    code_line: str


@dataclass
class AnalysisReport:
    root: Path
    training_scripts: List[Path]
    findings: List[AnalysisFinding]

    def format_human(self) -> str:
        rel = (
            lambda p: p.relative_to(self.root)
            if p.is_relative_to(self.root)
            else p
        )

        if not self.training_scripts:
            header = (
                "🎵 Lyra Performance Scan\n"
                f"✨ Root: {self.root}\n"
                "⚠️ No obvious training scripts detected. "
                "Lyra looked for filenames containing train/fit/finetune/downstream.\n"
            )
        else:
            train_list = "\n".join(f"  - {rel(p)}" for p in self.training_scripts)
            header = (
                "🎵 Lyra Performance Scan\n"
                f"✨ Root: {self.root}\n"
                "🎯 Training entrypoints considered:\n"
                f"{train_list}\n"
            )

        if not self.findings:
            return header + (
                "\n"
                "🔍 No clear signs of mixed precision, DDP/FSDP, or DeepSpeed usage were\n"
                "found in the scanned training scripts. This does not guarantee they are\n"
                "absent, but suggests they are not configured in a conventional way.\n"
            )

        # Group findings by kind for a compact overview.
        by_kind: Dict[str, List[AnalysisFinding]] = {}
        for f in self.findings:
            by_kind.setdefault(f.kind, []).append(f)

        lines = [header, "📊 Detected optimization patterns:"]
        kind_order = [
            "mixed_precision_autocast",
            "mixed_precision_grad_scaler",
            "mixed_precision_dtype_fp16",
            "mixed_precision_dtype_bf16",
            "lightning_precision_arg",
            "ddp_torch",
            "ddp_lightning_strategy",
            "ddp_torchrun",
            "fsdp",
            "deepspeed",
        ]

        labels = {
            "mixed_precision_autocast": "Mixed precision autocast (torch.cuda.amp/autocast)",
            "mixed_precision_grad_scaler": "GradScaler-based mixed precision",
            "mixed_precision_dtype_fp16": "FP16 dtype configuration",
            "mixed_precision_dtype_bf16": "BF16 dtype configuration",
            "lightning_precision_arg": "PyTorch Lightning precision flag",
            "ddp_torch": "Torch DistributedDataParallel",
            "ddp_lightning_strategy": "Lightning DDP / sharded strategies",
            "ddp_torchrun": "torchrun / torch.distributed launcher",
            "fsdp": "Fully Sharded Data Parallel (FSDP)",
            "deepspeed": "DeepSpeed strategy",
        }

        for kind in kind_order:
            findings = by_kind.get(kind)
            if not findings:
                continue
            label = labels.get(kind, kind)
            lines.append(f"\n- {label}: {len(findings)} occurrence(s)")
            # Show up to a few concrete locations
            for f in findings[:5]:
                lines.append(
                    f"    • {rel(f.file)}:{f.line_number}: {f.code_line.strip()}"
                )
            if len(findings) > 5:
                lines.append(f"    • … {len(findings) - 5} more")

        return "\n".join(lines) + "\n"


_PATTERN_KINDS: Dict[str, str] = {}


def _register_patterns(kind: str, patterns: Iterable[str]) -> None:
    for p in patterns:
        _PATTERN_KINDS[p] = kind


# Mixed precision patterns
_register_patterns(
    "mixed_precision_autocast",
    [
        "torch.cuda.amp.autocast",
        "torch.autocast(",
        "autocast(",
    ],
)
_register_patterns(
    "mixed_precision_grad_scaler",
    [
        "torch.cuda.amp.GradScaler",
        "GradScaler(",
    ],
)
_register_patterns(
    "mixed_precision_dtype_fp16",
    [
        "torch.float16",
        "torch.half",
        "dtype=torch.float16",
        "dtype = torch.float16",
    ],
)
_register_patterns(
    "mixed_precision_dtype_bf16",
    [
        "torch.bfloat16",
        "dtype=torch.bfloat16",
        "dtype = torch.bfloat16",
    ],
)
_register_patterns(
    "lightning_precision_arg",
    [
        "precision=",
        "precision =",
    ],
)

# DDP / distributed patterns
_register_patterns(
    "ddp_torch",
    [
        "DistributedDataParallel(",
        "torch.nn.parallel.DistributedDataParallel",
    ],
)
_register_patterns(
    "ddp_lightning_strategy",
    [
        "strategy=\"ddp\"",
        "strategy='ddp'",
        "strategy=\"ddp_sharded\"",
        "strategy='ddp_sharded'",
    ],
)
_register_patterns(
    "ddp_torchrun",
    [
        "torchrun ",
        "python -m torch.distributed.run",
        "torch.distributed.launch",
    ],
)

# Sharding / DeepSpeed patterns
_register_patterns(
    "fsdp",
    [
        "FullyShardedDataParallel(",
        "FSDP(",
        "strategy=\"fsdp\"",
        "strategy='fsdp'",
    ],
)
_register_patterns(
    "deepspeed",
    [
        "deepspeed",
        "DeepSpeedStrategy",
    ],
)


def analyze_repo(root: Path, *, scan_all_python_files: bool = False) -> AnalysisReport:
    """
    Scan likely training scripts for common optimization and distributed patterns:
    - mixed precision & AMP
    - DDP / sharded / torchrun
    - FSDP / DeepSpeed
    """
    summary = summarize_repo(root)
    candidates = summary.training_scripts

    # Fall back to scanning all python files if we didn't find any obvious
    # training entrypoints, or if the user requested a full scan.
    if scan_all_python_files or not candidates:
        candidates = list(_iter_python_files(summary.root))

    findings: List[AnalysisFinding] = []

    for f in candidates:
        try:
            with f.open("r", encoding="utf-8", errors="ignore") as fh:
                for lineno, line in enumerate(fh, start=1):
                    for pattern, kind in _PATTERN_KINDS.items():
                        if pattern in line:
                            findings.append(
                                AnalysisFinding(
                                    kind=kind,
                                    file=f,
                                    line_number=lineno,
                                    code_line=line.rstrip("\n"),
                                )
                            )
        except OSError:
            continue

    return AnalysisReport(
        root=summary.root,
        training_scripts=summary.training_scripts,
        findings=findings,
    )


# --- Safe profiling ---------------------------------------------------------


@dataclass
class ProfileResult:
    root: Path
    script: Path
    command: List[str]
    log_file: Path
    return_code: int

    def format_human(self) -> str:
        status = "✅ Completed" if self.return_code == 0 else f"⚠️ Exit code {self.return_code}"
        return (
            "🎵 Lyra Safe Profiling Run\n"
            f"✨ Root: {self.root}\n"
            f"🎯 Script: {self.script}\n"
            f"📝 Log file: {self.log_file}\n"
            f"🚀 Status: {status}\n"
        )


def _select_training_script(root: Path, explicit: Optional[str]) -> Path:
    root = root.resolve()
    if explicit:
        script_path = Path(explicit)
        if not script_path.is_absolute():
            script_path = root / script_path
        script_path = script_path.resolve()
        if not script_path.exists():
            raise FileNotFoundError(f"Training script not found: {script_path}")
        return script_path

    summary = summarize_repo(root)
    if not summary.training_scripts:
        raise RuntimeError(
            "Could not auto-detect a training script. "
            "Pass one explicitly, e.g. `lyra profile REPO_PATH train.py`."
        )
    # Pick the first candidate deterministically.
    return summary.training_scripts[0].resolve()


def run_safe_profile(
    root: Path,
    training_script: Optional[str] = None,
    max_steps: int = 100,
    log_dir: Optional[Path] = None,
    python_executable: Optional[str] = None,
) -> ProfileResult:
    """
    Run the training script in a best-effort "safe profiling" mode:
    - selects a training script (auto or explicit)
    - sets env vars signalling safe mode and step cap
    - captures stdout/stderr to a log file for later analysis

    NOTE: The training code needs to *honour* these env vars for strict
    safety guarantees. Lyra does not rewrite the user's code yet.
    """
    root = root.resolve()
    script_path = _select_training_script(root, training_script)

    # Determine where to put logs.
    if log_dir is None:
        log_dir = root / ".lyra" / "profiles"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"profile_{script_path.stem}_{timestamp}.log"

    python_exe = (
        python_executable
        or os.environ.get("PYTHON")
        or os.sys.executable
    )
    cmd = [python_exe, str(script_path)]

    # Best-effort safe mode signalling. The actual training script must
    # look at these to reduce steps / disable saving.
    env = os.environ.copy()
    env.setdefault("LYRA_SAFE_PROFILE", "1")
    env.setdefault("LYRA_MAX_STEPS", str(max_steps))
    env.setdefault("LYRA_DISABLE_SAVING", "1")

    with log_file.open("w", encoding="utf-8") as fh:
        process = subprocess.run(
            cmd,
            cwd=root,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )

    return ProfileResult(
        root=root,
        script=script_path,
        command=cmd,
        log_file=log_file,
        return_code=process.returncode,
    )

