# Lyra

Lyra is a Python-first CLI + agent toolkit for **inspecting, profiling, and optimizing ML training code** before it runs on expensive hardware. It supports local static analysis and optional LLM-driven workflows via **Claude Code CLI**.

## Features

- **Repo summary**: counts Python files/lines and finds likely training entrypoints.
- **Static analysis (AST-first)**: detects common training/perf patterns with lower false positives:
  - AMP / mixed precision (`autocast`, `GradScaler`, Lightning precision flags)
  - Distributed training (DDP, Lightning strategies)
  - Sharding / DeepSpeed (FSDP, DeepSpeedStrategy)
- **Safe profiling (PyTorch-first)**:
  - Runs in an **isolated copy** of the repo (default)
  - **Hard-caps optimizer steps** (patches `torch.optim.Optimizer.step`)
  - **Disables saving** (best-effort patching of `torch.save` / `torch.jit.save`)
  - Captures logs under `.lyra/`
- **LLM workflows (optional)**: runs `commands/*.md` prompts via Claude Code CLI in non-interactive mode.
- **Optimize orchestration**:
  - **dry-run**: profile only
  - **plan**: profile + LLM analysis only
  - **apply**: profile → LLM analyze → LLM optimize → re-profile + before/after diff
- **Structured output**: `--output-format json` for `summarize/analyze/profile/check/optimize`.

## Installation

### Prerequisites

- **Python 3.9+**
- **Claude Code CLI** (only required for `lyra llm ...`): see [`https://docs.anthropic.com/en/docs/claude-code/quickstart`](https://docs.anthropic.com/en/docs/claude-code/quickstart)

### Install (recommended)

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Verify:

```bash
lyra --version
python -m lyra --version
```

### Legacy installer (deprecated)

`install.sh` is **deprecated** and kept only for backward compatibility. Prefer the Python install above.

## Quickstart

```bash
# 1) Structural summary + likely training entrypoints
lyra summarize /path/to/your/ml/repo

# 2) Static analysis for AMP/DDP/FSDP/DeepSpeed usage (AST by default)
lyra analyze /path/to/your/ml/repo

# 3) Safe profiling run (isolated by default; writes logs under .lyra/)
lyra profile /path/to/your/ml/repo train.py --max-steps 100

# 4) Orchestrate (profile -> optional LLM -> re-profile)
lyra optimize /path/to/your/ml/repo train.py --max-steps 100
```

## Outputs and artifacts

Lyra writes artifacts under the target repo:

- **Profiling logs**: `.lyra/profiles/profile_<script>_<timestamp>.log`
- **Isolated run copies**: `.lyra/runs/<timestamp>/repo`
- **Optimize outputs** (when using `--plan` or `--apply`):
  - `.lyra/optimize/analysis.txt`
  - `.lyra/optimize/optimize.txt` (only for `--apply`)

## Commands

### `lyra summarize`

Summarizes a repo (Python file counts, total lines, and likely training scripts).

```bash
lyra summarize /path/to/repo --output /tmp/lyra-summary.txt
lyra summarize /path/to/repo --output-format json
```

### `lyra analyze`

Scans likely training scripts (or all Python files with `--scan-all`) for:
- **Mixed precision** (AMP autocast, GradScaler, Lightning precision)
- **Distributed training** (DDP, Lightning strategies)
- **Sharding / DeepSpeed** (FSDP, DeepSpeedStrategy)

```bash
lyra analyze /path/to/repo
lyra analyze /path/to/repo --scan-all
lyra analyze /path/to/repo --engine ast
lyra analyze /path/to/repo --engine string  # legacy fallback
lyra analyze /path/to/repo --output /tmp/lyra-analysis.txt
lyra analyze /path/to/repo --output-format json
```

### `lyra profile`

Runs a training script in **safe profiling** mode.

**Safety model (PyTorch-first)**:
- Default is **isolated**: Lyra copies the repo to `.lyra/runs/...` and runs there.
- Lyra injects a `sitecustomize.py` in the isolated repo that (when `LYRA_SAFE_PROFILE=1`):
  - caps optimizer steps by wrapping `torch.optim.Optimizer.step`
  - disables `torch.save` / `torch.jit.save` best-effort
- Logs are always written to `.lyra/profiles/...` in the original repo.

**Flags**:
- `--max-steps N`: step cap (optimizer steps)
- `--isolated` / `--no-isolated`: run in isolated copy (default) or in-place
- `--runs-root PATH`: override isolated run root directory
- `--output-format json`: prints structured run info + parsed metrics

```bash
lyra profile /path/to/repo train.py --max-steps 100
lyra profile /path/to/repo train.py --python /path/to/python
lyra profile /path/to/repo train.py --output-format json
lyra profile /path/to/repo train.py --no-isolated  # not recommended
```

### `lyra optimize`

Lyra optimize supports three modes:
- **dry-run** (default): profile only
- **plan**: profile + LLM analysis only (no edits)
- **apply**: profile → LLM analyze → LLM optimize → re-profile + diff

```bash
lyra optimize /path/to/repo train.py --max-steps 100
```

Plan (profile + Claude analysis only; no code changes):

```bash
lyra optimize /path/to/repo train.py --max-steps 100 --plan
```

Apply (runs Claude prompts and may modify the repo):

```bash
lyra optimize /path/to/repo train.py --max-steps 100 --apply
```

Notes:
- In `--apply` mode, optimize writes prompt outputs under `.lyra/optimize/` and emits a **before/after diff** (duration + parsed metrics).
- Use `--output-format json` for machine-readable reports.
- In `--apply` mode, Lyra also records an **undo snapshot** under `.lyra/history/<run-id>/`.

### `lyra llm` (Claude Code CLI integration)

Runs prompts from `commands/*.md` non-interactively using `claude -p`.

Convenience wrappers:

```bash
lyra llm analyze  --repo /path/to/workspace --profile-file /path/to/profile.txt
lyra llm profile  --repo /path/to/workspace --training-script train.py
lyra llm optimize --repo /path/to/workspace --analysis-file /path/to/analysis.md
```

Power-user mode:

```bash
lyra llm run lyraAnalyze --repo /path/to/workspace --arguments /path/to/profile.txt --output-format text
```

### `lyra setup`

Creates an environment for running/profiling a training repo.

Venv (default):

```bash
lyra setup /path/to/repo
source /path/to/repo/.venv/bin/activate
```

Skip dependency installation:

```bash
lyra setup /path/to/repo --skip-install
```

Conda (if `environment.yml` exists and `conda` is available):

```bash
lyra setup /path/to/repo --prefer conda --skip-install
```

### `lyra undo`

Revert changes made by `lyra optimize --apply` using snapshots stored under `.lyra/history/`.

```bash
# list snapshots (JSON)
lyra undo list --repo /path/to/repo

# undo most recent snapshot
lyra undo last --repo /path/to/repo

# undo a specific run-id
lyra undo apply --repo /path/to/repo <run-id>

# overwrite even if files changed since the run
lyra undo last --repo /path/to/repo --force
```

Limitations:
- Undo can only restore files that were backed up (by default: common code/text extensions).
- By default, undo refuses to overwrite files that have changed since the snapshot; use `--force` to override.

## JSON output

Most commands support `--output-format json` for integration with other tools/CI:

```bash
lyra check --output-format json
lyra summarize /path/to/repo --output-format json
lyra analyze /path/to/repo --output-format json
lyra profile /path/to/repo train.py --output-format json
lyra optimize /path/to/repo train.py --output-format json
```

## Development

Run lint + tests:

```bash
ruff check .
pytest -q
```

CI runs `ruff` and `pytest` on pull requests via GitHub Actions.

## Troubleshooting

### Run check first

```bash
lyra check
lyra check --repo /path/to/workspace
```

### `claude` not found

- Install Claude Code CLI and authenticate.
- Confirm it’s on PATH:

```bash
claude --version
```

### Profile run doesn’t stop at 100 steps

In isolated mode, Lyra enforces the cap by patching `torch.optim.Optimizer.step`. If your training loop does not call `Optimizer.step()` (or uses a custom stepping mechanism), the cap may not trigger. In that case:

- Try `--max-steps` with a smaller value to validate the hook is firing.
- For Lightning/HF Trainer workloads, we’ll add framework-specific hard caps next.
