# Lyra

Lyra is a Python-first CLI + agent toolkit for **inspecting, profiling, and optimizing ML training code** before it runs on expensive hardware. It supports local static analysis and optional LLM-driven workflows via **Claude Code CLI**.

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

# 3) Best-effort "safe" profiling run (writes logs under .lyra/)
lyra profile /path/to/your/ml/repo train.py --max-steps 100
```

## Commands

### `lyra summarize`

Summarizes a repo (Python file counts, total lines, and likely training scripts).

```bash
lyra summarize /path/to/repo --output /tmp/lyra-summary.txt
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
```

### `lyra profile`

Runs a training script in a best-effort safe mode:
- Captures stdout/stderr to `.lyra/profiles/profile_<script>_<timestamp>.log`
- Sets env vars (your training code can choose to honor these):
  - `LYRA_SAFE_PROFILE=1`
  - `LYRA_MAX_STEPS=<N>`
  - `LYRA_DISABLE_SAVING=1`

```bash
lyra profile /path/to/repo train.py --max-steps 100
lyra profile /path/to/repo train.py --python /path/to/python
```

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

## Development

Run lint + tests:

```bash
ruff check .
pytest -q
```

CI runs `ruff` and `pytest` on pull requests via GitHub Actions.

## Troubleshooting

### `claude` not found

- Install Claude Code CLI and authenticate.
- Confirm it’s on PATH:

```bash
claude --version
```

### Profile run doesn’t stop at 100 steps

`lyra profile` sets env vars; your training script must read them to enforce strict limits. The log file location will still be written under `.lyra/profiles/`.
