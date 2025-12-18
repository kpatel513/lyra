from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from lyra.llm import ClaudeCodeRunner
from lyra.prompts import load_prompt


def test_load_prompt_parses_args_and_renders(tmp_path: Path) -> None:
    p = tmp_path / "prompt.md"
    p.write_text(
        "args: --dangerously-skip-permissions --output-format json\n"
        "Analyze $ARGUMENTS and patch $TRAINING_SCRIPT.\n",
        encoding="utf-8",
    )

    spec = load_prompt(p)
    assert spec.cli_args == ["--dangerously-skip-permissions", "--output-format", "json"]
    assert "Analyze" in spec.template
    rendered = spec.render({"ARGUMENTS": "report.txt", "TRAINING_SCRIPT": "train.py"})
    assert "report.txt" in rendered
    assert "train.py" in rendered


def test_claude_runner_builds_noninteractive_command() -> None:
    runner = ClaudeCodeRunner(claude_executable="claude")

    with patch("subprocess.run") as run:
        run.return_value.stdout = "ok"
        run.return_value.stderr = ""
        run.return_value.returncode = 0

        res = runner.run(
            prompt="hello",
            extra_args=["--dangerously-skip-permissions"],
            output_format="json",
        )

        assert res.return_code == 0
        assert res.stdout == "ok"
        # Ensure -p and output format are included
        assert res.command[:4] == ["claude", "-p", "--output-format", "json"]
        assert "--dangerously-skip-permissions" in res.command
        assert res.command[-1] == "hello"


