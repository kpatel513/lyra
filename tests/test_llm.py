from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from lyra.llm import ClaudeCodeRunner, LlmResult
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


def test_cli_llm_wrapper_analyze_maps_profile_file(tmp_path: Path, monkeypatch) -> None:
    """
    Verify `lyra llm analyze` maps --profile-file -> $ARGUMENTS and selects lyraAnalyze.md.
    """
    # Build a fake repo root with commands/lyraAnalyze.md
    repo_root = tmp_path / "lyra_repo"
    (repo_root / "commands").mkdir(parents=True)
    (repo_root / "commands" / "lyraAnalyze.md").write_text(
        "args: --dangerously-skip-permissions\nAnalyze $ARGUMENTS\n",
        encoding="utf-8",
    )

    # Patch lyra.cli to think it's located under this repo_root/src/lyra/cli.py
    src_dir = repo_root / "src" / "lyra"
    src_dir.mkdir(parents=True)
    fake_cli_path = src_dir / "cli.py"
    fake_cli_path.write_text("# placeholder\n", encoding="utf-8")

    from lyra.commands import llm_cmd

    # Make llm_cmd resolve prompts relative to our fake repo root by monkeypatching
    # the `project_root` we pass in.
    project_root = repo_root

    captured = {}

    class FakeRunner:
        def is_available(self) -> bool:
            return True

        def run(self, *, prompt, cwd, extra_args, output_format):
            captured["prompt"] = prompt
            captured["cwd"] = cwd
            captured["extra_args"] = extra_args
            captured["output_format"] = output_format
            return LlmResult(command=["claude"], stdout="ok", stderr="", return_code=0)

    monkeypatch.setattr(llm_cmd, "ClaudeCodeRunner", FakeRunner)

    args = type(
        "Args",
        (),
        {
            "repo": str(tmp_path),
            "profile_file": "prof.txt",
            "output_format": "text",
            "output": None,
            "model": "sonnet",
            "permission_mode": "plan",
            "allowed_tools": ["Bash(git:*)"],
            "disallowed_tools": ["Edit"],
            "dangerously_skip_permissions": False,
        },
    )()

    rc = llm_cmd.cmd_llm_analyze(args, project_root=project_root)
    assert rc == 0
    assert "Analyze prof.txt" in captured["prompt"]
    assert captured["extra_args"] == [
        "--dangerously-skip-permissions",
        "--model",
        "sonnet",
        "--permission-mode",
        "plan",
        "--allowed-tools",
        "Bash(git:*)",
        "--disallowed-tools",
        "Edit",
    ]


