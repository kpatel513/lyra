from __future__ import annotations

from pathlib import Path

from lyra.metrics import parse_profile_log


def test_parse_profile_log_detects_max_steps_and_saving_disabled(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "[lyra-safe-profile] torch.save disabled (LYRA_DISABLE_SAVING=1).\n"
        "[lyra-safe-profile] Reached LYRA_MAX_STEPS=123. Exiting.\n",
        encoding="utf-8",
    )
    m = parse_profile_log(log)
    assert m.exit_reason == "max_steps"
    assert m.max_steps == 123
    assert m.saving_disabled is True


