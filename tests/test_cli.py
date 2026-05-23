from __future__ import annotations

from pathlib import Path

import pytest

from postlens import __version__
from postlens.cli import main


def test_cli_version_flag_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert __version__ in captured.out


def test_cli_info_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["info"])
    assert rc == 0
    captured = capsys.readouterr()
    assert __version__ in captured.out


def test_cli_no_subcommand_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([])
    assert rc == 0
    captured = capsys.readouterr()
    assert "postlens" in captured.out


def test_cli_run_missing_task_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["run", "/nonexistent/task.md"])
    assert rc == 2


def test_cli_run_emits_csv_with_dummy_backbone(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    task = tmp_path / "t.md"
    task.write_text("# task: test\n")
    skills_dir = Path(__file__).resolve().parents[1] / "examples" / "skills"
    rc = main(
        [
            "run",
            str(task),
            "--backbone",
            "dummy",
            "--skills",
            str(skills_dir),
            "--decode-tokens",
            "2",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "skill,ttft_s,tok_per_s,state_bytes" in out
    assert "csv_stat," in out


def test_cli_run_missing_skills_dir_returns_2(tmp_path) -> None:
    task = tmp_path / "t.md"
    task.write_text("# task: test\n")
    rc = main(["run", str(task), "--skills", "/nonexistent/skills"])
    assert rc == 2
