from __future__ import annotations

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
