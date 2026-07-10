"""スキャフォールドの疎通確認テスト。"""

import pytest

import atlasmith
import atlasmith.cli


def test_version_is_not_empty() -> None:
    assert atlasmith.__version__ != ""


def test_cli_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc_info:
        atlasmith.cli.main(["--help"])
    assert exc_info.value.code == 0
