"""Tests for __main__.py: argument parsing and NVIDIA path setup."""

import os
import sys
from unittest.mock import patch

import pytest

from photoaident.__main__ import _parse_args, ensure_nvidia_paths

# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------


def test_parse_args_default_log_level(monkeypatch):
    """Default log level is WARNING when no flag is given."""
    monkeypatch.setattr(sys, "argv", ["photoaident"])
    args = _parse_args()
    assert args.log_level == "WARNING"


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
def test_parse_args_explicit_log_level(monkeypatch, level):
    """Each valid log level is accepted and returned verbatim."""
    monkeypatch.setattr(sys, "argv", ["photoaident", "--log-level", level])
    args = _parse_args()
    assert args.log_level == level


def test_parse_args_unknown_qt_flags_are_ignored(monkeypatch):
    """Qt-style flags (e.g. -platform) do not cause parse errors."""
    monkeypatch.setattr(
        sys, "argv", ["photoaident", "-platform", "xcb", "--log-level", "DEBUG"]
    )
    args = _parse_args()
    assert args.log_level == "DEBUG"


def test_parse_args_invalid_level_exits(monkeypatch):
    """An unrecognised log level causes SystemExit (argparse behaviour)."""
    monkeypatch.setattr(sys, "argv", ["photoaident", "--log-level", "VERBOSE"])
    with pytest.raises(SystemExit):
        _parse_args()


# ---------------------------------------------------------------------------
# ensure_nvidia_paths — early-return branches
# ---------------------------------------------------------------------------


def test_ensure_nvidia_paths_skips_on_non_linux(monkeypatch):
    """Nothing happens on non-Linux platforms."""
    monkeypatch.setenv("ORT_PATHS_SET", "")
    with (
        patch("platform.system", return_value="Darwin"),
        patch("os.execv") as mock_execv,
    ):
        ensure_nvidia_paths()
    mock_execv.assert_not_called()


def test_ensure_nvidia_paths_skips_when_already_set(monkeypatch):
    """Guard env var ORT_PATHS_SET=1 prevents a second re-exec."""
    monkeypatch.setenv("ORT_PATHS_SET", "1")
    with (
        patch("platform.system", return_value="Linux"),
        patch("os.execv") as mock_execv,
    ):
        ensure_nvidia_paths()
    mock_execv.assert_not_called()


def test_ensure_nvidia_paths_skips_when_no_nvidia_dir(monkeypatch, tmp_path):
    """No re-exec when the nvidia/ directory doesn't exist."""
    monkeypatch.delenv("ORT_PATHS_SET", raising=False)
    site_pkgs = tmp_path / "site-packages"
    site_pkgs.mkdir()
    with (
        patch("platform.system", return_value="Linux"),
        patch("site.getsitepackages", return_value=[str(site_pkgs)]),
        patch("os.execv") as mock_execv,
    ):
        ensure_nvidia_paths()
    mock_execv.assert_not_called()


def test_ensure_nvidia_paths_skips_when_no_candidates(monkeypatch, tmp_path):
    """No re-exec when nvidia/ exists but contains no .so files or lib/ dirs."""
    monkeypatch.delenv("ORT_PATHS_SET", raising=False)
    site_pkgs = tmp_path / "site-packages"
    nvidia_root = site_pkgs / "nvidia"
    nvidia_root.mkdir(parents=True)
    # no *.so* files and no lib/ subdirs → candidates stays empty
    with (
        patch("platform.system", return_value="Linux"),
        patch("site.getsitepackages", return_value=[str(site_pkgs)]),
        patch("os.execv") as mock_execv,
    ):
        ensure_nvidia_paths()
    mock_execv.assert_not_called()


def test_ensure_nvidia_paths_skips_on_getsitepackages_index_error(monkeypatch):
    """IndexError from getsitepackages is handled gracefully."""
    monkeypatch.delenv("ORT_PATHS_SET", raising=False)
    with (
        patch("platform.system", return_value="Linux"),
        patch("site.getsitepackages", return_value=[]),
        patch("os.execv") as mock_execv,
    ):
        ensure_nvidia_paths()
    mock_execv.assert_not_called()


def test_ensure_nvidia_paths_reexecs_with_lib_dir(monkeypatch, tmp_path):
    """Re-execs with correct LD_LIBRARY_PATH when a lib/ subdir is present."""
    monkeypatch.delenv("ORT_PATHS_SET", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    site_pkgs = tmp_path / "site-packages"
    lib_dir = site_pkgs / "nvidia" / "cudnn" / "lib"
    lib_dir.mkdir(parents=True)

    with (
        patch("platform.system", return_value="Linux"),
        patch("site.getsitepackages", return_value=[str(site_pkgs)]),
        patch("os.execv") as mock_execv,
    ):
        ensure_nvidia_paths()

    mock_execv.assert_called_once()
    new_ld = os.environ.get("LD_LIBRARY_PATH", "")
    assert str(lib_dir.resolve()) in new_ld
    assert os.environ.get("ORT_PATHS_SET") == "1"


def test_ensure_nvidia_paths_prepends_to_existing_ld_library_path(
    monkeypatch, tmp_path
):
    """Existing LD_LIBRARY_PATH is preserved and appended after nvidia paths."""
    monkeypatch.delenv("ORT_PATHS_SET", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/lib")

    site_pkgs = tmp_path / "site-packages"
    lib_dir = site_pkgs / "nvidia" / "cuda" / "lib"
    lib_dir.mkdir(parents=True)

    with (
        patch("platform.system", return_value="Linux"),
        patch("site.getsitepackages", return_value=[str(site_pkgs)]),
        patch("os.execv"),
    ):
        ensure_nvidia_paths()

    new_ld = os.environ.get("LD_LIBRARY_PATH", "")
    assert new_ld.endswith(":/existing/lib")


def test_ensure_nvidia_paths_so_files_in_nvidia_root(monkeypatch, tmp_path):
    """nvidia/ root itself is included in the path when it contains .so files."""
    monkeypatch.delenv("ORT_PATHS_SET", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    site_pkgs = tmp_path / "site-packages"
    nvidia_root = site_pkgs / "nvidia"
    nvidia_root.mkdir(parents=True)
    (nvidia_root / "libtensorrt.so.8").touch()

    with (
        patch("platform.system", return_value="Linux"),
        patch("site.getsitepackages", return_value=[str(site_pkgs)]),
        patch("os.execv") as mock_execv,
    ):
        ensure_nvidia_paths()

    mock_execv.assert_called_once()
    new_ld = os.environ.get("LD_LIBRARY_PATH", "")
    assert str(nvidia_root.resolve()) in new_ld
