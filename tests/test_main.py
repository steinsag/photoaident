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


@pytest.mark.parametrize(
    "failure",
    [
        "migrations",
        "application",
        "translations",
        "window",
        "show",
        "exec",
        "corrupt",
        "normal",
    ],
)
def test_main_releases_lock_after_initialization_or_exit(
    tmp_app_paths, monkeypatch, failure
):
    """Ownership covers initialization failures and writer shutdown before release."""
    from unittest.mock import MagicMock
    import photoaident.__main__ as entry
    import photoaident.app as app_module

    calls = []
    lock = MagicMock()
    lock.acquire.return_value = True
    lock.release.side_effect = lambda: calls.append("release")
    application = MagicMock()
    application.exec.return_value = 7
    window = MagicMock()
    window.shutdown.side_effect = lambda: calls.append("shutdown")
    monkeypatch.setattr(sys, "argv", ["photoaident"])
    monkeypatch.setattr(entry, "AppPaths", lambda: tmp_app_paths)
    monkeypatch.setattr(entry, "ensure_nvidia_paths", lambda: None)
    monkeypatch.setattr(entry, "InstanceLock", lambda path: lock)
    migrations = MagicMock()
    application_factory = MagicMock(return_value=application)
    window_factory = MagicMock(return_value=window)
    translations = MagicMock()
    monkeypatch.setattr(entry, "apply_migrations", migrations)
    monkeypatch.setattr(entry.QtWidgets, "QApplication", application_factory)
    monkeypatch.setattr(entry.QtWidgets.QMessageBox, "critical", MagicMock())
    monkeypatch.setattr(entry, "MainWindow", window_factory)
    monkeypatch.setattr(app_module, "load_translations", translations)
    targets = {
        "migrations": migrations,
        "application": application_factory,
        "translations": translations,
        "window": window_factory,
        "show": window.show,
        "exec": application.exec,
    }
    if failure in targets:
        targets[failure].side_effect = RuntimeError("initialization failed")
    elif failure == "corrupt":
        window_factory.side_effect = entry.CorruptIndexError(tmp_app_paths.faiss_path)
    with pytest.raises(
        SystemExit if failure in ("normal", "corrupt") else RuntimeError
    ) as caught:
        entry.main()
    if failure == "normal":
        assert isinstance(caught.value, SystemExit)
        assert caught.value.code == 7
    elif failure == "corrupt":
        assert isinstance(caught.value, SystemExit)
        assert caught.value.code == 1
    lock.release.assert_called_once()
    assert calls == (
        ["shutdown", "release"]
        if failure in ("normal", "show", "exec")
        else ["release"]
    )


def test_main_rejected_instance_never_starts_writers(tmp_app_paths, monkeypatch):
    """A rejected instance exits without starting writers or releasing ownership."""
    from unittest.mock import MagicMock
    import photoaident.__main__ as entry
    import photoaident.app as app_module

    lock = MagicMock()
    lock.acquire.return_value = False
    monkeypatch.setattr(sys, "argv", ["photoaident"])
    monkeypatch.setattr(entry, "AppPaths", lambda: tmp_app_paths)
    monkeypatch.setattr(entry, "ensure_nvidia_paths", lambda: None)
    monkeypatch.setattr(entry, "InstanceLock", lambda path: lock)
    monkeypatch.setattr(entry.QtWidgets, "QApplication", MagicMock())
    monkeypatch.setattr(entry.QtWidgets.QMessageBox, "critical", MagicMock())
    monkeypatch.setattr(app_module, "load_translations", MagicMock())
    migrations, window = MagicMock(), MagicMock()
    monkeypatch.setattr(entry, "apply_migrations", migrations)
    monkeypatch.setattr(entry, "MainWindow", window)
    with pytest.raises(SystemExit) as caught:
        entry.main()
    assert caught.value.code == 1
    migrations.assert_not_called()
    window.assert_not_called()
    lock.release.assert_not_called()


def test_main_keeps_lock_if_writer_shutdown_fails(tmp_app_paths, monkeypatch):
    """Failure to join writers cannot fall through to explicit lock release."""
    from unittest.mock import MagicMock
    import photoaident.__main__ as entry
    import photoaident.app as app_module

    lock = MagicMock()
    lock.acquire.return_value = True
    window = MagicMock()
    window.shutdown.side_effect = RuntimeError("writer still active")
    app = MagicMock()
    app.exec.return_value = 0
    monkeypatch.setattr(sys, "argv", ["photoaident"])
    monkeypatch.setattr(entry, "AppPaths", lambda: tmp_app_paths)
    monkeypatch.setattr(entry, "ensure_nvidia_paths", lambda: None)
    monkeypatch.setattr(entry, "InstanceLock", lambda path: lock)
    monkeypatch.setattr(entry, "apply_migrations", MagicMock())
    monkeypatch.setattr(entry.QtWidgets, "QApplication", MagicMock(return_value=app))
    monkeypatch.setattr(entry, "MainWindow", MagicMock(return_value=window))
    monkeypatch.setattr(app_module, "load_translations", MagicMock())
    with pytest.raises(RuntimeError, match="writer still active"):
        entry.main()
    lock.release.assert_not_called()
