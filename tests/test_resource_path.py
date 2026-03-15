"""Tests for the shared resource-path utility (photoaident.utils.resource_path)."""

import os
import sys
from pathlib import Path

from photoaident.utils.resource_path import get_resource_path, icon_path

# ---------------------------------------------------------------------------
# get_resource_path
# ---------------------------------------------------------------------------


def test_get_resource_path_dev_mode():
    """In dev mode (no _MEIPASS) the path is anchored to the project root."""
    result = get_resource_path("assets/icons/search.svg")
    assert Path(result).parts[-3:] == ("assets", "icons", "search.svg")


def test_get_resource_path_pyinstaller_bundle(monkeypatch):
    """Uses _MEIPASS as the base directory when running in a PyInstaller bundle."""
    monkeypatch.setattr(sys, "_MEIPASS", "/bundle_root", raising=False)
    result = get_resource_path("assets/icon.png")
    assert result == os.path.join("/bundle_root", "assets/icon.png")


# ---------------------------------------------------------------------------
# icon_path
# ---------------------------------------------------------------------------


def test_icon_path_dev_mode():
    """In dev mode the path ends with assets/icons/<name>."""
    result = icon_path("zoom-in.svg")
    assert Path(result).parts[-3:] == ("assets", "icons", "zoom-in.svg")


def test_icon_path_pyinstaller_bundle(tmp_path, monkeypatch):
    """In a PyInstaller bundle the path is relative to _MEIPASS."""
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    result = icon_path("zoom-in.svg")
    assert result == str(tmp_path / "assets" / "icons" / "zoom-in.svg")
