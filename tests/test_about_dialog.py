import importlib
import importlib.metadata

import pytest
from PySide6 import QtWidgets

import photoaident.ui.about_dialog as about_mod
from photoaident.ui.about_dialog import AboutDialog


@pytest.fixture()
def dialog(qtbot) -> AboutDialog:
    dlg = AboutDialog()
    qtbot.addWidget(dlg)
    return dlg


def test_about_dialog_title(dialog):
    assert dialog.windowTitle() == "About PhotoAIdent"


def test_about_dialog_has_close_button(dialog):
    button_box = dialog.findChild(QtWidgets.QDialogButtonBox)
    assert button_box is not None
    close_btn = button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Close)
    assert close_btn is not None


def test_about_dialog_contains_author_info(dialog):
    labels = dialog.findChildren(QtWidgets.QLabel)
    combined_text = " ".join(lbl.text() for lbl in labels)
    assert "seb.kde@hpfsc.de" in combined_text
    assert "github.com/steinsag/photoaident" in combined_text
    assert "Apache 2.0" in combined_text


def test_about_dialog_contains_all_libraries(dialog):
    labels = dialog.findChildren(QtWidgets.QLabel)
    combined_text = " ".join(lbl.text() for lbl in labels)
    for name, url in AboutDialog._LIBRARIES:
        assert name in combined_text, f"Library '{name}' missing from about dialog"
        assert url in combined_text, f"URL '{url}' missing from about dialog"


def test_about_dialog_contains_all_icons(dialog):
    labels = dialog.findChildren(QtWidgets.QLabel)
    combined_text = " ".join(lbl.text() for lbl in labels)
    for name, url in AboutDialog._ICONS:
        assert name in combined_text, f"Icon source '{name}' missing from about dialog"
        assert url in combined_text, f"URL '{url}' missing from about dialog"


def test_about_dialog_links_open_externally(dialog):
    """All QLabels with links must have openExternalLinks enabled."""
    labels = dialog.findChildren(QtWidgets.QLabel)
    for lbl in labels:
        if "<a href=" in lbl.text():
            assert (
                lbl.openExternalLinks()
            ), f"Label with links does not have openExternalLinks=True: {lbl.text()!r}"


def test_about_dialog_close_button_rejects(qtbot, dialog):
    button_box = dialog.findChild(QtWidgets.QDialogButtonBox)
    close_btn = button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Close)
    with qtbot.waitSignal(dialog.rejected, timeout=1000):
        close_btn.click()


def test_about_dialog_shows_version_in_title(qtbot, monkeypatch):
    """Title label reflects _APP_VERSION when package metadata is available."""
    monkeypatch.setattr(about_mod, "_APP_VERSION", "1.2.3")
    dlg = AboutDialog()
    qtbot.addWidget(dlg)
    labels = dlg.findChildren(QtWidgets.QLabel)
    title_text = next(lbl.text() for lbl in labels if "PhotoAIdent" in lbl.text())
    assert "1.2.3" in title_text


def test_about_dialog_shows_unknown_version_when_fallback(qtbot, monkeypatch):
    """Title label shows 'unknown' when _APP_VERSION fell back to its default."""
    monkeypatch.setattr(about_mod, "_APP_VERSION", "unknown")
    dlg = AboutDialog()
    qtbot.addWidget(dlg)
    labels = dlg.findChildren(QtWidgets.QLabel)
    title_text = next(lbl.text() for lbl in labels if "PhotoAIdent" in lbl.text())
    assert "unknown" in title_text


def test_version_falls_back_to_unknown_on_package_not_found(monkeypatch, request):
    """_APP_VERSION is 'unknown' when PackageNotFoundError is raised on import."""
    from importlib.metadata import PackageNotFoundError

    original_version = about_mod._APP_VERSION
    request.addfinalizer(lambda: setattr(about_mod, "_APP_VERSION", original_version))

    def _raise(_name: str) -> str:
        raise PackageNotFoundError(_name)

    monkeypatch.setattr(importlib.metadata, "version", _raise)
    importlib.reload(about_mod)
    assert about_mod._APP_VERSION == "unknown"
