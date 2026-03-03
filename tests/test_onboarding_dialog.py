"""Tests for OnboardingDialog.

Critical: never call exec(), accept(), or reject() — see CLAUDE.md.
Call _on_accept() directly to test the accept path.
"""

from unittest.mock import MagicMock

from PySide6 import QtWidgets

from photoaident.ui.onboarding_dialog import OnboardingDialog


def test_start_btn_disabled_initially(qtbot):
    """The 'Start Indexing' button is disabled before a folder is selected."""
    dlg = OnboardingDialog()
    qtbot.addWidget(dlg)
    assert not dlg._start_btn.isEnabled()


def test_selected_path_empty_before_accept(qtbot):
    """selected_path() returns an empty string before the user accepts."""
    dlg = OnboardingDialog()
    qtbot.addWidget(dlg)
    assert dlg.selected_path() == ""


def test_browse_enables_start_btn(qtbot, monkeypatch):
    """Choosing a folder via _browse enables the Start Indexing button."""
    dlg = OnboardingDialog()
    qtbot.addWidget(dlg)

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getExistingDirectory",
        MagicMock(return_value="/chosen/folder"),
    )

    dlg._browse()

    assert dlg._start_btn.isEnabled()
    assert dlg._path_edit.text() == "/chosen/folder"


def test_browse_cancelled_leaves_start_btn_disabled(qtbot, monkeypatch):
    """Cancelling the folder dialog leaves the Start Indexing button disabled."""
    dlg = OnboardingDialog()
    qtbot.addWidget(dlg)

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getExistingDirectory",
        MagicMock(return_value=""),  # user cancelled
    )

    dlg._browse()

    assert not dlg._start_btn.isEnabled()
    assert dlg._path_edit.text() == ""


def test_on_accept_stores_path(qtbot, monkeypatch):
    """_on_accept() stores the current path and calls accept()."""
    dlg = OnboardingDialog()
    qtbot.addWidget(dlg)

    dlg._path_edit.setText("/my/photos")

    accept_called: list[bool] = []
    monkeypatch.setattr(dlg, "accept", lambda: accept_called.append(True))

    dlg._on_accept()

    assert dlg.selected_path() == "/my/photos"
    assert accept_called == [True]


def test_on_accept_with_empty_path(qtbot, monkeypatch):
    """_on_accept() stores an empty string when no folder was chosen."""
    dlg = OnboardingDialog()
    qtbot.addWidget(dlg)

    monkeypatch.setattr(dlg, "accept", lambda: None)
    dlg._on_accept()

    assert dlg.selected_path() == ""
