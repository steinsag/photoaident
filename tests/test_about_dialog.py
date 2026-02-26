import pytest
from PySide6 import QtWidgets

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
