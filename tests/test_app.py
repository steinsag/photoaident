from PySide6 import QtCore

from photoaident.app import MyWidget


def test_app_setup(qtbot):
    """
    Test that the application widget can be instantiated and basic interaction works.
    """
    widget = MyWidget()
    qtbot.addWidget(widget)

    # Check initial state
    assert widget.text.text() == "Hello World"

    # Click the button
    with qtbot.waitSignal(widget.button.clicked, timeout=1000):
        qtbot.mouseClick(widget.button, QtCore.Qt.MouseButton.LeftButton)

    # Check that text changed to one of the expected values
    qtbot.waitUntil(lambda: widget.text.text() in widget.hello, timeout=1000)
