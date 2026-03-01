from pathlib import Path
from unittest.mock import patch

from PySide6 import QtCore, QtGui

from photoaident.ui.widgets.image_preview_widget import ImagePreviewWidget


def _make_jpeg(path: Path) -> None:
    """Write a minimal valid JPEG file at path."""
    img = QtGui.QImage(80, 60, QtGui.QImage.Format.Format_RGB32)
    img.fill(QtGui.QColor(100, 150, 200))
    img.save(str(path))


# ===========================================================================
# ImagePreviewWidget
# ===========================================================================


def test_image_preview_load_missing_file(qtbot):
    """load() with a non-existent path shows the path text and keeps no pixmap."""
    widget = ImagePreviewWidget()
    qtbot.addWidget(widget)
    widget.load(Path("/nonexistent/image.jpg"), (0, 0, 50, 50))
    assert widget._original_pixmap is None
    assert "/nonexistent/image.jpg" in widget._label.text()


def test_image_preview_load_valid_image(qtbot, tmp_path):
    """load() with a real JPEG draws the bbox and stores the pixmap."""
    img_path = tmp_path / "face.jpg"
    _make_jpeg(img_path)

    widget = ImagePreviewWidget()
    qtbot.addWidget(widget)
    widget.load(img_path, (5, 5, 20, 20))

    assert widget._original_pixmap is not None
    assert not widget._original_pixmap.isNull()


def test_image_preview_load_invalid_file(qtbot, tmp_path):
    """load() with a file that exists but is not a valid image sets error text."""
    bad_path = tmp_path / "not_an_image.jpg"
    bad_path.write_bytes(b"this is not a jpeg")

    widget = ImagePreviewWidget()
    qtbot.addWidget(widget)
    widget.load(bad_path, (0, 0, 10, 10))

    # QImageReader fails → text is set, pixmap is None
    assert widget._original_pixmap is None


def test_image_preview_clear(qtbot, tmp_path):
    """clear() resets the pixmap and label text."""
    img_path = tmp_path / "face.jpg"
    _make_jpeg(img_path)

    widget = ImagePreviewWidget()
    qtbot.addWidget(widget)
    widget.load(img_path, (0, 0, 10, 10))
    assert widget._original_pixmap is not None

    widget.clear()
    assert widget._original_pixmap is None
    assert widget._label.pixmap().isNull()
    assert widget._label.text() == ""


def test_image_preview_update_display_no_pixmap(qtbot):
    """_update_display() is a no-op when no pixmap has been loaded."""
    widget = ImagePreviewWidget()
    qtbot.addWidget(widget)
    widget._update_display()  # must not raise


def test_image_preview_update_display_with_pixmap(qtbot, tmp_path):
    """_update_display() scales the stored pixmap to the label size."""
    img_path = tmp_path / "face.jpg"
    _make_jpeg(img_path)

    widget = ImagePreviewWidget()
    qtbot.addWidget(widget)
    widget.resize(400, 300)
    widget.show()

    widget.load(img_path, (0, 0, 10, 10))
    widget._update_display()

    assert not widget._label.pixmap().isNull()


def test_image_preview_resize_event(qtbot, tmp_path):
    """resizeEvent() schedules _update_display without raising."""
    img_path = tmp_path / "face.jpg"
    _make_jpeg(img_path)

    widget = ImagePreviewWidget()
    qtbot.addWidget(widget)
    widget.load(img_path, (0, 0, 10, 10))

    # Synthesise a resize — must not raise
    event = QtGui.QResizeEvent(QtCore.QSize(300, 200), QtCore.QSize(200, 150))
    widget.resizeEvent(event)


def test_image_preview_update_display_empty_label_size(qtbot, tmp_path):
    """_update_display returns early when the label reports zero size."""
    img_path = tmp_path / "face.jpg"
    _make_jpeg(img_path)

    widget = ImagePreviewWidget()
    qtbot.addWidget(widget)
    widget.load(img_path, (0, 0, 10, 10))
    assert widget._original_pixmap is not None

    # Force the label to report an empty size
    with patch.object(widget._label, "size", return_value=QtCore.QSize(0, 0)):
        widget._update_display()  # must not raise; early-returns
