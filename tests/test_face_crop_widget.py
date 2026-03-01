from pathlib import Path

from PySide6 import QtGui

from photoaident.ui.widgets.face_crop_widget import FaceCropWidget


def _make_jpeg(path: Path) -> None:
    """Write a minimal valid JPEG file at path."""
    img = QtGui.QImage(80, 60, QtGui.QImage.Format.Format_RGB32)
    img.fill(QtGui.QColor(100, 150, 200))
    img.save(str(path))


# ===========================================================================
# FaceCropWidget
# ===========================================================================


def test_load_valid_path_shows_pixmap(qtbot, tmp_path):
    """load() with a valid JPEG path shows the scaled pixmap and clears text."""
    crop_path = tmp_path / "crop.jpg"
    _make_jpeg(crop_path)

    widget = FaceCropWidget()
    qtbot.addWidget(widget)
    widget.load(crop_path)

    assert not widget.pixmap().isNull()
    assert widget.text() == ""


def test_load_none_shows_placeholder(qtbot):
    """load(None) shows 'No image' placeholder text."""
    widget = FaceCropWidget()
    qtbot.addWidget(widget)
    widget.load(None)

    assert widget.pixmap().isNull()
    assert widget.text() == widget.tr("No image")


def test_load_missing_file_shows_placeholder(qtbot, tmp_path):
    """load() with a non-existent path shows 'No image' placeholder text."""
    widget = FaceCropWidget()
    qtbot.addWidget(widget)
    widget.load(tmp_path / "nonexistent.jpg")

    assert widget.pixmap().isNull()
    assert widget.text() == widget.tr("No image")


def test_load_invalid_file_shows_placeholder(qtbot, tmp_path):
    """load() with a corrupt file shows 'No image' placeholder text."""
    bad_path = tmp_path / "bad.jpg"
    bad_path.write_bytes(b"not a jpeg")

    widget = FaceCropWidget()
    qtbot.addWidget(widget)
    widget.load(bad_path)

    assert widget.pixmap().isNull()
    assert widget.text() == widget.tr("No image")


def test_clear_resets_text_and_pixmap(qtbot, tmp_path):
    """clear() removes the pixmap and any displayed text."""
    crop_path = tmp_path / "crop.jpg"
    _make_jpeg(crop_path)

    widget = FaceCropWidget()
    qtbot.addWidget(widget)
    widget.load(crop_path)
    assert not widget.pixmap().isNull()

    widget.clear()

    assert widget.pixmap().isNull()
    assert widget.text() == ""


def test_fixed_size(qtbot):
    """Widget has a fixed 300×300 size."""
    widget = FaceCropWidget()
    qtbot.addWidget(widget)

    assert widget.width() == 300
    assert widget.height() == 300
