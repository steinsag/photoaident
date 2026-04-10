from pathlib import Path
from unittest.mock import patch

from PIL import Image as PILImage
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
    """Widget has a fixed 300×300 size by default."""
    widget = FaceCropWidget()
    qtbot.addWidget(widget)

    assert widget.width() == 300
    assert widget.height() == 300


def test_custom_size(qtbot):
    """Widget respects a custom size argument."""
    widget = FaceCropWidget(size=120)
    qtbot.addWidget(widget)

    assert widget.width() == 120
    assert widget.height() == 120


# ===========================================================================
# load — regeneration path
# ===========================================================================


def _write_real_jpeg(path: Path) -> None:
    """Save a small solid-green PIL JPEG at path (creates parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (80, 60), color=(0, 200, 0)).save(path, "JPEG")


def _make_ensure_side_effect():
    """Return a side-effect for ensure_face_crop that writes a JPEG and returns True."""

    def _side_effect(crop_path, _image_path, _bbox):
        _write_real_jpeg(crop_path)
        return True

    return _side_effect


def test_load_regenerates_crop_when_missing_and_image_exists(qtbot, tmp_path):
    """load() regenerates the crop when crop_path is absent but image_path exists."""
    image_path = tmp_path / "photo.jpg"
    _write_real_jpeg(image_path)
    crop_path = tmp_path / "faces" / "crop.jpg"
    assert not crop_path.exists()

    with patch(
        "photoaident.ui.widgets.face_crop_widget.ensure_face_crop",
        side_effect=_make_ensure_side_effect(),
    ):
        widget = FaceCropWidget()
        qtbot.addWidget(widget)
        widget.load(crop_path, image_path=image_path, bbox=(10, 20, 50, 60))

    assert not widget.pixmap().isNull()
    assert widget.text() == ""


def test_load_calls_ensure_face_crop_with_correct_args(qtbot, tmp_path):
    """load() passes crop_path, image_path, and bbox to ensure_face_crop."""
    image_path = tmp_path / "photo.jpg"
    _write_real_jpeg(image_path)
    crop_path = tmp_path / "faces" / "crop.jpg"
    bbox = (5, 10, 40, 80)

    with patch(
        "photoaident.ui.widgets.face_crop_widget.ensure_face_crop",
        side_effect=_make_ensure_side_effect(),
    ) as mock_ensure:
        widget = FaceCropWidget()
        qtbot.addWidget(widget)
        widget.load(crop_path, image_path=image_path, bbox=bbox)

    mock_ensure.assert_called_once_with(crop_path, image_path, bbox)


def test_load_shows_placeholder_when_image_path_missing(qtbot, tmp_path):
    """load() shows placeholder when image_path does not exist."""
    image_path = tmp_path / "nonexistent_photo.jpg"
    crop_path = tmp_path / "crop.jpg"

    widget = FaceCropWidget()
    qtbot.addWidget(widget)
    widget.load(crop_path, image_path=image_path, bbox=(0, 0, 50, 50))

    assert widget.text() == widget.tr("No image")


def test_load_shows_placeholder_when_no_image_path_given(qtbot, tmp_path):
    """load() shows placeholder when image_path is omitted."""
    crop_path = tmp_path / "crop.jpg"
    assert not crop_path.exists()

    widget = FaceCropWidget()
    qtbot.addWidget(widget)
    widget.load(crop_path)

    assert widget.text() == widget.tr("No image")


def test_load_shows_placeholder_when_bbox_not_given(qtbot, tmp_path):
    """load() shows placeholder when bbox is None even if image_path exists."""
    image_path = tmp_path / "photo.jpg"
    _write_real_jpeg(image_path)
    crop_path = tmp_path / "crop.jpg"

    widget = FaceCropWidget()
    qtbot.addWidget(widget)
    widget.load(crop_path, image_path=image_path)

    assert widget.text() == widget.tr("No image")


def test_load_shows_pixmap_for_existing_valid_crop(qtbot, tmp_path):
    """load() shows the existing pixmap without regeneration when crop is valid."""
    crop_path = tmp_path / "crop.jpg"
    _write_real_jpeg(crop_path)
    image_path = tmp_path / "photo.jpg"
    _write_real_jpeg(image_path)

    widget = FaceCropWidget()
    qtbot.addWidget(widget)
    widget.load(crop_path, image_path=image_path, bbox=(0, 0, 50, 50))

    assert not widget.pixmap().isNull()


def test_load_corrupt_crop_regenerates_when_image_path_given(qtbot, tmp_path):
    """load() heals a corrupt crop when image_path/bbox are provided."""
    crop_path = tmp_path / "faces" / "crop.jpg"
    crop_path.parent.mkdir()
    crop_path.write_bytes(b"not a jpeg")
    image_path = tmp_path / "photo.jpg"
    _write_real_jpeg(image_path)

    with patch(
        "photoaident.ui.widgets.face_crop_widget.ensure_face_crop",
        side_effect=_make_ensure_side_effect(),
    ) as mock_ensure:
        widget = FaceCropWidget()
        qtbot.addWidget(widget)
        widget.load(crop_path, image_path=image_path, bbox=(10, 20, 50, 60))

    mock_ensure.assert_called_once_with(crop_path, image_path, (10, 20, 50, 60))
    assert not widget.pixmap().isNull()
    assert widget.text() == ""


def test_load_corrupt_crop_shows_placeholder_without_image_path(qtbot, tmp_path):
    """load() falls back to placeholder for a corrupt crop when no image_path given."""
    crop_path = tmp_path / "crop.jpg"
    crop_path.write_bytes(b"not a jpeg")

    widget = FaceCropWidget()
    qtbot.addWidget(widget)
    widget.load(crop_path)

    assert widget.pixmap().isNull()
    assert widget.text() == widget.tr("No image")
