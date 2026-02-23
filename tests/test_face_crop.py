"""Tests for FaceCropWidget."""

import pytest
from PIL import Image as PILImage
from PySide6 import QtGui

from photoaident.ui.widgets.face_crop import FaceCropWidget


@pytest.fixture
def crop_image(tmp_path):
    """A real JPEG file suitable for use as a face crop."""
    path = tmp_path / "crop.jpg"
    PILImage.new("RGB", (100, 100), "red").save(path, "JPEG")
    return path


@pytest.fixture
def thumb_image(tmp_path):
    """A real JPEG file suitable for use as a photo thumbnail."""
    path = tmp_path / "thumb.jpg"
    PILImage.new("RGB", (200, 150), "blue").save(path, "JPEG")
    return path


def test_initial_state_is_clear(qtbot):
    """Widget starts in a blank/clear state."""
    widget = FaceCropWidget()
    qtbot.addWidget(widget)

    assert widget.crop_label.text() == ""
    assert widget.thumb_label.text() == ""
    assert widget.meta_label.text() == ""


def test_clear_resets_widget(qtbot, crop_image, thumb_image):
    """clear() returns the widget to its blank state."""
    widget = FaceCropWidget()
    qtbot.addWidget(widget)

    widget.load(
        crop_path=crop_image,
        thumb_path=thumb_image,
        taken_at="2020-01-01",
        confidence=0.9,
    )
    widget.clear()

    assert widget.crop_label.text() == ""
    assert widget.thumb_label.text() == ""
    assert widget.meta_label.text() == ""


def test_load_with_valid_images_sets_pixmaps(qtbot, crop_image, thumb_image):
    """load() with valid image files sets non-null pixmaps (lines 65-70, 81-87)."""
    widget = FaceCropWidget()
    qtbot.addWidget(widget)

    widget.load(
        crop_path=crop_image,
        thumb_path=thumb_image,
        taken_at="2023-06-15",
        confidence=0.95,
    )

    assert not widget.crop_label.pixmap().isNull()
    assert not widget.thumb_label.pixmap().isNull()
    assert "2023-06-15" in widget.meta_label.text()
    assert "95.0" in widget.meta_label.text()


def test_load_with_corrupt_crop_shows_placeholder(qtbot, tmp_path, thumb_image):
    """Unreadable crop file shows 'No image' placeholder (lines 73-74)."""
    bad_crop = tmp_path / "corrupt_crop.jpg"
    bad_crop.write_bytes(b"not a jpeg")

    widget = FaceCropWidget()
    qtbot.addWidget(widget)

    widget.load(
        crop_path=bad_crop,
        thumb_path=thumb_image,
        taken_at="2023-01-01",
        confidence=0.8,
    )

    # Null pixmap → falls into else → "No image"
    assert widget.crop_label.pixmap().isNull()
    assert widget.crop_label.text() == widget.tr("No image")


def test_load_with_corrupt_thumb_shows_placeholder(qtbot, tmp_path, crop_image):
    """Unreadable thumbnail file shows 'No thumbnail' placeholder (lines 89-90)."""
    bad_thumb = tmp_path / "corrupt_thumb.jpg"
    bad_thumb.write_bytes(b"not a jpeg")

    widget = FaceCropWidget()
    qtbot.addWidget(widget)

    widget.load(
        crop_path=crop_image,
        thumb_path=bad_thumb,
        taken_at="2023-01-01",
        confidence=0.7,
    )

    assert widget.thumb_label.pixmap().isNull()
    assert widget.thumb_label.text() == widget.tr("No thumbnail")


def test_load_with_none_paths_shows_placeholders(qtbot):
    """None paths fall into else branches and show placeholder text."""
    widget = FaceCropWidget()
    qtbot.addWidget(widget)

    widget.load(
        crop_path=None,
        thumb_path=None,
        taken_at="2022-12-25",
        confidence=0.5,
    )

    assert widget.crop_label.text() == widget.tr("No image")
    assert widget.thumb_label.text() == widget.tr("No thumbnail")
