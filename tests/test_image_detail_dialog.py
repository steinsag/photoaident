from datetime import datetime

import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from photoaident.db.database import Face, FaceState, Image, ImageMetadata, TakenAtSource
from photoaident.ui.widgets.image_detail_dialog import ImageDetailDialog


@pytest.fixture
def sample_image_with_metadata(tmp_path):
    """Create a temporary image file and a DB model for it."""
    img_path = tmp_path / "test_image.jpg"
    # Create a real small JPEG
    from PIL import Image as PILImage

    img = PILImage.new("RGB", (1000, 800), color="red")
    img.save(img_path)

    db_image = Image(
        id=123,
        file_path=str(img_path),
        file_hash="fakehash",
        file_size=1024,
    )
    db_image.metadata_rel = ImageMetadata(
        width=1000,
        height=800,
        camera_make="TestCamera",
        camera_model="Model X",
        taken_at_source=TakenAtSource.FILESYSTEM,
    )
    db_image.faces = [
        Face(
            bbox_x=100,
            bbox_y=100,
            bbox_w=50,
            bbox_h=50,
            detection_confidence=0.9,
            model_version="v1",
            faiss_id=0,
            state=FaceState.UNIDENTIFIED,
        )
    ]
    return db_image


def test_image_detail_dialog_init(qtbot, sample_image_with_metadata):
    dialog = ImageDetailDialog(sample_image_with_metadata)
    qtbot.add_widget(dialog)

    # Check metadata display (look for some expected strings)
    labels = dialog.findChildren(QtWidgets.QLabel)
    texts = [label.text() for label in labels]

    assert any("123" in t for t in texts)  # ID
    assert any("TestCamera" in t for t in texts)  # Camera make
    assert any("1000 x 800" in t for t in texts)  # Dimensions
    assert any("1.0 KB" in t for t in texts)  # File size formatted

    # Check if image is loaded
    assert not dialog.image_label.pixmap().isNull()


def test_image_detail_dialog_missing_file(qtbot):
    db_image = Image(
        id=456,
        file_path="/non/existent/path.jpg",
        file_size=0,
    )
    dialog = ImageDetailDialog(db_image)
    qtbot.add_widget(dialog)

    assert "not found" in dialog.image_label.text().lower()


def test_image_detail_dialog_close(qtbot, sample_image_with_metadata):
    dialog = ImageDetailDialog(sample_image_with_metadata)
    qtbot.add_widget(dialog)

    # Close button should accept the dialog
    close_button = None
    for button in dialog.findChildren(QtWidgets.QPushButton):
        if "Close" in button.text():
            close_button = button
            break

    assert close_button is not None

    with qtbot.wait_signal(dialog.finished):
        qtbot.mouseClick(close_button, QtCore.Qt.MouseButton.LeftButton)


def test_large_file_size_shows_mb(qtbot, tmp_path):
    """File size ≥ 1 MB is displayed in MB units."""
    from PIL import Image as PILImage

    img_path = tmp_path / "big.jpg"
    PILImage.new("RGB", (10, 10), "green").save(img_path)

    db_image = Image(
        id=200,
        file_path=str(img_path),
        file_size=2 * 1024 * 1024,  # 2 MB
    )
    dialog = ImageDetailDialog(db_image)
    qtbot.add_widget(dialog)

    labels = dialog.findChildren(QtWidgets.QLabel)
    texts = [label.text() for label in labels]
    assert any("MB" in t for t in texts)


def test_taken_at_in_metadata_is_displayed(qtbot, tmp_path):
    """taken_at field in ImageMetadata appears in the dialog."""
    from PIL import Image as PILImage

    img_path = tmp_path / "dated.jpg"
    PILImage.new("RGB", (10, 10), "yellow").save(img_path)

    db_image = Image(id=300, file_path=str(img_path), file_size=500)
    db_image.metadata_rel = ImageMetadata(
        width=100,
        height=100,
        taken_at=datetime(2021, 3, 14, 15, 9, 26),
        taken_at_source=TakenAtSource.EXIF,
    )
    db_image.faces = []

    dialog = ImageDetailDialog(db_image)
    qtbot.add_widget(dialog)

    labels = dialog.findChildren(QtWidgets.QLabel)
    texts = [label.text() for label in labels]
    assert any("2021-03-14" in t for t in texts)


def test_load_image_failure_shows_error(qtbot, tmp_path):
    """A file with invalid image content shows a failure message."""
    bad_file = tmp_path / "bad.jpg"
    bad_file.write_bytes(b"this is not a valid jpeg")

    db_image = Image(id=400, file_path=str(bad_file), file_size=25)
    dialog = ImageDetailDialog(db_image)
    qtbot.add_widget(dialog)

    text = dialog.image_label.text().lower()
    assert "failed" in text


def test_update_image_display_without_pixmap_is_noop(qtbot):
    """_update_image_display returns early if _original_pixmap is not set."""
    db_image = Image(id=500, file_path="/nonexistent.jpg", file_size=0)
    dialog = ImageDetailDialog(db_image)
    qtbot.add_widget(dialog)

    assert not hasattr(dialog, "_original_pixmap")
    dialog._update_image_display()  # must not raise


def test_resize_event_schedules_redisplay(qtbot, sample_image_with_metadata):
    """resizeEvent does not raise and schedules a display update."""
    dialog = ImageDetailDialog(sample_image_with_metadata)
    qtbot.add_widget(dialog)

    event = QtGui.QResizeEvent(QtCore.QSize(900, 700), QtCore.QSize(800, 600))
    dialog.resizeEvent(event)  # must not raise


def test_label_faces_button_enabled_when_unidentified_faces(
    qtbot, sample_image_with_metadata
):
    """Label button is enabled when the image has at least one unidentified face."""
    dialog = ImageDetailDialog(sample_image_with_metadata)
    qtbot.add_widget(dialog)

    buttons = dialog.findChildren(QtWidgets.QPushButton)
    label_btn = next((b for b in buttons if "Label" in b.text()), None)
    assert label_btn is not None
    assert label_btn.isEnabled()


def test_label_faces_button_disabled_when_no_unidentified_faces(qtbot, tmp_path):
    """Label button is disabled when the image has no unidentified faces."""
    from PIL import Image as PILImage

    img_path = tmp_path / "identified.jpg"
    PILImage.new("RGB", (100, 100), "blue").save(img_path)

    db_image = Image(id=600, file_path=str(img_path), file_size=500)
    db_image.faces = [
        Face(
            bbox_x=0,
            bbox_y=0,
            bbox_w=50,
            bbox_h=50,
            detection_confidence=0.9,
            model_version="v1",
            faiss_id=1,
            state=FaceState.IDENTIFIED,
        )
    ]

    dialog = ImageDetailDialog(db_image)
    qtbot.add_widget(dialog)

    buttons = dialog.findChildren(QtWidgets.QPushButton)
    label_btn = next((b for b in buttons if "Label" in b.text()), None)
    assert label_btn is not None
    assert not label_btn.isEnabled()


def test_label_faces_button_emits_signal(qtbot, tmp_path):
    """Clicking the label button emits navigate_to_labelling with the image id."""
    from PIL import Image as PILImage

    img_path = tmp_path / "signal_test.jpg"
    PILImage.new("RGB", (100, 100), "green").save(img_path)

    db_image = Image(id=700, file_path=str(img_path), file_size=500)
    db_image.faces = [
        Face(
            bbox_x=0,
            bbox_y=0,
            bbox_w=50,
            bbox_h=50,
            detection_confidence=0.9,
            model_version="v1",
            faiss_id=2,
            state=FaceState.UNIDENTIFIED,
        )
    ]

    dialog = ImageDetailDialog(db_image)
    qtbot.add_widget(dialog)

    emitted_ids: list[int] = []
    dialog.navigate_to_labelling.connect(emitted_ids.append)

    buttons = dialog.findChildren(QtWidgets.QPushButton)
    label_btn = next((b for b in buttons if "Label" in b.text()), None)
    assert label_btn is not None

    qtbot.mouseClick(label_btn, QtCore.Qt.MouseButton.LeftButton)

    assert emitted_ids == [700]
