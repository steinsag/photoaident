import pytest
from PySide6 import QtWidgets, QtCore

from photoaident.db.database import Image, ImageMetadata, Face, FaceState, TakenAtSource
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
