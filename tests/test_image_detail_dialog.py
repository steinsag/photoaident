from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from PySide6 import QtCore, QtGui, QtQuickWidgets, QtWidgets

from photoaident.db.database import (
    Face,
    FaceState,
    Image,
    ImageMetadata,
    Person,
    TakenAtSource,
)
from photoaident.db.vector_store import VectorStore
from photoaident.ui.widgets.image_detail_dialog import (
    ImageDetailDialog,
    _FaceOverlayLabel,
)
from photoaident.ui.widgets.map_widget import MapWidget


@pytest.fixture
def mock_session_factory():
    return MagicMock()


@pytest.fixture
def mock_vector_store():
    return MagicMock(spec=VectorStore)


def test_image_detail_dialog_init(
    qtbot,
    sample_image_with_metadata,
    mock_session_factory,
    mock_vector_store,
    tmp_app_paths,
):
    dialog = ImageDetailDialog(
        sample_image_with_metadata,
        mock_session_factory,
        mock_vector_store,
        tmp_app_paths,
    )
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


def test_image_detail_dialog_missing_file(
    qtbot, mock_session_factory, mock_vector_store, tmp_app_paths
):
    db_image = Image(
        id=456,
        file_path="/non/existent/path.jpg",
        file_size=0,
    )
    dialog = ImageDetailDialog(
        db_image, mock_session_factory, mock_vector_store, tmp_app_paths
    )
    qtbot.add_widget(dialog)

    assert "not found" in dialog.image_label.text().lower()


def test_image_detail_dialog_close(
    qtbot,
    sample_image_with_metadata,
    mock_session_factory,
    mock_vector_store,
    tmp_app_paths,
):
    dialog = ImageDetailDialog(
        sample_image_with_metadata,
        mock_session_factory,
        mock_vector_store,
        tmp_app_paths,
    )
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


def test_large_file_size_shows_mb(
    qtbot, tmp_path, mock_session_factory, mock_vector_store, tmp_app_paths
):
    """File size ≥ 1 MB is displayed in MB units."""
    from PIL import Image as PILImage

    img_path = tmp_path / "big.jpg"
    PILImage.new("RGB", (10, 10), "green").save(img_path)

    db_image = Image(
        id=200,
        file_path=str(img_path),
        file_size=2 * 1024 * 1024,  # 2 MB
    )
    dialog = ImageDetailDialog(
        db_image, mock_session_factory, mock_vector_store, tmp_app_paths
    )
    qtbot.add_widget(dialog)

    labels = dialog.findChildren(QtWidgets.QLabel)
    texts = [label.text() for label in labels]
    assert any("MB" in t for t in texts)


def test_taken_at_in_metadata_is_displayed(
    qtbot, tmp_path, mock_session_factory, mock_vector_store, tmp_app_paths
):
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

    dialog = ImageDetailDialog(
        db_image, mock_session_factory, mock_vector_store, tmp_app_paths
    )
    qtbot.add_widget(dialog)

    labels = dialog.findChildren(QtWidgets.QLabel)
    texts = [label.text() for label in labels]
    assert any("2021-03-14" in t for t in texts)


def test_load_image_failure_shows_error(
    qtbot, tmp_path, mock_session_factory, mock_vector_store, tmp_app_paths
):
    """A file with invalid image content shows a failure message."""
    bad_file = tmp_path / "bad.jpg"
    bad_file.write_bytes(b"this is not a valid jpeg")

    db_image = Image(id=400, file_path=str(bad_file), file_size=25)
    dialog = ImageDetailDialog(
        db_image, mock_session_factory, mock_vector_store, tmp_app_paths
    )
    qtbot.add_widget(dialog)

    text = dialog.image_label.text().lower()
    assert "failed" in text


def test_update_image_display_without_pixmap_is_noop(
    qtbot, mock_session_factory, mock_vector_store, tmp_app_paths
):
    """_update_image_display returns early if _original_pixmap is not set."""
    db_image = Image(id=500, file_path="/nonexistent.jpg", file_size=0)
    dialog = ImageDetailDialog(
        db_image, mock_session_factory, mock_vector_store, tmp_app_paths
    )
    qtbot.add_widget(dialog)

    assert not hasattr(dialog, "_original_pixmap")
    dialog._update_image_display()  # must not raise


def test_resize_event_schedules_redisplay(
    qtbot,
    sample_image_with_metadata,
    mock_session_factory,
    mock_vector_store,
    tmp_app_paths,
):
    """resizeEvent does not raise and schedules a display update."""
    dialog = ImageDetailDialog(
        sample_image_with_metadata,
        mock_session_factory,
        mock_vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    event = QtGui.QResizeEvent(QtCore.QSize(900, 700), QtCore.QSize(800, 600))
    dialog.resizeEvent(event)  # must not raise


def test_label_faces_button_enabled_when_unidentified_faces(
    qtbot,
    sample_image_with_metadata,
    mock_session_factory,
    mock_vector_store,
    tmp_app_paths,
):
    """Label button is enabled when the image has at least one unidentified face."""
    dialog = ImageDetailDialog(
        sample_image_with_metadata,
        mock_session_factory,
        mock_vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    buttons = dialog.findChildren(QtWidgets.QPushButton)
    label_btn = next((b for b in buttons if "Label" in b.text()), None)
    assert label_btn is not None
    assert label_btn.isEnabled()


def test_label_faces_button_disabled_when_no_unidentified_faces(
    qtbot, tmp_path, mock_session_factory, mock_vector_store, tmp_app_paths
):
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
            state=FaceState.IDENTIFIED,
        )
    ]

    dialog = ImageDetailDialog(
        db_image, mock_session_factory, mock_vector_store, tmp_app_paths
    )
    qtbot.add_widget(dialog)

    buttons = dialog.findChildren(QtWidgets.QPushButton)
    label_btn = next((b for b in buttons if "Label" in b.text()), None)
    assert label_btn is not None
    assert not label_btn.isEnabled()


def test_label_faces_button_emits_signal(
    qtbot, tmp_path, mock_session_factory, mock_vector_store, tmp_app_paths
):
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
            state=FaceState.UNIDENTIFIED,
        )
    ]

    dialog = ImageDetailDialog(
        db_image, mock_session_factory, mock_vector_store, tmp_app_paths
    )
    qtbot.add_widget(dialog)

    emitted_ids: list[int] = []
    dialog.navigate_to_labelling.connect(emitted_ids.append)

    buttons = dialog.findChildren(QtWidgets.QPushButton)
    label_btn = next((b for b in buttons if "Label" in b.text()), None)
    assert label_btn is not None

    qtbot.mouseClick(label_btn, QtCore.Qt.MouseButton.LeftButton)

    assert emitted_ids == [700]


# ===========================================================================
# show_in_file_manager_button
# ===========================================================================


def test_show_in_file_manager_button_exists(
    qtbot,
    sample_image_with_metadata,
    mock_session_factory,
    mock_vector_store,
    tmp_app_paths,
):
    """Show in File Manager button is present in the dialog."""
    dialog = ImageDetailDialog(
        sample_image_with_metadata,
        mock_session_factory,
        mock_vector_store,
        tmp_app_paths,
    )
    qtbot.addWidget(dialog)

    buttons = dialog.findChildren(QtWidgets.QPushButton)
    show_btn = next(
        (b for b in buttons if b.text() == dialog.tr("Show in File Manager")), None
    )
    assert show_btn is not None


def test_show_in_file_manager_button_calls_reveal(
    qtbot,
    sample_image_with_metadata,
    mock_session_factory,
    mock_vector_store,
    tmp_app_paths,
):
    """Clicking the button calls reveal_in_file_manager with the image file path."""
    dialog = ImageDetailDialog(
        sample_image_with_metadata,
        mock_session_factory,
        mock_vector_store,
        tmp_app_paths,
    )
    qtbot.addWidget(dialog)

    buttons = dialog.findChildren(QtWidgets.QPushButton)
    show_btn = next(
        (b for b in buttons if b.text() == dialog.tr("Show in File Manager")), None
    )
    assert show_btn is not None

    target = "photoaident.ui.widgets.image_detail_dialog.reveal_in_file_manager"
    with patch(target) as mock_reveal:
        qtbot.mouseClick(show_btn, QtCore.Qt.MouseButton.LeftButton)

    mock_reveal.assert_called_once_with(sample_image_with_metadata.file_path)


def test_show_in_file_manager_button_always_enabled(
    qtbot, tmp_path, mock_session_factory, mock_vector_store, tmp_app_paths
):
    """Show in File Manager button is enabled regardless of face labelling state."""
    from PIL import Image as PILImage

    img_path = tmp_path / "all_identified.jpg"
    PILImage.new("RGB", (100, 100), "cyan").save(img_path)

    db_image = Image(id=800, file_path=str(img_path), file_size=500)
    db_image.faces = [
        Face(
            bbox_x=0,
            bbox_y=0,
            bbox_w=50,
            bbox_h=50,
            detection_confidence=0.95,
            model_version="v1",
            state=FaceState.IDENTIFIED,
        )
    ]

    dialog = ImageDetailDialog(
        db_image, mock_session_factory, mock_vector_store, tmp_app_paths
    )
    qtbot.addWidget(dialog)

    buttons = dialog.findChildren(QtWidgets.QPushButton)
    show_btn = next(
        (b for b in buttons if b.text() == dialog.tr("Show in File Manager")), None
    )
    assert show_btn is not None
    assert show_btn.isEnabled()


# ===========================================================================
# navigate_to_browse
# ===========================================================================


def test_browse_photo_folder_button_emits_navigate_to_browse(
    qtbot, tmp_path, mock_session_factory, mock_vector_store, tmp_app_paths
):
    """Clicking 'Browse Photo Folder' emits navigate_to_browse with the file path."""
    from PIL import Image as PILImage

    img_path = tmp_path / "browse_signal.jpg"
    PILImage.new("RGB", (100, 100), "red").save(img_path)

    db_image = Image(id=1100, file_path=str(img_path), file_size=500)
    db_image.faces = []

    dialog = ImageDetailDialog(
        db_image, mock_session_factory, mock_vector_store, tmp_app_paths
    )
    qtbot.addWidget(dialog)

    received: list[str] = []
    dialog.navigate_to_browse.connect(received.append)

    browse_btn = next(
        (
            b
            for b in dialog.findChildren(QtWidgets.QPushButton)
            if b.text() == dialog.tr("Browse Photo Folder")
        ),
        None,
    )
    assert browse_btn is not None

    # The button calls accept() before emitting; use waitSignal on finished so
    # we catch both the dialog closing and the signal in the correct order.
    with qtbot.waitSignal(dialog.finished):
        qtbot.mouseClick(browse_btn, QtCore.Qt.MouseButton.LeftButton)

    assert received == [str(img_path)]


def test_file_path_link_emits_navigate_to_browse(
    qtbot, tmp_path, mock_session_factory, mock_vector_store, tmp_app_paths
):
    """Activating the file path link in the metadata panel emits navigate_to_browse."""
    from PIL import Image as PILImage

    img_path = tmp_path / "link_signal.jpg"
    PILImage.new("RGB", (100, 100), "blue").save(img_path)

    db_image = Image(id=1101, file_path=str(img_path), file_size=500)
    db_image.faces = []

    dialog = ImageDetailDialog(
        db_image, mock_session_factory, mock_vector_store, tmp_app_paths
    )
    qtbot.addWidget(dialog)

    received: list[str] = []
    dialog.navigate_to_browse.connect(received.append)

    # Locate the clickable link label — it is the QLabel whose text contains
    # an <a href="#"> wrapping the (HTML-escaped) file path.
    file_path_label = next(
        (
            lbl
            for lbl in dialog.findChildren(QtWidgets.QLabel)
            if 'href="#"' in lbl.text()
        ),
        None,
    )
    assert file_path_label is not None

    # Emit linkActivated directly; simulating a real mouse click on a QLabel
    # HTML link is not reliable in headless tests.
    # The handler calls accept() before emitting; use waitSignal on finished so
    # we catch both the dialog closing and the signal in the correct order.
    with qtbot.waitSignal(dialog.finished):
        file_path_label.linkActivated.emit("#")

    assert received == [str(img_path)]
    assert dialog.isHidden()


# ===========================================================================
# Tooltip tests — _build_face_display_info
# ===========================================================================


def test_face_tooltip_identified_shows_person_name(
    qtbot, tmp_path, mock_session_factory, mock_vector_store, tmp_app_paths
):
    """_build_face_display_info returns the person's name for IDENTIFIED faces."""
    from PIL import Image as PILImage

    img_path = tmp_path / "identified.jpg"
    PILImage.new("RGB", (200, 200), "white").save(img_path)

    person = Person(id=1, name="Alice")
    face = Face(
        bbox_x=10,
        bbox_y=10,
        bbox_w=50,
        bbox_h=50,
        detection_confidence=0.9,
        model_version="v1",
        state=FaceState.IDENTIFIED,
    )
    face.person = person

    db_image = Image(id=1, file_path=str(img_path), file_size=500)
    db_image.faces = [face]

    dialog = ImageDetailDialog(
        db_image, mock_session_factory, mock_vector_store, tmp_app_paths
    )
    qtbot.add_widget(dialog)

    regions = dialog._build_face_display_info()
    assert len(regions) == 1
    assert regions[0][2] == "Alice"
    assert regions[0][1] == QtCore.Qt.GlobalColor.green


def test_face_tooltip_anonymous_shows_anonymous(
    qtbot, tmp_path, mock_session_factory, mock_vector_store, tmp_app_paths
):
    """_build_face_display_info returns 'Anonymous' for ANONYMOUS faces."""
    from PIL import Image as PILImage

    img_path = tmp_path / "anon.jpg"
    PILImage.new("RGB", (200, 200), "white").save(img_path)

    face = Face(
        bbox_x=10,
        bbox_y=10,
        bbox_w=50,
        bbox_h=50,
        detection_confidence=0.9,
        model_version="v1",
        state=FaceState.ANONYMOUS,
    )

    db_image = Image(id=2, file_path=str(img_path), file_size=500)
    db_image.faces = [face]

    dialog = ImageDetailDialog(
        db_image, mock_session_factory, mock_vector_store, tmp_app_paths
    )
    qtbot.add_widget(dialog)

    regions = dialog._build_face_display_info()
    assert len(regions) == 1
    assert regions[0][2] == dialog.tr("Anonymous")
    assert regions[0][1] == QtCore.Qt.GlobalColor.green


def test_face_tooltip_unidentified_no_vector_match(
    qtbot, tmp_path, mock_session_factory, mock_vector_store, tmp_app_paths
):
    """_build_face_display_info: UNIDENTIFIED with no vector match → Unknown + red."""
    # This covers the case where _resolve_best_person_name returns None
    # even though a vector_store instance is available.
    from PIL import Image as PILImage

    img_path = tmp_path / "unknown.jpg"
    PILImage.new("RGB", (200, 200), "white").save(img_path)

    face = Face(
        bbox_x=10,
        bbox_y=10,
        bbox_w=50,
        bbox_h=50,
        detection_confidence=0.9,
        model_version="v1",
        state=FaceState.UNIDENTIFIED,
    )

    db_image = Image(id=3, file_path=str(img_path), file_size=500)
    db_image.faces = [face]

    dialog = ImageDetailDialog(
        db_image, mock_session_factory, mock_vector_store, tmp_app_paths
    )
    qtbot.add_widget(dialog)

    regions = dialog._build_face_display_info()
    assert len(regions) == 1
    assert regions[0][2] == dialog.tr("Unknown")
    assert regions[0][1] == QtCore.Qt.GlobalColor.red


def test_face_tooltip_deleted_face_excluded(
    qtbot, tmp_path, mock_session_factory, mock_vector_store, tmp_app_paths
):
    """_build_face_display_info skips faces where deleted_at is set."""
    from PIL import Image as PILImage

    img_path = tmp_path / "deleted.jpg"
    PILImage.new("RGB", (200, 200), "white").save(img_path)

    active_face = Face(
        bbox_x=10,
        bbox_y=10,
        bbox_w=50,
        bbox_h=50,
        detection_confidence=0.9,
        model_version="v1",
        state=FaceState.UNIDENTIFIED,
    )
    deleted_face = Face(
        bbox_x=100,
        bbox_y=100,
        bbox_w=50,
        bbox_h=50,
        detection_confidence=0.8,
        model_version="v1",
        state=FaceState.UNIDENTIFIED,
        deleted_at=datetime(2024, 1, 1),
    )

    db_image = Image(id=4, file_path=str(img_path), file_size=500)
    db_image.faces = [active_face, deleted_face]

    dialog = ImageDetailDialog(
        db_image, mock_session_factory, mock_vector_store, tmp_app_paths
    )
    qtbot.add_widget(dialog)

    regions = dialog._build_face_display_info()
    assert len(regions) == 1
    assert regions[0][0] == QtCore.QRectF(10, 10, 50, 50)


# ===========================================================================
# _FaceOverlayLabel hit/miss detection
# ===========================================================================


def test_face_overlay_label_hit_detection(qtbot):
    """Mouse inside a face bbox triggers QToolTip.showText."""
    label = _FaceOverlayLabel()
    qtbot.add_widget(label)

    # Set a pixmap and face region
    pixmap = QtGui.QPixmap(200, 200)
    pixmap.fill(QtGui.QColor("white"))
    label.setPixmap(pixmap)
    label.resize(200, 200)

    region = (QtCore.QRectF(50, 50, 60, 60), "Alice")
    label.set_face_regions([region], QtCore.QSize(200, 200))

    with patch(
        "photoaident.ui.widgets.image_detail_dialog.QtWidgets.QToolTip.showText"
    ) as mock_show:
        # Simulate mouse move inside the bbox (center = 80, 80)
        event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(80.0, 80.0),
            QtCore.QPointF(80.0, 80.0),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        label.mouseMoveEvent(event)
        mock_show.assert_called_once()
        assert mock_show.call_args[0][1] == "Alice"


def test_face_overlay_label_no_regions_early_return(qtbot):
    """mouseMoveEvent returns early when no face regions are set."""
    label = _FaceOverlayLabel()
    qtbot.add_widget(label)

    pixmap = QtGui.QPixmap(200, 200)
    pixmap.fill(QtGui.QColor("white"))
    label.setPixmap(pixmap)
    label.resize(200, 200)
    # Do NOT call set_face_regions — _face_regions stays empty

    with (
        patch(
            "photoaident.ui.widgets.image_detail_dialog.QtWidgets.QToolTip.showText"
        ) as mock_show,
        patch(
            "photoaident.ui.widgets.image_detail_dialog.QtWidgets.QToolTip.hideText"
        ) as mock_hide,
    ):
        event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(80.0, 80.0),
            QtCore.QPointF(80.0, 80.0),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        label.mouseMoveEvent(event)
        mock_show.assert_not_called()
        mock_hide.assert_not_called()


def test_face_overlay_label_no_pixmap_early_return(qtbot):
    """mouseMoveEvent returns early when pixmap is null."""
    label = _FaceOverlayLabel()
    qtbot.add_widget(label)
    label.resize(200, 200)

    region = (QtCore.QRectF(50, 50, 60, 60), "Alice")
    label.set_face_regions([region], QtCore.QSize(200, 200))
    # No pixmap set — label.pixmap() returns a null pixmap

    with (
        patch(
            "photoaident.ui.widgets.image_detail_dialog.QtWidgets.QToolTip.showText"
        ) as mock_show,
        patch(
            "photoaident.ui.widgets.image_detail_dialog.QtWidgets.QToolTip.hideText"
        ) as mock_hide,
    ):
        event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(80.0, 80.0),
            QtCore.QPointF(80.0, 80.0),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        label.mouseMoveEvent(event)
        mock_show.assert_not_called()
        mock_hide.assert_not_called()


def test_face_overlay_label_zero_size_pixmap_early_return(qtbot):
    """mouseMoveEvent returns early when pixmap reports zero dimensions."""
    label = _FaceOverlayLabel()
    qtbot.add_widget(label)
    label.resize(200, 200)

    region = (QtCore.QRectF(50, 50, 60, 60), "Alice")
    label.set_face_regions([region], QtCore.QSize(200, 200))

    mock_pm = MagicMock()
    mock_pm.isNull.return_value = False
    mock_pm.width.return_value = 0
    mock_pm.height.return_value = 0

    with (
        patch.object(label, "pixmap", return_value=mock_pm),
        patch(
            "photoaident.ui.widgets.image_detail_dialog.QtWidgets.QToolTip.showText"
        ) as mock_show,
        patch(
            "photoaident.ui.widgets.image_detail_dialog.QtWidgets.QToolTip.hideText"
        ) as mock_hide,
    ):
        event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(80.0, 80.0),
            QtCore.QPointF(80.0, 80.0),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        label.mouseMoveEvent(event)
        mock_show.assert_not_called()
        mock_hide.assert_not_called()


def test_face_overlay_label_miss_detection(qtbot):
    """Mouse outside all face bboxes calls QToolTip.hideText."""
    label = _FaceOverlayLabel()
    qtbot.add_widget(label)

    pixmap = QtGui.QPixmap(200, 200)
    pixmap.fill(QtGui.QColor("white"))
    label.setPixmap(pixmap)
    label.resize(200, 200)

    region = (QtCore.QRectF(50, 50, 60, 60), "Alice")
    label.set_face_regions([region], QtCore.QSize(200, 200))

    with patch(
        "photoaident.ui.widgets.image_detail_dialog.QtWidgets.QToolTip.hideText"
    ) as mock_hide:
        # Simulate mouse move outside the bbox (10, 10 is outside 50-110 range)
        event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(10.0, 10.0),
            QtCore.QPointF(10.0, 10.0),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        label.mouseMoveEvent(event)
        mock_hide.assert_called_once()


# ===========================================================================
# Bounding box colors by state
# ===========================================================================


def test_face_tooltip_unidentified_with_vector_store_match(
    qtbot, tmp_path, tmp_app_paths
):
    """UNIDENTIFIED face with a FAISS match shows name + percentage and green color."""
    from PIL import Image as PILImage

    img_path = tmp_path / "matched.jpg"
    PILImage.new("RGB", (200, 200), "white").save(img_path)

    face = Face(
        id=5,
        bbox_x=10,
        bbox_y=10,
        bbox_w=50,
        bbox_h=50,
        detection_confidence=0.9,
        model_version="v1",
        state=FaceState.UNIDENTIFIED,
    )
    db_image = Image(id=10, file_path=str(img_path), file_size=500)
    db_image.faces = [face]

    mock_sf = MagicMock()
    mock_vs = MagicMock()

    with patch(
        "photoaident.ui.widgets.image_detail_dialog.resolve_faces_to_persons",
        return_value={5: ("Dave", 0.82)},
    ):
        dialog = ImageDetailDialog(
            db_image, session_factory=mock_sf, vector_store=mock_vs, paths=tmp_app_paths
        )
        qtbot.add_widget(dialog)
        regions = dialog._build_face_display_info()

    assert len(regions) == 1
    assert regions[0][2] == "Dave (82%)"
    assert regions[0][1] == QtCore.Qt.GlobalColor.green


def test_face_tooltip_unidentified_with_vector_store_no_match(
    qtbot, tmp_path, tmp_app_paths
):
    """UNIDENTIFIED face with no FAISS match shows Unknown and red color."""
    from PIL import Image as PILImage

    img_path = tmp_path / "unmatched.jpg"
    PILImage.new("RGB", (200, 200), "white").save(img_path)

    face = Face(
        id=6,
        bbox_x=10,
        bbox_y=10,
        bbox_w=50,
        bbox_h=50,
        detection_confidence=0.9,
        model_version="v1",
        state=FaceState.UNIDENTIFIED,
    )
    db_image = Image(id=11, file_path=str(img_path), file_size=500)
    db_image.faces = [face]

    mock_sf = MagicMock()
    mock_vs = MagicMock()

    with patch(
        "photoaident.ui.widgets.image_detail_dialog.resolve_faces_to_persons",
        return_value={6: None},
    ):
        dialog = ImageDetailDialog(
            db_image, session_factory=mock_sf, vector_store=mock_vs, paths=tmp_app_paths
        )
        qtbot.add_widget(dialog)
        regions = dialog._build_face_display_info()

    assert len(regions) == 1
    assert regions[0][2] == dialog.tr("Unknown")
    assert regions[0][1] == QtCore.Qt.GlobalColor.red


def test_bounding_box_colors_by_state(
    qtbot, tmp_path, mock_session_factory, mock_vector_store, tmp_app_paths
):
    """Verify that face boxes are drawn in different colors for different states."""
    from PIL import Image as PILImage

    img_path = tmp_path / "color_test.jpg"
    PILImage.new("RGB", (200, 200), "white").save(img_path)

    person = Person(id=1, name="Carol")
    identified_face = Face(
        bbox_x=10,
        bbox_y=10,
        bbox_w=40,
        bbox_h=40,
        detection_confidence=0.9,
        model_version="v1",
        state=FaceState.IDENTIFIED,
    )
    identified_face.person = person

    anonymous_face = Face(
        bbox_x=70,
        bbox_y=10,
        bbox_w=40,
        bbox_h=40,
        detection_confidence=0.9,
        model_version="v1",
        state=FaceState.ANONYMOUS,
    )

    unidentified_face = Face(
        bbox_x=130,
        bbox_y=10,
        bbox_w=40,
        bbox_h=40,
        detection_confidence=0.9,
        model_version="v1",
        state=FaceState.UNIDENTIFIED,
    )

    db_image = Image(id=5, file_path=str(img_path), file_size=500)
    db_image.faces = [identified_face, anonymous_face, unidentified_face]

    dialog = ImageDetailDialog(
        db_image, mock_session_factory, mock_vector_store, tmp_app_paths
    )
    qtbot.add_widget(dialog)

    # The pixmap has bounding boxes drawn; verify it's not null and loaded correctly
    assert hasattr(dialog, "_original_pixmap")
    assert not dialog._original_pixmap.isNull()

    # Sample pixels at box centers to verify colors
    img = dialog._original_pixmap.toImage()

    def pixel_color(x: int, y: int) -> QtGui.QColor:
        return QtGui.QColor(img.pixel(x, y))

    # Green box at identified face border (top edge y=10, center x=30)
    green_identified = pixel_color(30, 10)
    assert green_identified.green() > green_identified.red()
    assert green_identified.green() > green_identified.blue()

    # Green box at anonymous face border (top edge y=10, center x=90) — same color
    green_anon = pixel_color(90, 10)
    assert green_anon.green() > green_anon.red()
    assert green_anon.green() > green_anon.blue()

    # Red box at unidentified face border (top edge y=10, center x=150)
    red = pixel_color(150, 10)
    assert red.red() > red.green() and red.red() > red.blue()


def test_image_detail_dialog_with_exif_rotation(
    qtbot, tmp_path, mock_session_factory, mock_vector_store, tmp_app_paths
):
    """Image with EXIF rotation triggers the transformation block."""
    from PIL import Image as PILImage

    img_path = tmp_path / "rotated.jpg"
    # Create an image and save it with orientation metadata.
    # Orientation 6 is Rotate 90 CW
    img = PILImage.new("RGB", (100, 200), "blue")
    exif = img.getexif()
    exif[0x0112] = 6  # Orientation tag
    img.save(img_path, exif=exif)

    db_image = Image(id=900, file_path=str(img_path), file_size=1000)
    db_image.faces = [
        Face(
            bbox_x=10,
            bbox_y=10,
            bbox_w=20,
            bbox_h=20,
            detection_confidence=0.9,
            model_version="v1",
            state=FaceState.IDENTIFIED,
        )
    ]

    dialog = ImageDetailDialog(
        db_image, mock_session_factory, mock_vector_store, tmp_app_paths
    )
    qtbot.add_widget(dialog)

    # If the EXIF rotation transformation block was executed, the
    # pixmap dimensions should be swapped (100x200 -> 200x100)
    # PIL (100, 200) -> QImageReader reads it.
    # Orientation 6 is Rotate 90.
    assert hasattr(dialog, "_original_pixmap")
    pm = dialog._original_pixmap
    # After 90 degree rotation, width should be 200 and height 100
    assert pm.width() == 200
    assert pm.height() == 100


# ===========================================================================
# GPS map widget presence tests
# ===========================================================================


def test_map_shown_when_gps_available(
    qtbot, tmp_path, mock_session_factory, mock_vector_store, tmp_app_paths
):
    """A MapWidget is added to the metadata panel when GPS coordinates are present."""
    from PIL import Image as PILImage

    img_path = tmp_path / "gps.jpg"
    PILImage.new("RGB", (100, 100), "green").save(img_path)

    db_image = Image(id=1001, file_path=str(img_path), file_size=500)
    db_image.metadata_rel = ImageMetadata(
        width=100,
        height=100,
        gps_lat=48.137154,
        gps_lon=11.576124,
        taken_at_source=TakenAtSource.EXIF,
    )
    db_image.faces = []

    with patch.object(QtQuickWidgets.QQuickWidget, "setSource"):
        dialog = ImageDetailDialog(
            db_image, mock_session_factory, mock_vector_store, tmp_app_paths
        )
    qtbot.add_widget(dialog)

    map_widgets = dialog.findChildren(MapWidget)
    assert len(map_widgets) == 1


def test_map_not_shown_when_no_gps(
    qtbot, tmp_path, mock_session_factory, mock_vector_store, tmp_app_paths
):
    """No MapWidget is added when metadata exists but has no GPS coordinates."""
    from PIL import Image as PILImage

    img_path = tmp_path / "no_gps.jpg"
    PILImage.new("RGB", (100, 100), "red").save(img_path)

    db_image = Image(id=1002, file_path=str(img_path), file_size=500)
    db_image.metadata_rel = ImageMetadata(
        width=100,
        height=100,
        taken_at_source=TakenAtSource.EXIF,
    )
    db_image.faces = []

    dialog = ImageDetailDialog(
        db_image, mock_session_factory, mock_vector_store, tmp_app_paths
    )
    qtbot.add_widget(dialog)

    map_widgets = dialog.findChildren(MapWidget)
    assert len(map_widgets) == 0


def test_map_not_shown_when_no_metadata(
    qtbot, tmp_path, mock_session_factory, mock_vector_store, tmp_app_paths
):
    """No MapWidget is added when metadata_rel is None."""
    from PIL import Image as PILImage

    img_path = tmp_path / "no_meta.jpg"
    PILImage.new("RGB", (100, 100), "blue").save(img_path)

    db_image = Image(id=1003, file_path=str(img_path), file_size=500)
    db_image.faces = []

    dialog = ImageDetailDialog(
        db_image, mock_session_factory, mock_vector_store, tmp_app_paths
    )
    qtbot.add_widget(dialog)

    map_widgets = dialog.findChildren(MapWidget)
    assert len(map_widgets) == 0
