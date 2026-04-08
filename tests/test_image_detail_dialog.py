from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage
from PySide6 import QtCore, QtGui, QtQuickWidgets, QtWidgets

from photoaident.core.search import SearchResult
from photoaident.db.database import (
    Face,
    FaceState,
    Image,
    ImageMetadata,
    Person,
    TakenAtSource,
    get_session_factory,
)
from photoaident.ui.widgets.image_detail_dialog import (
    ImageDetailDialog,
    _FaceOverlayLabel,
)
from photoaident.ui.widgets.map_widget import MapWidget

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_result(db_image: Image, thumb_path: Path) -> SearchResult:
    """Build a SearchResult from a DB Image object."""
    return SearchResult(
        image_id=db_image.id,
        file_path=db_image.file_path,
        thumb_path=thumb_path,
    )


def _persist_image(session_factory, db_image: Image) -> None:
    """Insert db_image (and its related rows) into the real test database.

    expire_on_commit=False keeps attributes accessible after the session closes,
    so callers can still read fields like .id and .file_path on the same object.
    """
    session = session_factory()
    session.expire_on_commit = False
    with session:
        session.add(db_image)
        session.commit()


def _create_dialog(
    db_image: Image,
    session_factory,
    vector_store,
    paths,
    results: list[SearchResult] | None = None,
    current_index: int = 0,
) -> ImageDetailDialog:
    """Persist db_image to the test DB and create an ImageDetailDialog for it."""
    if results is None:
        results = [_make_result(db_image, paths.thumbs_dir / "fake_thumb.jpg")]
    _persist_image(session_factory, db_image)
    return ImageDetailDialog(
        results=results,
        current_index=current_index,
        session_factory=session_factory,
        vector_store=vector_store,
        paths=paths,
    )


def _make_face(
    state: FaceState,
    bbox_x: int = 0,
    bbox_y: int = 0,
    bbox_w: int = 50,
    bbox_h: int = 50,
    confidence: float = 0.9,
    **kwargs,
) -> Face:
    """Build a Face with common defaults."""
    return Face(
        bbox_x=bbox_x,
        bbox_y=bbox_y,
        bbox_w=bbox_w,
        bbox_h=bbox_h,
        detection_confidence=confidence,
        model_version="v1",
        state=state,
        **kwargs,
    )


def _make_db_image(
    tmp_path: Path,
    image_id: int,
    filename: str,
    color: str = "white",
    size: tuple[int, int] = (100, 100),
    faces: list[Face] | None = None,
) -> Image:
    """Create a JPEG on disk and a matching Image DB object."""
    img_path = tmp_path / filename
    PILImage.new("RGB", size, color).save(img_path)
    db_image = Image(id=image_id, file_path=str(img_path), file_size=500)
    db_image.faces = faces if faces is not None else []
    return db_image


def _get_label_texts(dialog) -> list[str]:
    """Return all QLabel text values from a dialog."""
    return [label.text() for label in dialog.findChildren(QtWidgets.QLabel)]


def _find_button(dialog, text: str) -> QtWidgets.QPushButton | None:
    """Find the first QPushButton whose text contains ``text``."""
    return next(
        (b for b in dialog.findChildren(QtWidgets.QPushButton) if text in b.text()),
        None,
    )


def _make_mouse_move_event(x: float, y: float) -> QtGui.QMouseEvent:
    """Create a MouseMove event at the given position."""
    return QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        QtCore.QPointF(x, y),
        QtCore.QPointF(x, y),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )


def _setup_two_image_dialog(
    tmp_path: Path, session_factory, vector_store, tmp_app_paths, current_index: int
) -> tuple[Image, Image, ImageDetailDialog]:
    """Persist two images and return a dialog starting at current_index."""
    db_image1 = _make_db_image(tmp_path, 1, "img1.jpg", "red")
    db_image2 = _make_db_image(tmp_path, 2, "img2.jpg", "blue")
    results = [
        SearchResult(image_id=1, file_path=db_image1.file_path, thumb_path=Path("/t1")),
        SearchResult(image_id=2, file_path=db_image2.file_path, thumb_path=Path("/t2")),
    ]
    _persist_image(session_factory, db_image1)
    _persist_image(session_factory, db_image2)
    dialog = ImageDetailDialog(
        results=results,
        current_index=current_index,
        session_factory=session_factory,
        vector_store=vector_store,
        paths=tmp_app_paths,
    )
    return db_image1, db_image2, dialog


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_image_detail_dialog_init(
    qtbot,
    sample_image_with_metadata,
    session_factory,
    vector_store,
    tmp_app_paths,
):
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    texts = _get_label_texts(dialog)
    assert any("123" in t for t in texts)  # ID
    assert any("TestCamera" in t for t in texts)  # Camera make
    assert any("1000 x 800" in t for t in texts)  # Dimensions
    assert any("1.0 KB" in t for t in texts)  # File size formatted

    assert not dialog.image_label.pixmap().isNull()


def test_image_detail_dialog_missing_file(
    qtbot, session_factory, vector_store, tmp_app_paths
):
    db_image = Image(id=456, file_path="/non/existent/path.jpg", file_size=0)
    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    assert "not found" in dialog.image_label.text().lower()


def test_image_detail_dialog_close(
    qtbot,
    sample_image_with_metadata,
    session_factory,
    vector_store,
    tmp_app_paths,
):
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    close_button = _find_button(dialog, "Close")
    assert close_button is not None

    with qtbot.wait_signal(dialog.finished):
        qtbot.mouseClick(close_button, QtCore.Qt.MouseButton.LeftButton)


def test_large_file_size_shows_mb(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """File size >= 1 MB is displayed in MB units."""
    db_image = _make_db_image(tmp_path, 200, "big.jpg", "green")
    db_image.file_size = 2 * 1024 * 1024  # 2 MB

    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    assert any("MB" in t for t in _get_label_texts(dialog))


def test_taken_at_in_metadata_is_displayed(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """taken_at field in ImageMetadata appears in the dialog."""
    db_image = _make_db_image(tmp_path, 300, "dated.jpg", "yellow")
    db_image.metadata_rel = ImageMetadata(
        width=100,
        height=100,
        taken_at=datetime(2021, 3, 14, 15, 9, 26),
        taken_at_source=TakenAtSource.EXIF,
    )

    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    assert any("2021-03-14" in t for t in _get_label_texts(dialog))


def test_load_image_failure_shows_error(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """A file with invalid image content shows a failure message."""
    bad_file = tmp_path / "bad.jpg"
    bad_file.write_bytes(b"this is not a valid jpeg")

    db_image = Image(id=400, file_path=str(bad_file), file_size=25)
    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    assert "failed" in dialog.image_label.text().lower()


def test_update_image_display_without_pixmap_is_noop(
    qtbot, session_factory, vector_store, tmp_app_paths
):
    """_update_image_display returns early if _original_pixmap is None."""
    db_image = Image(id=500, file_path="/nonexistent.jpg", file_size=0)
    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    assert dialog._original_pixmap is None
    dialog._update_image_display()  # must not raise


def test_resize_event_schedules_redisplay(
    qtbot,
    sample_image_with_metadata,
    session_factory,
    vector_store,
    tmp_app_paths,
):
    """resizeEvent does not raise and schedules a display update."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    event = QtGui.QResizeEvent(QtCore.QSize(900, 700), QtCore.QSize(800, 600))
    dialog.resizeEvent(event)  # must not raise


def test_label_faces_button_enabled_when_unidentified_faces(
    qtbot,
    sample_image_with_metadata,
    session_factory,
    vector_store,
    tmp_app_paths,
):
    """Label button is enabled when the image has at least one unidentified face."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    label_btn = _find_button(dialog, "Label")
    assert label_btn is not None
    assert label_btn.isEnabled()


def test_label_faces_button_disabled_when_no_unidentified_faces(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Label button is disabled when the image has no unidentified faces."""
    db_image = _make_db_image(
        tmp_path,
        600,
        "identified.jpg",
        "blue",
        faces=[_make_face(FaceState.IDENTIFIED)],
    )

    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    label_btn = _find_button(dialog, "Label")
    assert label_btn is not None
    assert not label_btn.isEnabled()


def test_label_faces_button_emits_signal(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Clicking the label button emits navigate_to_labelling with the image id."""
    db_image = _make_db_image(
        tmp_path,
        700,
        "signal_test.jpg",
        "green",
        faces=[_make_face(FaceState.UNIDENTIFIED)],
    )

    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    emitted_ids: list[int] = []
    dialog.navigate_to_labelling.connect(emitted_ids.append)

    label_btn = _find_button(dialog, "Label")
    assert label_btn is not None

    qtbot.mouseClick(label_btn, QtCore.Qt.MouseButton.LeftButton)

    assert emitted_ids == [700]


# ===========================================================================
# show_in_file_manager_button
# ===========================================================================


def _find_show_in_file_manager_button(
    dialog: ImageDetailDialog,
) -> QtWidgets.QPushButton | None:
    """Find the 'Show in File Manager' button (exact translated text match)."""
    return next(
        (
            b
            for b in dialog.findChildren(QtWidgets.QPushButton)
            if b.text() == dialog.tr("Show in File Manager")
        ),
        None,
    )


def test_show_in_file_manager_button_exists(
    qtbot,
    sample_image_with_metadata,
    session_factory,
    vector_store,
    tmp_app_paths,
):
    """Show in File Manager button is present in the dialog."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    assert _find_show_in_file_manager_button(dialog) is not None


def test_show_in_file_manager_button_calls_reveal(
    qtbot,
    sample_image_with_metadata,
    session_factory,
    vector_store,
    tmp_app_paths,
):
    """Clicking the button calls reveal_in_file_manager with the image file path."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    show_btn = _find_show_in_file_manager_button(dialog)
    assert show_btn is not None

    target = "photoaident.ui.widgets.image_detail_dialog.reveal_in_file_manager"
    with patch(target) as mock_reveal:
        qtbot.mouseClick(show_btn, QtCore.Qt.MouseButton.LeftButton)

    mock_reveal.assert_called_once_with(sample_image_with_metadata.file_path)


def test_show_in_file_manager_button_always_enabled(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Show in File Manager button is enabled regardless of face labelling state."""
    db_image = _make_db_image(
        tmp_path,
        800,
        "all_identified.jpg",
        "cyan",
        faces=[_make_face(FaceState.IDENTIFIED, confidence=0.95)],
    )

    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    show_btn = _find_show_in_file_manager_button(dialog)
    assert show_btn is not None
    assert show_btn.isEnabled()


# ===========================================================================
# navigate_to_browse
# ===========================================================================


def test_browse_photo_folder_button_emits_navigate_to_browse(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Clicking 'Browse Photo Folder' emits navigate_to_browse with the file path."""
    db_image = _make_db_image(tmp_path, 1100, "browse_signal.jpg", "red")

    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

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

    with qtbot.waitSignal(dialog.finished):
        qtbot.mouseClick(browse_btn, QtCore.Qt.MouseButton.LeftButton)

    assert received == [db_image.file_path]


def test_file_path_link_emits_navigate_to_browse(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Activating the file path link in the metadata panel emits navigate_to_browse."""
    db_image = _make_db_image(tmp_path, 1101, "link_signal.jpg", "blue")

    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    received: list[str] = []
    dialog.navigate_to_browse.connect(received.append)

    file_path_label = next(
        (
            lbl
            for lbl in dialog.findChildren(QtWidgets.QLabel)
            if 'href="#"' in lbl.text()
        ),
        None,
    )
    assert file_path_label is not None

    with qtbot.waitSignal(dialog.finished):
        file_path_label.linkActivated.emit("#")

    assert received == [db_image.file_path]
    assert dialog.isHidden()


# ===========================================================================
# Tooltip tests — _build_face_display_info
# ===========================================================================


def test_face_tooltip_identified_shows_person_name(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """_build_face_display_info returns the person's name for IDENTIFIED faces."""
    face = _make_face(FaceState.IDENTIFIED, bbox_x=10, bbox_y=10, bbox_w=50, bbox_h=50)
    face.person = Person(id=1, name="Alice")

    db_image = _make_db_image(
        tmp_path, 1, "identified.jpg", size=(200, 200), faces=[face]
    )
    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    regions = dialog._build_face_display_info()
    assert len(regions) == 1
    assert regions[0][2] == "Alice"
    assert regions[0][1] == QtCore.Qt.GlobalColor.green


def test_face_tooltip_anonymous_shows_anonymous(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """_build_face_display_info returns 'Anonymous' for ANONYMOUS faces."""
    face = _make_face(FaceState.ANONYMOUS, bbox_x=10, bbox_y=10, bbox_w=50, bbox_h=50)

    db_image = _make_db_image(tmp_path, 2, "anon.jpg", size=(200, 200), faces=[face])
    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    regions = dialog._build_face_display_info()
    assert len(regions) == 1
    assert regions[0][2] == dialog.tr("Anonymous")
    assert regions[0][1] == QtCore.Qt.GlobalColor.green


def test_face_tooltip_unidentified_no_vector_match(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """_build_face_display_info: UNIDENTIFIED with no vector match -> Unknown + red."""
    face = _make_face(
        FaceState.UNIDENTIFIED, bbox_x=10, bbox_y=10, bbox_w=50, bbox_h=50
    )

    db_image = _make_db_image(tmp_path, 3, "unknown.jpg", size=(200, 200), faces=[face])
    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    regions = dialog._build_face_display_info()
    assert len(regions) == 1
    assert regions[0][2] == dialog.tr("Unknown")
    assert regions[0][1] == QtCore.Qt.GlobalColor.red


def test_face_tooltip_deleted_face_excluded(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """_build_face_display_info skips faces where deleted_at is set."""
    active_face = _make_face(
        FaceState.UNIDENTIFIED, bbox_x=10, bbox_y=10, bbox_w=50, bbox_h=50
    )
    deleted_face = _make_face(
        FaceState.UNIDENTIFIED,
        bbox_x=100,
        bbox_y=100,
        confidence=0.8,
        deleted_at=datetime(2024, 1, 1),
    )

    db_image = _make_db_image(
        tmp_path,
        4,
        "deleted.jpg",
        size=(200, 200),
        faces=[active_face, deleted_face],
    )
    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    regions = dialog._build_face_display_info()
    assert len(regions) == 1
    assert regions[0][0] == QtCore.QRectF(10, 10, 50, 50)


# ===========================================================================
# _FaceOverlayLabel hit/miss detection
# ===========================================================================


def _make_overlay_label_with_pixmap(qtbot) -> _FaceOverlayLabel:
    """Create a _FaceOverlayLabel with a 200x200 white pixmap."""
    label = _FaceOverlayLabel()
    qtbot.add_widget(label)
    pixmap = QtGui.QPixmap(200, 200)
    pixmap.fill(QtGui.QColor("white"))
    label.setPixmap(pixmap)
    label.resize(200, 200)
    return label


def test_face_overlay_label_hit_detection(qtbot):
    """Mouse inside a face bbox triggers QToolTip.showText."""
    label = _make_overlay_label_with_pixmap(qtbot)

    region = (QtCore.QRectF(50, 50, 60, 60), "Alice")
    label.set_face_regions([region], QtCore.QSize(200, 200))

    with patch(
        "photoaident.ui.widgets.image_detail_dialog.QtWidgets.QToolTip.showText"
    ) as mock_show:
        label.mouseMoveEvent(_make_mouse_move_event(80.0, 80.0))
        mock_show.assert_called_once()
        assert mock_show.call_args[0][1] == "Alice"


def test_face_overlay_label_no_regions_early_return(qtbot):
    """mouseMoveEvent returns early when no face regions are set."""
    label = _make_overlay_label_with_pixmap(qtbot)
    # Do NOT call set_face_regions — _face_regions stays empty

    with (
        patch(
            "photoaident.ui.widgets.image_detail_dialog.QtWidgets.QToolTip.showText"
        ) as mock_show,
        patch(
            "photoaident.ui.widgets.image_detail_dialog.QtWidgets.QToolTip.hideText"
        ) as mock_hide,
    ):
        label.mouseMoveEvent(_make_mouse_move_event(80.0, 80.0))
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
        label.mouseMoveEvent(_make_mouse_move_event(80.0, 80.0))
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
        label.mouseMoveEvent(_make_mouse_move_event(80.0, 80.0))
        mock_show.assert_not_called()
        mock_hide.assert_not_called()


def test_face_overlay_label_miss_detection(qtbot):
    """Mouse outside all face bboxes calls QToolTip.hideText."""
    label = _make_overlay_label_with_pixmap(qtbot)

    region = (QtCore.QRectF(50, 50, 60, 60), "Alice")
    label.set_face_regions([region], QtCore.QSize(200, 200))

    with patch(
        "photoaident.ui.widgets.image_detail_dialog.QtWidgets.QToolTip.hideText"
    ) as mock_hide:
        # 10, 10 is outside the 50-110 bbox range
        label.mouseMoveEvent(_make_mouse_move_event(10.0, 10.0))
        mock_hide.assert_called_once()


# ===========================================================================
# Bounding box colors by state
# ===========================================================================


def test_face_tooltip_unidentified_with_vector_store_match(
    qtbot, tmp_path, tmp_app_paths, session_factory, vector_store
):
    """UNIDENTIFIED face with a FAISS match shows name + percentage and green color."""
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
    db_image = _make_db_image(
        tmp_path, 10, "matched.jpg", size=(200, 200), faces=[face]
    )

    with patch(
        "photoaident.ui.widgets.image_detail_dialog.resolve_faces_to_persons",
        return_value={5: ("Dave", 0.82)},
    ):
        dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
        qtbot.add_widget(dialog)
        regions = dialog._build_face_display_info()

    assert len(regions) == 1
    assert regions[0][2] == "Dave (82%)"
    assert regions[0][1] == QtCore.Qt.GlobalColor.green


def test_face_tooltip_unidentified_with_vector_store_no_match(
    qtbot, tmp_path, tmp_app_paths, session_factory, vector_store
):
    """UNIDENTIFIED face with no FAISS match shows Unknown and red color."""
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
    db_image = _make_db_image(
        tmp_path, 11, "unmatched.jpg", size=(200, 200), faces=[face]
    )

    with patch(
        "photoaident.ui.widgets.image_detail_dialog.resolve_faces_to_persons",
        return_value={6: None},
    ):
        dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
        qtbot.add_widget(dialog)
        regions = dialog._build_face_display_info()

    assert len(regions) == 1
    assert regions[0][2] == dialog.tr("Unknown")
    assert regions[0][1] == QtCore.Qt.GlobalColor.red


def test_bounding_box_colors_by_state(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Verify that face boxes are drawn in different colors for different states."""
    person = Person(id=1, name="Carol")
    identified_face = _make_face(
        FaceState.IDENTIFIED, bbox_x=10, bbox_y=10, bbox_w=40, bbox_h=40
    )
    identified_face.person = person
    anonymous_face = _make_face(
        FaceState.ANONYMOUS, bbox_x=70, bbox_y=10, bbox_w=40, bbox_h=40
    )
    unidentified_face = _make_face(
        FaceState.UNIDENTIFIED, bbox_x=130, bbox_y=10, bbox_w=40, bbox_h=40
    )

    db_image = _make_db_image(
        tmp_path,
        5,
        "color_test.jpg",
        size=(200, 200),
        faces=[identified_face, anonymous_face, unidentified_face],
    )

    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    assert dialog._original_pixmap is not None
    assert not dialog._original_pixmap.isNull()

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
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Image with EXIF rotation triggers the transformation block."""
    img_path = tmp_path / "rotated.jpg"
    img = PILImage.new("RGB", (100, 200), "blue")
    exif = img.getexif()
    exif[0x0112] = 6  # Orientation tag: Rotate 90 CW
    img.save(img_path, exif=exif)

    db_image = Image(id=900, file_path=str(img_path), file_size=1000)
    db_image.faces = [
        _make_face(FaceState.IDENTIFIED, bbox_x=10, bbox_y=10, bbox_w=20, bbox_h=20)
    ]

    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    assert dialog._original_pixmap is not None
    pm = dialog._original_pixmap
    # After 90 degree rotation, width should be 200 and height 100
    assert pm.width() == 200
    assert pm.height() == 100


# ===========================================================================
# GPS map widget presence tests
# ===========================================================================


def test_map_shown_when_gps_available(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """A MapWidget is added to the metadata panel when GPS coordinates are present."""
    db_image = _make_db_image(tmp_path, 1001, "gps.jpg", "green")
    db_image.metadata_rel = ImageMetadata(
        width=100,
        height=100,
        gps_lat=48.137154,
        gps_lon=11.576124,
        taken_at_source=TakenAtSource.EXIF,
    )

    with patch.object(QtQuickWidgets.QQuickWidget, "setSource"):
        dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    map_widgets = dialog.findChildren(MapWidget)
    assert len(map_widgets) == 1


def test_map_not_shown_when_no_gps(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """MapWidget is hidden when metadata exists but has no GPS coordinates."""
    db_image = _make_db_image(tmp_path, 1002, "no_gps.jpg", "red")
    db_image.metadata_rel = ImageMetadata(
        width=100,
        height=100,
        taken_at_source=TakenAtSource.EXIF,
    )

    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    map_widgets = dialog.findChildren(MapWidget)
    assert len(map_widgets) == 1
    assert map_widgets[0].isHidden()


def test_map_not_shown_when_no_metadata(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """MapWidget is hidden when metadata_rel is None."""
    db_image = _make_db_image(tmp_path, 1003, "no_meta.jpg", "blue")

    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    map_widgets = dialog.findChildren(MapWidget)
    assert len(map_widgets) == 1
    assert map_widgets[0].isHidden()


# ===========================================================================
# Navigation tests
# ===========================================================================


def test_navigation_buttons_disabled_for_single_image(
    qtbot,
    sample_image_with_metadata,
    session_factory,
    vector_store,
    tmp_app_paths,
):
    """With a single image, both Previous and Next are disabled."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    assert not dialog._prev_btn.isEnabled()
    assert not dialog._next_btn.isEnabled()


def test_navigation_buttons_state_at_start(
    qtbot,
    tmp_path,
    session_factory,
    vector_store,
    tmp_app_paths,
):
    """At index 0 of multiple results, Previous is disabled and Next is enabled."""
    db_image = _make_db_image(tmp_path, 1, "nav.jpg", "red")
    results = [
        SearchResult(image_id=1, file_path=db_image.file_path, thumb_path=Path("/t1")),
        SearchResult(image_id=2, file_path=db_image.file_path, thumb_path=Path("/t2")),
        SearchResult(image_id=3, file_path=db_image.file_path, thumb_path=Path("/t3")),
    ]

    dialog = _create_dialog(
        db_image,
        session_factory,
        vector_store,
        tmp_app_paths,
        results=results,
        current_index=0,
    )
    qtbot.add_widget(dialog)

    assert not dialog._prev_btn.isEnabled()
    assert dialog._next_btn.isEnabled()


def test_navigate_next_loads_next_image(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Clicking Next advances to the next image."""
    _, db_image2, dialog = _setup_two_image_dialog(
        tmp_path, session_factory, vector_store, tmp_app_paths, current_index=0
    )
    qtbot.add_widget(dialog)

    assert dialog._current_index == 0
    dialog._navigate_next()

    assert dialog._current_index == 1
    assert dialog._image_data is not None
    assert dialog._image_data.id == db_image2.id


def test_navigate_previous_loads_previous_image(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Clicking Previous goes back to the previous image."""
    db_image1, _, dialog = _setup_two_image_dialog(
        tmp_path, session_factory, vector_store, tmp_app_paths, current_index=1
    )
    qtbot.add_widget(dialog)

    assert dialog._current_index == 1
    dialog._navigate_previous()

    assert dialog._current_index == 0
    assert dialog._image_data is not None
    assert dialog._image_data.id == db_image1.id


def test_arrow_keys_navigate(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Left/Right arrow keys trigger navigation."""
    db_image = _make_db_image(tmp_path, 1, "arrow.jpg", "green")
    results = [
        SearchResult(image_id=1, file_path=db_image.file_path, thumb_path=Path("/t1")),
        SearchResult(image_id=2, file_path=db_image.file_path, thumb_path=Path("/t2")),
        SearchResult(image_id=3, file_path=db_image.file_path, thumb_path=Path("/t3")),
    ]

    dialog = _create_dialog(
        db_image,
        session_factory,
        vector_store,
        tmp_app_paths,
        results=results,
        current_index=1,
    )
    qtbot.add_widget(dialog)

    assert dialog._current_index == 1

    dialog._shortcut_next.activated.emit()
    assert dialog._current_index == 2

    dialog._shortcut_prev.activated.emit()
    assert dialog._current_index == 1


# ===========================================================================
# Zoom functionality tests
# ===========================================================================


def test_zoom_buttons_present(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """Zoom buttons are present in the dialog."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    assert dialog._zoom_in_btn is not None
    assert dialog._zoom_out_btn is not None
    assert dialog._zoom_100_btn is not None
    assert dialog._zoom_fit_btn is not None


def test_zoom_in_increases_factor(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """Zoom In button increases the zoom factor."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    assert dialog._zoom_factor is None
    dialog._zoom_in()
    assert dialog._zoom_factor is not None
    assert dialog._zoom_factor > 0


def test_zoom_out_decreases_factor(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """Zoom Out button decreases the zoom factor."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    dialog._zoom_factor = 2.0
    dialog._zoom_out()
    assert dialog._zoom_factor < 2.0


def test_zoom_to_100_sets_factor_to_1(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """Zoom 100% button sets zoom factor to 1.0 (original size)."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    dialog._zoom_factor = 3.0
    dialog._zoom_to_100()
    assert dialog._zoom_factor == pytest.approx(1.0)


def test_zoom_to_fit_sets_none_factor(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """Reset Zoom button sets zoom factor to None (fit to viewport)."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    dialog._zoom_factor = 3.0
    dialog._zoom_to_fit()
    assert dialog._zoom_factor is None


def test_zoom_min_limit(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """Zoom does not go below fit-to-viewport when zoomed in."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    dialog._zoom_factor = 2.0
    dialog._zoom_out()
    assert dialog._zoom_factor >= dialog._min_zoom


def test_zoom_max_limit(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """Zoom does not exceed maximum (10.0) when zoomed in."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    dialog._zoom_factor = 9.0
    dialog._zoom_in()
    assert dialog._zoom_factor <= dialog._max_zoom


def test_wheel_zoom_triggers_zoom(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """Mouse wheel event triggers zoom in/out."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    viewport = dialog.scroll_area.viewport()
    initial_factor = dialog._zoom_factor

    wheel_event_up = QtGui.QWheelEvent(
        QtCore.QPointF(100, 100),
        QtCore.QPointF(100, 100),
        QtCore.QPoint(0, 120),
        QtCore.QPoint(0, 120),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    dialog.eventFilter(viewport, wheel_event_up)

    assert dialog._zoom_factor != initial_factor


def test_wheel_zoom_with_ctrl_modifier_does_not_zoom(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """Mouse wheel with Ctrl modifier does not trigger zoom (allows scroll)."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    initial_factor = dialog._zoom_factor

    wheel_event = QtGui.QWheelEvent(
        QtCore.QPointF(100, 100),
        QtCore.QPointF(100, 100),
        QtCore.QPoint(0, 120),
        QtCore.QPoint(0, 120),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.ControlModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    viewport = dialog.scroll_area.viewport()
    result = dialog.eventFilter(viewport, wheel_event)

    assert result is False
    assert dialog._zoom_factor == initial_factor


def test_zoom_center_tracks_mouse_position(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """Zooming updates _last_zoom_center to the mouse position."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    center = QtCore.QPointF(150, 200)
    dialog._zoom_in(center)

    assert dialog._last_zoom_center == center


def test_zoom_to_100_clears_zoom_center(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """Zoom to 100% sets zoom center to the original image center."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    dialog._last_zoom_center = QtCore.QPointF(100, 100)
    dialog._zoom_to_100()

    assert dialog._last_zoom_center is not None
    expected = dialog._get_image_center()
    assert dialog._last_zoom_center.x() == expected.x()
    assert dialog._last_zoom_center.y() == expected.y()


def test_zoom_to_fit_clears_zoom_center(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """Reset Zoom clears the zoom center."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    dialog._last_zoom_center = QtCore.QPointF(100, 100)
    dialog._zoom_to_fit()

    assert dialog._last_zoom_center is None


def test_zoom_buttons_have_correct_labels(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """Zoom buttons have expected translated text."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    assert dialog._zoom_in_btn.text() == dialog.tr("Zoom In")
    assert dialog._zoom_out_btn.text() == dialog.tr("Zoom Out")
    assert dialog._zoom_100_btn.text() == dialog.tr("Zoom 100%")
    assert dialog._zoom_fit_btn.text() == dialog.tr("Reset Zoom")


def test_image_detail_dialog_resets_zoom_on_new_image(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Loading a new image resets zoom factor to None."""
    db_image1 = _make_db_image(tmp_path, 1, "img1.jpg", "red")
    db_image2 = _make_db_image(tmp_path, 2, "img2.jpg", "blue")
    results = [
        SearchResult(image_id=1, file_path=db_image1.file_path, thumb_path=Path("/t1")),
        SearchResult(image_id=2, file_path=db_image2.file_path, thumb_path=Path("/t2")),
    ]
    _persist_image(session_factory, db_image1)
    _persist_image(session_factory, db_image2)

    dialog = ImageDetailDialog(
        results=results,
        current_index=0,
        session_factory=session_factory,
        vector_store=vector_store,
        paths=tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    assert dialog._zoom_factor is None
    dialog._zoom_factor = 2.5

    dialog._current_index = 1
    dialog._show_current_image()

    assert dialog._zoom_factor is None


def test_fit_factor_calculated_from_viewport(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """_fit_factor is calculated based on viewport size vs original image size."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    assert dialog._fit_factor > 0
    assert dialog._fit_factor <= 1.0


def test_zoom_100_means_original_size(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Zoom factor 1.0 means display at original image size (not fit to viewport)."""
    db_image = _make_db_image(tmp_path, 1, "test.jpg", "red", size=(1000, 800))
    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    dialog._zoom_factor = 1.0
    dialog._update_image_display()

    label_pixmap = dialog.image_label.pixmap()
    assert label_pixmap.width() == 1000
    assert label_pixmap.height() == 800


def test_zoom_to_fit_uses_fit_factor(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Zoom to fit displays image scaled to fit viewport."""
    db_image = _make_db_image(tmp_path, 1, "test.jpg", "red", size=(1000, 800))
    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    viewport_size = dialog.scroll_area.viewport().size()

    dialog._zoom_to_fit()

    label_pixmap = dialog.image_label.pixmap()
    assert label_pixmap.width() <= viewport_size.width()
    assert label_pixmap.height() <= viewport_size.height()


def test_zoom_in_above_fit_shows_larger_image(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Zoom in with factor > fit_factor shows image larger than original."""
    db_image = _make_db_image(tmp_path, 1, "test.jpg", "red", size=(1000, 800))
    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    dialog._zoom_factor = 2.0
    dialog._update_image_display()

    label_pixmap = dialog.image_label.pixmap()
    assert label_pixmap.width() > 1000
    assert label_pixmap.height() > 800


def test_zoom_in_zooms_to_visible_center(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """Zoom In button zooms to center of the currently visible portion of the image."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    viewport_size = dialog.scroll_area.viewport().size()
    initial_scroll_x = viewport_size.width() // 4
    initial_scroll_y = viewport_size.height() // 4
    dialog.scroll_area.horizontalScrollBar().setValue(initial_scroll_x)
    dialog.scroll_area.verticalScrollBar().setValue(initial_scroll_y)

    dialog._zoom_factor = 2.0
    dialog._update_image_display()

    center = dialog._get_visible_center()
    dialog._zoom_in()

    assert dialog._last_zoom_center is not None
    assert abs(dialog._last_zoom_center.x() - center.x()) < 1
    assert abs(dialog._last_zoom_center.y() - center.y()) < 1


def test_zoom_out_zooms_to_visible_center(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """Zoom Out button zooms to center of the currently visible portion of the image."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    viewport_size = dialog.scroll_area.viewport().size()
    initial_scroll_x = viewport_size.width() // 4
    initial_scroll_y = viewport_size.height() // 4
    dialog.scroll_area.horizontalScrollBar().setValue(initial_scroll_x)
    dialog.scroll_area.verticalScrollBar().setValue(initial_scroll_y)

    dialog._zoom_factor = 2.0
    dialog._update_image_display()

    center = dialog._get_visible_center()
    dialog._zoom_out()

    assert dialog._last_zoom_center is not None
    assert abs(dialog._last_zoom_center.x() - center.x()) < 1
    assert abs(dialog._last_zoom_center.y() - center.y()) < 1


def test_zoom_in_smaller_image_zooms_to_center_of_image(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Zoom In on an image smaller than viewport zooms to center of the viewport."""
    db_image = _make_db_image(tmp_path, 1, "small.jpg", "red", size=(100, 100))
    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    dialog._zoom_factor = 1.0
    dialog._update_image_display()

    center = dialog._get_visible_center()
    dialog._zoom_in()

    assert dialog._last_zoom_center is not None
    assert abs(dialog._last_zoom_center.x() - center.x()) < 1
    assert abs(dialog._last_zoom_center.y() - center.y()) < 1


def test_zoom_out_smaller_image_zooms_to_center_of_image(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Zoom Out on an image smaller than viewport zooms to center of the viewport."""
    db_image = _make_db_image(tmp_path, 1, "small.jpg", "red", size=(100, 100))
    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    dialog._zoom_factor = 1.0
    dialog._update_image_display()

    center = dialog._get_visible_center()
    dialog._zoom_out()

    assert dialog._last_zoom_center is not None
    assert abs(dialog._last_zoom_center.x() - center.x()) < 1
    assert abs(dialog._last_zoom_center.y() - center.y()) < 1


def test_zoom_100_zooms_to_image_center(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """Zoom 100% button zooms to center of the original image."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    viewport_size = dialog.scroll_area.viewport().size()
    dialog.scroll_area.horizontalScrollBar().setValue(viewport_size.width() // 4)
    dialog.scroll_area.verticalScrollBar().setValue(viewport_size.height() // 4)

    dialog._zoom_factor = 2.0
    dialog._update_image_display()

    image_center = dialog._get_image_center()
    dialog._zoom_to_100()

    assert dialog._last_zoom_center is not None
    assert abs(dialog._last_zoom_center.x() - image_center.x()) < 1
    assert abs(dialog._last_zoom_center.y() - image_center.y()) < 1


def test_get_visible_center_when_image_smaller_than_viewport(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """_get_visible_center returns viewport center regardless of image size."""
    db_image = _make_db_image(tmp_path, 1, "tiny.jpg", "blue", size=(50, 50))
    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    dialog._zoom_factor = 1.0
    dialog._update_image_display()

    viewport_size = dialog.scroll_area.viewport().size()
    center = dialog._get_visible_center()

    assert abs(center.x() - viewport_size.width() / 2) < 1
    assert abs(center.y() - viewport_size.height() / 2) < 1


def test_get_visible_center_with_scrolled_image(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """_get_visible_center returns visible center when image is larger than viewport."""
    db_image = _make_db_image(tmp_path, 1, "large.jpg", "green", size=(1000, 1000))
    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    dialog._zoom_factor = 2.0
    dialog._update_image_display()

    display_pixmap = dialog.image_label.pixmap()
    viewport_size = dialog.scroll_area.viewport().size()

    if display_pixmap.width() > viewport_size.width():
        scroll_x = dialog.scroll_area.horizontalScrollBar().maximum() // 2
        scroll_y = dialog.scroll_area.verticalScrollBar().maximum() // 2
        dialog.scroll_area.horizontalScrollBar().setValue(scroll_x)
        dialog.scroll_area.verticalScrollBar().setValue(scroll_y)

        center = dialog._get_visible_center()

        assert center.x() > 0
        assert center.y() > 0
    else:
        center = dialog._get_visible_center()
        assert center.x() > 0
        assert center.y() > 0


def test_constructor_rejects_empty_results(tmp_app_paths, db_engine):
    """ImageDetailDialog raises ValueError when results list is empty."""
    session_factory = get_session_factory(db_engine)

    with pytest.raises(ValueError, match="must not be empty"):
        ImageDetailDialog(
            results=[],
            current_index=0,
            session_factory=session_factory,
            vector_store=MagicMock(),
            paths=tmp_app_paths,
        )


def test_constructor_rejects_invalid_current_index(tmp_app_paths, db_engine):
    """ImageDetailDialog raises ValueError when current_index is out of range."""
    session_factory = get_session_factory(db_engine)
    results = [SearchResult(image_id=1, file_path="/test.jpg", thumb_path=Path("/t"))]

    with pytest.raises(ValueError, match="out of range"):
        ImageDetailDialog(
            results=results,
            current_index=5,
            session_factory=session_factory,
            vector_store=MagicMock(),
            paths=tmp_app_paths,
        )


def test_clear_layout_removes_nested_widgets(
    qtbot, session_factory, vector_store, tmp_app_paths
):
    """_clear_layout recursively removes nested widgets."""
    dialog = _create_dialog(
        Image(id=999, file_path="/nonexistent.jpg", file_size=0),
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    nested_widget = QtWidgets.QWidget()
    nested_layout = QtWidgets.QHBoxLayout(nested_widget)
    inner_label = QtWidgets.QLabel("test")
    nested_layout.addWidget(inner_label)

    dialog._clear_layout(nested_layout)
    assert nested_layout.count() == 0


def test_clear_layout_recursive(qtbot, session_factory, vector_store, tmp_app_paths):
    """_clear_layout recursively removes nested layouts."""
    dialog = _create_dialog(
        Image(id=999, file_path="/nonexistent.jpg", file_size=0),
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    outer_layout = QtWidgets.QVBoxLayout()
    inner_layout = QtWidgets.QHBoxLayout()
    inner_label = QtWidgets.QLabel("test")
    inner_layout.addWidget(inner_label)
    outer_layout.addLayout(inner_layout)

    dialog._clear_layout(outer_layout)
    assert outer_layout.count() == 0


def test_add_meta_row_skips_none_value(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """_add_meta_row does not add a row when value is None."""
    db_image = _make_db_image(tmp_path, 1, "test.jpg", "red")
    dialog = _create_dialog(db_image, session_factory, vector_store, tmp_app_paths)
    qtbot.add_widget(dialog)

    layout = QtWidgets.QVBoxLayout()
    initial_count = layout.count()

    dialog._add_meta_row(layout, "TestLabel", None)
    assert layout.count() == initial_count


def test_get_visible_center_returns_viewport_center():
    """_get_visible_center always returns the geometric center of the viewport."""
    dialog = ImageDetailDialog.__new__(ImageDetailDialog)
    mock_viewport = MagicMock()
    mock_viewport.size.return_value = QtCore.QSize(800, 600)
    dialog.scroll_area = MagicMock()
    dialog.scroll_area.viewport.return_value = mock_viewport

    result = dialog._get_visible_center()

    assert result.x() == 400
    assert result.y() == 300


def test_get_image_center_returns_origin_when_no_pixmap():
    """_get_image_center returns (0, 0) when _original_pixmap is None."""
    dialog = ImageDetailDialog.__new__(ImageDetailDialog)
    dialog._original_pixmap = None

    result = dialog._get_image_center()
    assert result.x() == 0 and result.y() == 0


def test_zoom_out_does_nothing_when_already_at_fit(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """Zoom out does nothing when already at fit-to-viewport."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    dialog._zoom_factor = None
    dialog._min_zoom = 0.5
    dialog._fit_factor = 0.5

    dialog._zoom_out()

    assert dialog._zoom_factor is None


def test_event_filter_wheel_zoom_in(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """eventFilter handles wheel zoom in (delta > 0)."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    viewport = dialog.scroll_area.viewport()

    initial = dialog._zoom_factor
    wheel_event = QtGui.QWheelEvent(
        QtCore.QPointF(100, 100),
        QtCore.QPointF(100, 100),
        QtCore.QPoint(0, 120),
        QtCore.QPoint(0, 120),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    dialog.eventFilter(viewport, wheel_event)

    assert dialog._zoom_factor != initial


def test_event_filter_wheel_zoom_out(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """eventFilter handles wheel zoom out (delta < 0)."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    dialog._zoom_factor = 1.5

    viewport = dialog.scroll_area.viewport()
    wheel_event = QtGui.QWheelEvent(
        QtCore.QPointF(100, 100),
        QtCore.QPointF(100, 100),
        QtCore.QPoint(0, -120),
        QtCore.QPoint(0, -120),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    dialog.eventFilter(viewport, wheel_event)

    assert dialog._zoom_factor < 1.5


def test_build_face_display_info_returns_empty_when_no_image_data():
    """_build_face_display_info returns [] when _image_data is None."""
    dialog = ImageDetailDialog.__new__(ImageDetailDialog)
    dialog._image_data = None

    result = dialog._build_face_display_info()
    assert result == []


def test_load_image_returns_early_when_no_image_data():
    """_load_image returns early when _image_data is None."""
    dialog = ImageDetailDialog.__new__(ImageDetailDialog)
    dialog._image_data = None

    dialog._load_image()


# ===========================================================================
# _is_pannable
# ===========================================================================


def test_is_pannable_returns_false_when_no_scrollbar_overflow(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """_is_pannable returns False when both scrollbars have maximum of 0."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    dialog.scroll_area.horizontalScrollBar().setMaximum(0)
    dialog.scroll_area.verticalScrollBar().setMaximum(0)

    assert dialog._is_pannable() is False


def test_is_pannable_returns_true_when_horizontal_scrollbar_overflows(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """_is_pannable returns True when horizontal scrollbar maximum > 0."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    dialog.scroll_area.horizontalScrollBar().setMaximum(100)
    dialog.scroll_area.verticalScrollBar().setMaximum(0)

    assert dialog._is_pannable() is True


def test_is_pannable_returns_true_when_vertical_scrollbar_overflows(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """_is_pannable returns True when vertical scrollbar maximum > 0."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    dialog.scroll_area.horizontalScrollBar().setMaximum(0)
    dialog.scroll_area.verticalScrollBar().setMaximum(50)

    assert dialog._is_pannable() is True


# ===========================================================================
# _update_pan_cursor
# ===========================================================================


def test_update_pan_cursor_sets_open_hand_when_pannable(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """_update_pan_cursor sets OpenHandCursor when image overflows the viewport."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)
    dialog._panning = False

    with patch.object(dialog, "_is_pannable", return_value=True):
        dialog._update_pan_cursor()

    assert dialog.image_label.cursor().shape() == QtCore.Qt.CursorShape.OpenHandCursor


def test_update_pan_cursor_unsets_cursor_when_not_pannable(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """_update_pan_cursor unsets cursor when image fits in the viewport."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)
    dialog._panning = False
    # Pre-set a cursor so we can verify it gets removed.
    dialog.image_label.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)

    with patch.object(dialog, "_is_pannable", return_value=False):
        dialog._update_pan_cursor()

    # After unsetCursor the shape falls back to the inherited/default cursor.
    assert dialog.image_label.cursor().shape() != QtCore.Qt.CursorShape.OpenHandCursor


def test_update_pan_cursor_skips_update_during_active_panning(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """_update_pan_cursor does not change cursor while panning is active."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)
    dialog._panning = True
    dialog.image_label.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)

    with patch.object(dialog, "_is_pannable", return_value=True) as mock_pannable:
        dialog._update_pan_cursor()
        mock_pannable.assert_not_called()

    # Cursor must remain ClosedHandCursor (unchanged).
    assert dialog.image_label.cursor().shape() == QtCore.Qt.CursorShape.ClosedHandCursor


# ===========================================================================
# _stop_panning
# ===========================================================================


def test_stop_panning_clears_panning_flag(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """_stop_panning sets _panning to False when panning was active."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)
    dialog._panning = True

    dialog._stop_panning()

    assert dialog._panning is False


def test_stop_panning_unsets_cursor(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """_stop_panning removes the closed-hand cursor after stopping."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)
    dialog._panning = True
    dialog.image_label.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)

    dialog._stop_panning()

    assert dialog.image_label.cursor().shape() != QtCore.Qt.CursorShape.ClosedHandCursor


def test_stop_panning_does_nothing_when_not_panning(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """_stop_panning is a no-op when _panning is already False."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)
    dialog._panning = False
    dialog.image_label.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)

    dialog._stop_panning()

    # _panning stays False and the cursor is untouched.
    assert dialog._panning is False
    assert dialog.image_label.cursor().shape() == QtCore.Qt.CursorShape.OpenHandCursor


# ===========================================================================
# eventFilter — mouse panning
# ===========================================================================


def _make_mouse_press_event(
    x: float,
    y: float,
    button: QtCore.Qt.MouseButton = QtCore.Qt.MouseButton.LeftButton,
    global_x: float | None = None,
    global_y: float | None = None,
) -> QtGui.QMouseEvent:
    """Build a MouseButtonPress event at the given local position."""
    gx = global_x if global_x is not None else x
    gy = global_y if global_y is not None else y
    return QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseButtonPress,
        QtCore.QPointF(x, y),
        QtCore.QPointF(gx, gy),
        button,
        button,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )


def _make_mouse_release_event(
    x: float,
    y: float,
    button: QtCore.Qt.MouseButton = QtCore.Qt.MouseButton.LeftButton,
    global_x: float | None = None,
    global_y: float | None = None,
) -> QtGui.QMouseEvent:
    """Build a MouseButtonRelease event at the given local position."""
    gx = global_x if global_x is not None else x
    gy = global_y if global_y is not None else y
    return QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseButtonRelease,
        QtCore.QPointF(x, y),
        QtCore.QPointF(gx, gy),
        button,
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )


def _make_mouse_move_event_with_global(
    x: float,
    y: float,
    global_x: float | None = None,
    global_y: float | None = None,
) -> QtGui.QMouseEvent:
    """Build a MouseMove event with distinct global position."""
    gx = global_x if global_x is not None else x
    gy = global_y if global_y is not None else y
    return QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        QtCore.QPointF(x, y),
        QtCore.QPointF(gx, gy),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )


def test_event_filter_press_pannable_starts_panning(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """eventFilter left-press on pannable image starts panning and returns True."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)
    dialog.scroll_area.horizontalScrollBar().setMaximum(200)

    press = _make_mouse_press_event(50, 50, global_x=300.0, global_y=400.0)
    result = dialog.eventFilter(dialog.image_label, press)

    assert result is True
    assert dialog._panning is True
    assert dialog._pan_start == QtCore.QPoint(300, 400)
    assert dialog._pan_scroll_start == (
        dialog.scroll_area.horizontalScrollBar().value(),
        dialog.scroll_area.verticalScrollBar().value(),
    )
    assert dialog.image_label.cursor().shape() == QtCore.Qt.CursorShape.ClosedHandCursor


def test_event_filter_press_not_pannable_does_not_start_panning(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """eventFilter left-press on non-pannable image does not start panning."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)
    dialog.scroll_area.horizontalScrollBar().setMaximum(0)
    dialog.scroll_area.verticalScrollBar().setMaximum(0)

    press = _make_mouse_press_event(50, 50)
    result = dialog.eventFilter(dialog.image_label, press)

    assert result is False
    assert dialog._panning is False


def test_event_filter_press_right_button_does_not_start_panning(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """eventFilter right-press never starts panning even if image is pannable."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)
    dialog.scroll_area.horizontalScrollBar().setMaximum(200)

    press = _make_mouse_press_event(50, 50, button=QtCore.Qt.MouseButton.RightButton)
    result = dialog.eventFilter(dialog.image_label, press)

    assert result is False
    assert dialog._panning is False


def test_event_filter_mouse_move_while_panning_adjusts_scrollbars(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """eventFilter MouseMove while panning shifts scrollbars by the drag delta."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)

    # Set up scrollbar ranges so values can actually change.
    dialog.scroll_area.horizontalScrollBar().setRange(0, 500)
    dialog.scroll_area.verticalScrollBar().setRange(0, 500)
    dialog.scroll_area.horizontalScrollBar().setValue(100)
    dialog.scroll_area.verticalScrollBar().setValue(100)

    dialog._panning = True
    dialog._pan_start = QtCore.QPoint(200, 300)
    dialog._pan_scroll_start = (100, 100)

    # Move 10px right and 20px down → scrollbars decrease by those amounts.
    move = _make_mouse_move_event_with_global(60, 70, global_x=210.0, global_y=320.0)
    result = dialog.eventFilter(dialog.image_label, move)

    assert result is True
    assert dialog.scroll_area.horizontalScrollBar().value() == 100 - 10
    assert dialog.scroll_area.verticalScrollBar().value() == 100 - 20


def test_event_filter_mouse_move_not_panning_falls_through(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """eventFilter MouseMove while not panning returns False."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)
    dialog._panning = False

    move = _make_mouse_move_event_with_global(60, 70)
    result = dialog.eventFilter(dialog.image_label, move)

    assert result is False


def test_event_filter_release_while_panning_stops_panning(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """eventFilter left-release while panning stops the pan and returns True."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)
    dialog._panning = True
    dialog._pan_start = QtCore.QPoint(100, 100)
    dialog._pan_scroll_start = (0, 0)

    release = _make_mouse_release_event(50, 50)
    result = dialog.eventFilter(dialog.image_label, release)

    assert result is True
    assert dialog._panning is False


def test_event_filter_release_not_panning_falls_through(
    qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
):
    """eventFilter left-release when not panning returns False."""
    dialog = _create_dialog(
        sample_image_with_metadata,
        session_factory,
        vector_store,
        tmp_app_paths,
    )
    qtbot.add_widget(dialog)
    dialog._panning = False

    release = _make_mouse_release_event(50, 50)
    result = dialog.eventFilter(dialog.image_label, release)

    assert result is False
