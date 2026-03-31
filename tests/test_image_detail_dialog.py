from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from PySide6 import QtCore, QtGui, QtQuickWidgets, QtWidgets

from photoaident.core.search import SearchResult
from photoaident.db.database import (
    Face,
    FaceState,
    Image,
    ImageMetadata,
    Person,
    TakenAtSource,
)
from photoaident.ui.widgets.image_detail_dialog import (
    ImageDetailDialog,
    _FaceOverlayLabel,
)
from photoaident.ui.widgets.map_widget import MapWidget


def _make_test_image(tmp_path, color: str, img_id: int, file_size: int = 500):
    from PIL import Image as PILImage

    img_path = tmp_path / f"img_{img_id}.jpg"
    PILImage.new("RGB", (100, 100), color).save(img_path)
    return img_path, Image(id=img_id, file_path=str(img_path), file_size=file_size)


def _make_dialog(
    qtbot,
    image,
    session_factory,
    vector_store,
    tmp_app_paths,
    current_index: int = 0,
    all_results: list | None = None,
):
    dialog = ImageDetailDialog(
        image,
        session_factory,
        vector_store,
        tmp_app_paths,
        current_index=current_index,
        all_results=all_results or [],
    )
    qtbot.add_widget(dialog)
    return dialog


def _find_button(dialog, text: str):
    return next(
        (b for b in dialog.findChildren(QtWidgets.QPushButton) if text in b.text()),
        None,
    )


def _make_mock_session_factory(return_image):
    mock_sf = MagicMock()
    mock_session = MagicMock()
    mock_sf.side_effect = contextmanager(lambda: (yield mock_session))

    def fake_execute(_stmt):
        result = MagicMock()
        result.unique = MagicMock(return_value=result)
        result.scalar_one_or_none = MagicMock(return_value=return_image)
        return result

    mock_session.execute = fake_execute
    return mock_sf


def _make_search_result(image_id: int, file_path: str, thumb_path: str | Path):
    return SearchResult(
        image_id=image_id, file_path=file_path, thumb_path=Path(thumb_path)
    )


def test_image_detail_dialog_init(
    qtbot,
    sample_image_with_metadata,
    session_factory,
    vector_store,
    tmp_app_paths,
):
    dialog = _make_dialog(
        qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
    )

    labels = dialog.findChildren(QtWidgets.QLabel)
    texts = [label.text() for label in labels]

    assert any("123" in t for t in texts)
    assert any("TestCamera" in t for t in texts)
    assert any("1000 x 800" in t for t in texts)
    assert any("1.0 KB" in t for t in texts)

    assert not dialog.image_label.pixmap().isNull()


def test_image_detail_dialog_missing_file(
    qtbot, session_factory, vector_store, tmp_app_paths
):
    db_image = Image(id=456, file_path="/non/existent/path.jpg", file_size=0)
    dialog = _make_dialog(qtbot, db_image, session_factory, vector_store, tmp_app_paths)

    assert "not found" in dialog.image_label.text().lower()


def test_image_detail_dialog_close(
    qtbot,
    sample_image_with_metadata,
    session_factory,
    vector_store,
    tmp_app_paths,
):
    dialog = _make_dialog(
        qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
    )

    close_button = None
    for button in dialog.findChildren(QtWidgets.QPushButton):
        if "Close" in button.text():
            close_button = button
            break

    assert close_button is not None

    with qtbot.wait_signal(dialog.finished):
        qtbot.mouseClick(close_button, QtCore.Qt.MouseButton.LeftButton)


def test_large_file_size_shows_mb(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """File size ≥ 1 MB is displayed in MB units."""
    from PIL import Image as PILImage

    img_path = tmp_path / "big.jpg"
    PILImage.new("RGB", (10, 10), "green").save(img_path)

    db_image = Image(id=200, file_path=str(img_path), file_size=2 * 1024 * 1024)
    dialog = _make_dialog(qtbot, db_image, session_factory, vector_store, tmp_app_paths)

    labels = dialog.findChildren(QtWidgets.QLabel)
    texts = [label.text() for label in labels]
    assert any("MB" in t for t in texts)


def test_taken_at_in_metadata_is_displayed(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
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

    dialog = _make_dialog(qtbot, db_image, session_factory, vector_store, tmp_app_paths)

    labels = dialog.findChildren(QtWidgets.QLabel)
    texts = [label.text() for label in labels]
    assert any("2021-03-14" in t for t in texts)


def test_load_image_failure_shows_error(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """A file with invalid image content shows a failure message."""
    bad_file = tmp_path / "bad.jpg"
    bad_file.write_bytes(b"this is not a valid jpeg")

    db_image = Image(id=400, file_path=str(bad_file), file_size=25)
    dialog = _make_dialog(qtbot, db_image, session_factory, vector_store, tmp_app_paths)

    text = dialog.image_label.text().lower()
    assert "failed" in text


def test_update_image_display_without_pixmap_is_noop(
    qtbot, session_factory, vector_store, tmp_app_paths
):
    """_update_image_display returns early if _original_pixmap is not set."""
    db_image = Image(id=500, file_path="/nonexistent.jpg", file_size=0)
    dialog = _make_dialog(qtbot, db_image, session_factory, vector_store, tmp_app_paths)

    assert not hasattr(dialog, "_original_pixmap")
    dialog._update_image_display()


def test_resize_event_schedules_redisplay(
    qtbot,
    sample_image_with_metadata,
    session_factory,
    vector_store,
    tmp_app_paths,
):
    """resizeEvent does not raise and schedules a display update."""
    dialog = _make_dialog(
        qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
    )

    event = QtGui.QResizeEvent(QtCore.QSize(900, 700), QtCore.QSize(800, 600))
    dialog.resizeEvent(event)


def test_label_faces_button_enabled_when_unidentified_faces(
    qtbot,
    sample_image_with_metadata,
    session_factory,
    vector_store,
    tmp_app_paths,
):
    """Label button is enabled when the image has at least one unidentified face."""
    dialog = _make_dialog(
        qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
    )

    buttons = dialog.findChildren(QtWidgets.QPushButton)
    label_btn = next((b for b in buttons if "Label" in b.text()), None)
    assert label_btn is not None
    assert label_btn.isEnabled()


def test_label_faces_button_disabled_when_no_unidentified_faces(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Label button is disabled when the image has no unidentified faces."""
    img_path, db_image = _make_test_image(tmp_path, "blue", 600)
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

    dialog = _make_dialog(qtbot, db_image, session_factory, vector_store, tmp_app_paths)

    label_btn = _find_button(dialog, "Label")
    assert label_btn is not None
    assert not label_btn.isEnabled()


def test_label_faces_button_emits_signal(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Clicking the label button emits navigate_to_labelling with the image id."""
    img_path, db_image = _make_test_image(tmp_path, "green", 700)
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

    dialog = _make_dialog(qtbot, db_image, session_factory, vector_store, tmp_app_paths)

    emitted_ids: list[int] = []
    dialog.navigate_to_labelling.connect(emitted_ids.append)

    label_btn = _find_button(dialog, "Label")
    assert label_btn is not None

    qtbot.mouseClick(label_btn, QtCore.Qt.MouseButton.LeftButton)

    assert emitted_ids == [700]


# ===========================================================================
# show_in_file_manager_button
# ===========================================================================


def test_show_in_file_manager_button_exists(
    qtbot,
    sample_image_with_metadata,
    session_factory,
    vector_store,
    tmp_app_paths,
):
    """Show in File Manager button is present in the dialog."""
    dialog = _make_dialog(
        qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
    )

    buttons = dialog.findChildren(QtWidgets.QPushButton)
    show_btn = next(
        (b for b in buttons if b.text() == dialog.tr("Show in File Manager")), None
    )
    assert show_btn is not None


def test_show_in_file_manager_button_calls_reveal(
    qtbot,
    sample_image_with_metadata,
    session_factory,
    vector_store,
    tmp_app_paths,
):
    """Clicking the button calls reveal_in_file_manager with the image file path."""
    dialog = _make_dialog(
        qtbot, sample_image_with_metadata, session_factory, vector_store, tmp_app_paths
    )

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
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Show in File Manager button is enabled regardless of face labelling state."""
    img_path, db_image = _make_test_image(tmp_path, "cyan", 800)
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

    dialog = _make_dialog(qtbot, db_image, session_factory, vector_store, tmp_app_paths)

    show_btn = next(
        (
            b
            for b in dialog.findChildren(QtWidgets.QPushButton)
            if b.text() == dialog.tr("Show in File Manager")
        ),
        None,
    )
    assert show_btn is not None
    assert show_btn.isEnabled()


# ===========================================================================
# navigate_to_browse
# ===========================================================================


def test_browse_photo_folder_button_emits_navigate_to_browse(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Clicking 'Browse Photo Folder' emits navigate_to_browse with the file path."""
    img_path, db_image = _make_test_image(tmp_path, "red", 1100)
    db_image.faces = []

    dialog = _make_dialog(qtbot, db_image, session_factory, vector_store, tmp_app_paths)

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

    with qtbot.wait_signal(dialog.finished):
        qtbot.mouseClick(browse_btn, QtCore.Qt.MouseButton.LeftButton)

    assert received == [str(img_path)]


def test_file_path_link_emits_navigate_to_browse(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Activating the file path link in the metadata panel emits navigate_to_browse."""
    img_path, db_image = _make_test_image(tmp_path, "blue", 1101)
    db_image.faces = []

    dialog = _make_dialog(qtbot, db_image, session_factory, vector_store, tmp_app_paths)

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

    with qtbot.wait_signal(dialog.finished):
        file_path_label.linkActivated.emit("#")

    assert received == [str(img_path)]
    assert dialog.isHidden()


# ===========================================================================
# Tooltip tests — _build_face_display_info
# ===========================================================================


def test_face_tooltip_identified_shows_person_name(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
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

    dialog = _make_dialog(qtbot, db_image, session_factory, vector_store, tmp_app_paths)

    regions = dialog._build_face_display_info()
    assert len(regions) == 1
    assert regions[0][2] == "Alice"
    assert regions[0][1] == QtCore.Qt.GlobalColor.green


def test_face_tooltip_anonymous_shows_anonymous(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
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

    dialog = _make_dialog(qtbot, db_image, session_factory, vector_store, tmp_app_paths)

    regions = dialog._build_face_display_info()
    assert len(regions) == 1
    assert regions[0][2] == dialog.tr("Anonymous")
    assert regions[0][1] == QtCore.Qt.GlobalColor.green


def test_face_tooltip_unidentified_no_vector_match(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """_build_face_display_info: UNIDENTIFIED with no vector match → Unknown + red."""
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

    dialog = _make_dialog(qtbot, db_image, session_factory, vector_store, tmp_app_paths)

    regions = dialog._build_face_display_info()
    assert len(regions) == 1
    assert regions[0][2] == dialog.tr("Unknown")
    assert regions[0][1] == QtCore.Qt.GlobalColor.red


def test_face_tooltip_deleted_face_excluded(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
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

    dialog = _make_dialog(qtbot, db_image, session_factory, vector_store, tmp_app_paths)

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

    pixmap = QtGui.QPixmap(200, 200)
    pixmap.fill(QtGui.QColor("white"))
    label.setPixmap(pixmap)
    label.resize(200, 200)

    region = (QtCore.QRectF(50, 50, 60, 60), "Alice")
    label.set_face_regions([region], QtCore.QSize(200, 200))

    with patch(
        "photoaident.ui.widgets.image_detail_dialog.QtWidgets.QToolTip.showText"
    ) as mock_show:
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


def _make_mouse_move_event(x: float, y: float):
    return QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        QtCore.QPointF(x, y),
        QtCore.QPointF(x, y),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )


def test_face_overlay_label_no_regions_early_return(qtbot):
    """mouseMoveEvent returns early when no face regions are set."""
    label = _FaceOverlayLabel()
    qtbot.add_widget(label)

    pixmap = QtGui.QPixmap(200, 200)
    pixmap.fill(QtGui.QColor("white"))
    label.setPixmap(pixmap)
    label.resize(200, 200)

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
        label.mouseMoveEvent(_make_mouse_move_event(10.0, 10.0))
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
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
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

    dialog = _make_dialog(qtbot, db_image, session_factory, vector_store, tmp_app_paths)

    assert hasattr(dialog, "_original_pixmap")
    assert not dialog._original_pixmap.isNull()

    img = dialog._original_pixmap.toImage()

    def pixel_color(x: int, y: int) -> QtGui.QColor:
        return QtGui.QColor(img.pixel(x, y))

    green_identified = pixel_color(30, 10)
    assert green_identified.green() > green_identified.red()
    assert green_identified.green() > green_identified.blue()

    green_anon = pixel_color(90, 10)
    assert green_anon.green() > green_anon.red()
    assert green_anon.green() > green_anon.blue()

    red = pixel_color(150, 10)
    assert red.red() > red.green() and red.red() > red.blue()


def test_image_detail_dialog_with_exif_rotation(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Image with EXIF rotation triggers the transformation block."""
    from PIL import Image as PILImage

    img_path = tmp_path / "rotated.jpg"
    img = PILImage.new("RGB", (100, 200), "blue")
    exif = img.getexif()
    exif[0x0112] = 6
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

    dialog = _make_dialog(qtbot, db_image, session_factory, vector_store, tmp_app_paths)

    assert hasattr(dialog, "_original_pixmap")
    pm = dialog._original_pixmap
    assert pm.width() == 200
    assert pm.height() == 100


# ===========================================================================
# GPS map widget presence tests
# ===========================================================================


def test_map_shown_when_gps_available(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """A MapWidget is added to the metadata panel when GPS coordinates are present."""
    img_path, db_image = _make_test_image(tmp_path, "green", 1001)
    db_image.metadata_rel = ImageMetadata(
        width=100,
        height=100,
        gps_lat=48.137154,
        gps_lon=11.576124,
        taken_at_source=TakenAtSource.EXIF,
    )
    db_image.faces = []

    with patch.object(QtQuickWidgets.QQuickWidget, "setSource"):
        dialog = _make_dialog(
            qtbot, db_image, session_factory, vector_store, tmp_app_paths
        )

    map_widgets = dialog.findChildren(MapWidget)
    assert len(map_widgets) == 1


def test_map_not_shown_when_no_gps(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """No MapWidget is added when metadata exists but has no GPS coordinates."""
    img_path, db_image = _make_test_image(tmp_path, "red", 1002)
    db_image.metadata_rel = ImageMetadata(
        width=100,
        height=100,
        taken_at_source=TakenAtSource.EXIF,
    )
    db_image.faces = []

    dialog = _make_dialog(qtbot, db_image, session_factory, vector_store, tmp_app_paths)

    map_widgets = dialog.findChildren(MapWidget)
    assert len(map_widgets) == 0


def test_map_not_shown_when_no_metadata(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """No MapWidget is added when metadata_rel is None."""
    img_path, db_image = _make_test_image(tmp_path, "blue", 1003)
    db_image.faces = []

    dialog = _make_dialog(qtbot, db_image, session_factory, vector_store, tmp_app_paths)

    map_widgets = dialog.findChildren(MapWidget)
    assert len(map_widgets) == 0


# ===========================================================================
# Navigation buttons — back/next
# ===========================================================================


def test_navigation_buttons_exist(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Previous and Next buttons are present in the dialog."""
    img_path, db_image = _make_test_image(tmp_path, "red", 2000)
    db_image.faces = []

    results = [
        _make_search_result(2000, str(img_path), str(tmp_path / "t.jpg")),
        _make_search_result(2001, str(img_path), str(tmp_path / "t2.jpg")),
    ]

    dialog = _make_dialog(
        qtbot, db_image, session_factory, vector_store, tmp_app_paths, 0, results
    )

    prev_btn = _find_button(dialog, "Previous")
    next_btn = _find_button(dialog, "Next")

    assert prev_btn is not None
    assert next_btn is not None


def test_navigation_buttons_disabled_at_first_position(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Previous is disabled when viewing the first image in the list."""
    img_path, db_image = _make_test_image(tmp_path, "green", 3000)
    db_image.faces = []

    results = [
        _make_search_result(3000, str(img_path), str(tmp_path / "t1.jpg")),
        _make_search_result(3001, str(img_path), str(tmp_path / "t2.jpg")),
    ]

    dialog = _make_dialog(
        qtbot, db_image, session_factory, vector_store, tmp_app_paths, 0, results
    )

    prev_btn = _find_button(dialog, "Previous")
    next_btn = _find_button(dialog, "Next")

    assert prev_btn is not None
    assert next_btn is not None
    assert not prev_btn.isEnabled()
    assert next_btn.isEnabled()


def test_navigation_buttons_disabled_at_last_position(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Next is disabled when viewing the last image in the list."""
    img_path, db_image = _make_test_image(tmp_path, "blue", 4000)
    db_image.faces = []

    results = [
        _make_search_result(4000, str(img_path), str(tmp_path / "t1.jpg")),
        _make_search_result(4001, str(img_path), str(tmp_path / "t2.jpg")),
    ]

    dialog = _make_dialog(
        qtbot, db_image, session_factory, vector_store, tmp_app_paths, 1, results
    )

    prev_btn = _find_button(dialog, "Previous")
    next_btn = _find_button(dialog, "Next")

    assert prev_btn is not None
    assert next_btn is not None
    assert prev_btn.isEnabled()
    assert not next_btn.isEnabled()


def test_navigation_buttons_both_disabled_when_no_results(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Both buttons are disabled when there are no results."""
    img_path, db_image = _make_test_image(tmp_path, "yellow", 5000)
    db_image.faces = []

    dialog = _make_dialog(
        qtbot, db_image, session_factory, vector_store, tmp_app_paths, 0, []
    )

    prev_btn = _find_button(dialog, "Previous")
    next_btn = _find_button(dialog, "Next")

    assert prev_btn is not None
    assert next_btn is not None
    assert not prev_btn.isEnabled()
    assert not next_btn.isEnabled()


def test_navigation_buttons_both_disabled_for_single_result(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Both buttons are disabled when there is only one image in the results."""
    img_path, db_image = _make_test_image(tmp_path, "red", 5500)
    db_image.faces = []

    dialog = _make_dialog(
        qtbot,
        db_image,
        session_factory,
        vector_store,
        tmp_app_paths,
        0,
        [_make_search_result(5500, str(img_path), str(tmp_path / "t.jpg"))],
    )

    prev_btn = _find_button(dialog, "Previous")
    next_btn = _find_button(dialog, "Next")

    assert prev_btn is not None
    assert next_btn is not None
    assert not prev_btn.isEnabled()
    assert not next_btn.isEnabled()


def test_navigation_label_shows_position(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """The navigation label displays the correct position (e.g., '2 / 3')."""
    img_path, db_image = _make_test_image(tmp_path, "purple", 6000)
    db_image.faces = []

    results = [
        _make_search_result(6000, str(img_path), str(tmp_path / "t1.jpg")),
        _make_search_result(6001, str(img_path), str(tmp_path / "t2.jpg")),
        _make_search_result(6002, str(img_path), str(tmp_path / "t3.jpg")),
    ]

    dialog = _make_dialog(
        qtbot, db_image, session_factory, vector_store, tmp_app_paths, 1, results
    )

    assert dialog._nav_label.text() == "2 / 3"


def test_navigation_label_empty_when_no_results(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """The navigation label is empty when there are no results."""
    img_path, db_image = _make_test_image(tmp_path, "orange", 7000)
    db_image.faces = []

    dialog = _make_dialog(
        qtbot, db_image, session_factory, vector_store, tmp_app_paths, 0, []
    )

    assert dialog._nav_label.text() == ""


def test_keyboard_left_arrow_navigates_previous(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Pressing the left arrow key calls _show_previous_image."""
    img_path, db_image = _make_test_image(tmp_path, "pink", 8000)
    db_image.faces = []

    results = [
        _make_search_result(8000, str(img_path), str(tmp_path / "t1.jpg")),
        _make_search_result(8001, str(img_path), str(tmp_path / "t2.jpg")),
    ]

    dialog = _make_dialog(
        qtbot, db_image, session_factory, vector_store, tmp_app_paths, 1, results
    )

    with patch.object(dialog, "_show_previous_image") as mock_prev:
        key_event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Left,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        dialog.keyPressEvent(key_event)
        mock_prev.assert_called_once()


def test_keyboard_right_arrow_navigates_next(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Pressing the right arrow key calls _show_next_image."""
    img_path, db_image = _make_test_image(tmp_path, "cyan", 9000)
    db_image.faces = []

    results = [
        _make_search_result(9000, str(img_path), str(tmp_path / "t1.jpg")),
        _make_search_result(9001, str(img_path), str(tmp_path / "t2.jpg")),
    ]

    dialog = _make_dialog(
        qtbot, db_image, session_factory, vector_store, tmp_app_paths, 0, results
    )

    with patch.object(dialog, "_show_next_image") as mock_next:
        key_event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Right,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        dialog.keyPressEvent(key_event)
        mock_next.assert_called_once()


def test_other_keys_do_not_navigate(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """Keys other than left/right do not trigger navigation."""
    img_path, db_image = _make_test_image(tmp_path, "gray", 10000)
    db_image.faces = []

    results = [
        _make_search_result(10000, str(img_path), str(tmp_path / "t1.jpg")),
        _make_search_result(10001, str(img_path), str(tmp_path / "t2.jpg")),
    ]

    dialog = _make_dialog(
        qtbot, db_image, session_factory, vector_store, tmp_app_paths, 1, results
    )

    with (
        patch.object(dialog, "_show_previous_image") as mock_prev,
        patch.object(dialog, "_show_next_image") as mock_next,
    ):
        for key in (
            QtCore.Qt.Key.Key_Up,
            QtCore.Qt.Key.Key_Down,
            QtCore.Qt.Key.Key_Space,
        ):
            key_event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress,
                key,
                QtCore.Qt.KeyboardModifier.NoModifier,
            )
            dialog.keyPressEvent(key_event)

        mock_prev.assert_not_called()
        mock_next.assert_not_called()


def test_show_next_image_disabled_at_end(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """_show_next_image does nothing when already at the last image."""
    img_path, db_image = _make_test_image(tmp_path, "white", 11000)
    db_image.faces = []

    results = [
        _make_search_result(11000, str(img_path), str(tmp_path / "t1.jpg")),
        _make_search_result(11001, str(img_path), str(tmp_path / "t2.jpg")),
    ]

    dialog = _make_dialog(
        qtbot, db_image, session_factory, vector_store, tmp_app_paths, 1, results
    )

    assert dialog._current_index == 1
    dialog._show_next_image()
    assert dialog._current_index == 1


def test_show_previous_image_disabled_at_start(
    qtbot, tmp_path, session_factory, vector_store, tmp_app_paths
):
    """_show_previous_image does nothing when already at the first image."""
    img_path, db_image = _make_test_image(tmp_path, "black", 12000)
    db_image.faces = []

    results = [
        _make_search_result(12000, str(img_path), str(tmp_path / "t1.jpg")),
        _make_search_result(12001, str(img_path), str(tmp_path / "t2.jpg")),
    ]

    dialog = _make_dialog(
        qtbot, db_image, session_factory, vector_store, tmp_app_paths, 0, results
    )

    assert dialog._current_index == 0
    dialog._show_previous_image()
    assert dialog._current_index == 0


# ===========================================================================
# _add_meta_row early-return and _load_image_by_index
# ===========================================================================


def test_add_meta_row_skipped_when_value_is_none(
    qtbot, session_factory, vector_store, tmp_app_paths
):
    """_add_meta_row adds no row when value_text is None."""
    db_image = Image(id=9001, file_path="/nonexistent.jpg", file_size=0)
    db_image.faces = []
    dialog = _make_dialog(qtbot, db_image, session_factory, vector_store, tmp_app_paths)

    layout = QtWidgets.QVBoxLayout()
    dialog._add_meta_row(layout, "Label", None)
    assert layout.count() == 0


def test_show_next_image_loads_next_image(qtbot, tmp_path, vector_store, tmp_app_paths):
    """Clicking Next actually loads the next image via _load_image_by_index."""
    img1_path, db_image1 = _make_test_image(tmp_path, "red", 9100, 500)
    img2_path, db_image2 = _make_test_image(tmp_path, "blue", 9101, 600)
    db_image1.faces = []
    db_image2.faces = []

    results = [
        SearchResult(
            image_id=9100, file_path=str(img1_path), thumb_path=tmp_path / "t1.jpg"
        ),
        SearchResult(
            image_id=9101, file_path=str(img2_path), thumb_path=tmp_path / "t2.jpg"
        ),
    ]

    mock_sf = _make_mock_session_factory(db_image2)

    dialog = ImageDetailDialog(
        db_image1,
        mock_sf,
        vector_store,
        tmp_app_paths,
        current_index=0,
        all_results=results,
    )
    qtbot.add_widget(dialog)

    assert dialog._current_index == 0
    dialog._show_next_image()
    assert dialog._current_index == 1
    assert dialog.image_data.id == 9101


def test_show_previous_image_loads_previous_image(
    qtbot, tmp_path, vector_store, tmp_app_paths
):
    """Clicking Previous actually loads the previous image via _load_image_by_index."""
    img1_path, db_image1 = _make_test_image(tmp_path, "green", 9200, 500)
    img2_path, db_image2 = _make_test_image(tmp_path, "yellow", 9201, 600)
    db_image1.faces = []
    db_image2.faces = []

    results = [
        SearchResult(
            image_id=9200, file_path=str(img1_path), thumb_path=tmp_path / "t1.jpg"
        ),
        SearchResult(
            image_id=9201, file_path=str(img2_path), thumb_path=tmp_path / "t2.jpg"
        ),
    ]

    mock_sf = _make_mock_session_factory(db_image1)

    dialog = ImageDetailDialog(
        db_image2,
        mock_sf,
        vector_store,
        tmp_app_paths,
        current_index=1,
        all_results=results,
    )
    qtbot.add_widget(dialog)

    assert dialog._current_index == 1
    dialog._show_previous_image()
    assert dialog._current_index == 0
    assert dialog.image_data.id == 9200


def test_load_image_by_index_clears_resolved_names(
    qtbot, tmp_path, vector_store, tmp_app_paths
):
    """_load_image_by_index clears _resolved_names before loading the new image."""
    img1_path, db_image1 = _make_test_image(tmp_path, "red", 9300, 500)
    img2_path, db_image2 = _make_test_image(tmp_path, "blue", 9301, 600)
    db_image1.faces = []
    db_image2.faces = []

    results = [
        SearchResult(
            image_id=9300, file_path=str(img1_path), thumb_path=tmp_path / "t1.jpg"
        ),
        SearchResult(
            image_id=9301, file_path=str(img2_path), thumb_path=tmp_path / "t2.jpg"
        ),
    ]

    mock_sf = _make_mock_session_factory(db_image2)

    dialog = ImageDetailDialog(
        db_image1,
        mock_sf,
        vector_store,
        tmp_app_paths,
        current_index=0,
        all_results=results,
    )
    qtbot.add_widget(dialog)

    dialog._resolved_names = {1: ("Bob", 0.9), 2: ("Alice", 0.85)}
    assert len(dialog._resolved_names) == 2

    dialog._load_image_by_index()
    assert dialog._resolved_names == {}


def test_load_image_by_index_updates_navigation(
    qtbot, tmp_path, vector_store, tmp_app_paths
):
    """_load_image_by_index calls _update_navigation after loading."""
    img1_path, db_image1 = _make_test_image(tmp_path, "red", 9400, 500)
    img2_path, db_image2 = _make_test_image(tmp_path, "blue", 9401, 600)
    img3_path, db_image3 = _make_test_image(tmp_path, "green", 9402, 700)
    db_image1.faces = []
    db_image2.faces = []
    db_image3.faces = []

    results = [
        SearchResult(
            image_id=9400, file_path=str(img1_path), thumb_path=tmp_path / "t1.jpg"
        ),
        SearchResult(
            image_id=9401, file_path=str(img2_path), thumb_path=tmp_path / "t2.jpg"
        ),
        SearchResult(
            image_id=9402, file_path=str(img3_path), thumb_path=tmp_path / "t3.jpg"
        ),
    ]

    mock_sf = _make_mock_session_factory(db_image2)

    dialog = ImageDetailDialog(
        db_image1,
        mock_sf,
        vector_store,
        tmp_app_paths,
        current_index=0,
        all_results=results,
    )
    qtbot.add_widget(dialog)

    assert dialog._current_index == 0
    dialog._show_next_image()
    assert dialog._current_index == 1

    assert dialog._nav_label.text() == "2 / 3"
    assert dialog._prev_btn.isEnabled()
    assert dialog._next_btn.isEnabled()


# ===========================================================================
# Metadata panel update on navigation
# ===========================================================================


def test_load_image_by_index_updates_metadata_panel(
    qtbot, tmp_path, vector_store, tmp_app_paths
):
    """_load_image_by_index calls _update_metadata_panel after loading the new image."""
    img1_path, db_image1 = _make_test_image(tmp_path, "red", 9500, 500)
    img2_path, db_image2 = _make_test_image(tmp_path, "blue", 9501, 600)
    db_image1.faces = []
    db_image2.faces = []

    results = [
        SearchResult(
            image_id=9500, file_path=str(img1_path), thumb_path=tmp_path / "t1.jpg"
        ),
        SearchResult(
            image_id=9501, file_path=str(img2_path), thumb_path=tmp_path / "t2.jpg"
        ),
    ]

    mock_sf = _make_mock_session_factory(db_image2)

    dialog = ImageDetailDialog(
        db_image1,
        mock_sf,
        vector_store,
        tmp_app_paths,
        current_index=0,
        all_results=results,
    )
    qtbot.add_widget(dialog)

    with patch.object(dialog, "_update_metadata_panel") as mock_update:
        dialog._load_image_by_index()
        mock_update.assert_called_once()


def test_update_metadata_panel_updates_id_label(
    qtbot, tmp_path, vector_store, tmp_app_paths
):
    """_update_metadata_panel updates the ID label text."""
    img_path, db_image = _make_test_image(tmp_path, "red", 9600)
    db_image.faces = []

    dialog = _make_dialog(qtbot, db_image, MagicMock(), vector_store, tmp_app_paths)

    db_image.id = 9601
    dialog._update_metadata_panel()

    id_label = dialog._metadata_widgets.get("id")
    assert id_label is not None
    assert id_label.text() == "9601"


def test_update_metadata_panel_updates_file_path_label(
    qtbot, tmp_path, vector_store, tmp_app_paths
):
    """_update_metadata_panel updates the file path label text."""
    img_path, db_image = _make_test_image(tmp_path, "blue", 9700)
    db_image.faces = []

    dialog = _make_dialog(qtbot, db_image, MagicMock(), vector_store, tmp_app_paths)

    new_path = str(tmp_path / "new_path.jpg")
    db_image.file_path = new_path
    dialog._update_metadata_panel()

    fp_label = dialog._metadata_widgets.get("file_path")
    assert fp_label is not None
    assert new_path in fp_label.text()


def test_update_metadata_panel_updates_file_size_label(
    qtbot, tmp_path, vector_store, tmp_app_paths
):
    """_update_metadata_panel updates the file size label with formatted value."""
    img_path, db_image = _make_test_image(tmp_path, "green", 9800)
    db_image.faces = []

    dialog = _make_dialog(qtbot, db_image, MagicMock(), vector_store, tmp_app_paths)

    db_image.file_size = 2048
    dialog._update_metadata_panel()

    fs_label = dialog._metadata_widgets.get("file_size")
    assert fs_label is not None
    assert "KB" in fs_label.text()


def test_update_metadata_panel_updates_dimensions_label(
    qtbot, tmp_path, vector_store, tmp_app_paths
):
    """_update_metadata_panel updates the dimensions label from metadata_rel."""
    img_path, db_image = _make_test_image(tmp_path, "yellow", 9900)
    db_image.metadata_rel = ImageMetadata(width=200, height=150)
    db_image.faces = []

    dialog = _make_dialog(qtbot, db_image, MagicMock(), vector_store, tmp_app_paths)

    db_image.metadata_rel.width = 300
    db_image.metadata_rel.height = 400
    dialog._update_metadata_panel()

    dim_label = dialog._metadata_widgets.get("dimensions")
    assert dim_label is not None
    assert "300 x 400" in dim_label.text()


def test_update_metadata_panel_updates_taken_at_label(
    qtbot, tmp_path, vector_store, tmp_app_paths
):
    """_update_metadata_panel updates the taken_at label from metadata_rel."""
    img_path, db_image = _make_test_image(tmp_path, "cyan", 10000)
    db_image.metadata_rel = ImageMetadata(taken_at=datetime(2020, 1, 1, 12, 0, 0))
    db_image.faces = []

    dialog = _make_dialog(qtbot, db_image, MagicMock(), vector_store, tmp_app_paths)

    db_image.metadata_rel.taken_at = datetime(2023, 6, 15, 9, 30, 0)
    dialog._update_metadata_panel()

    ta_label = dialog._metadata_widgets.get("taken_at")
    assert ta_label is not None
    assert "2023-06-15" in ta_label.text()


def test_update_metadata_panel_updates_camera_label(
    qtbot, tmp_path, vector_store, tmp_app_paths
):
    """_update_metadata_panel updates the camera label from metadata_rel."""
    img_path, db_image = _make_test_image(tmp_path, "magenta", 10100)
    db_image.metadata_rel = ImageMetadata(camera_make="Canon", camera_model="EOS")
    db_image.faces = []

    dialog = _make_dialog(qtbot, db_image, MagicMock(), vector_store, tmp_app_paths)

    db_image.metadata_rel.camera_make = "Nikon"
    db_image.metadata_rel.camera_model = "Z6"
    dialog._update_metadata_panel()

    cam_label = dialog._metadata_widgets.get("camera")
    assert cam_label is not None
    assert "Nikon" in cam_label.text()
    assert "Z6" in cam_label.text()


def test_update_metadata_panel_updates_map_widget(
    qtbot, tmp_path, vector_store, tmp_app_paths
):
    """_update_metadata_panel calls set_center and set_marker on the map widget."""
    img_path, db_image = _make_test_image(tmp_path, "white", 10200)
    db_image.metadata_rel = ImageMetadata(gps_lat=48.0, gps_lon=11.0)
    db_image.faces = []

    with patch.object(QtQuickWidgets.QQuickWidget, "setSource"):
        dialog = _make_dialog(qtbot, db_image, MagicMock(), vector_store, tmp_app_paths)

    assert dialog._map_widget is not None
    with (
        patch.object(dialog._map_widget, "set_center") as mock_center,
        patch.object(dialog._map_widget, "set_marker") as mock_marker,
    ):
        db_image.metadata_rel.gps_lat = 52.5
        db_image.metadata_rel.gps_lon = 13.4
        dialog._update_metadata_panel()
        mock_center.assert_called_once_with(52.5, 13.4, zoom=14)
        mock_marker.assert_called_once_with(52.5, 13.4)


def test_show_next_image_updates_metadata_panel(
    qtbot, tmp_path, vector_store, tmp_app_paths
):
    """Clicking Next updates the metadata panel to show the new image's metadata."""
    img1_path, db_image1 = _make_test_image(tmp_path, "red", 10300, 500)
    img2_path, db_image2 = _make_test_image(tmp_path, "blue", 10301, 600)
    db_image1.metadata_rel = ImageMetadata(width=100, height=100)
    db_image1.faces = []
    db_image2.metadata_rel = ImageMetadata(width=200, height=150)
    db_image2.faces = []

    results = [
        SearchResult(
            image_id=10300, file_path=str(img1_path), thumb_path=tmp_path / "t1.jpg"
        ),
        SearchResult(
            image_id=10301, file_path=str(img2_path), thumb_path=tmp_path / "t2.jpg"
        ),
    ]

    mock_sf = _make_mock_session_factory(db_image2)

    dialog = ImageDetailDialog(
        db_image1,
        mock_sf,
        vector_store,
        tmp_app_paths,
        current_index=0,
        all_results=results,
    )
    qtbot.add_widget(dialog)

    assert dialog._metadata_widgets["id"].text() == "10300"

    dialog._show_next_image()

    assert dialog._metadata_widgets["id"].text() == "10301"


def test_show_previous_image_updates_metadata_panel(
    qtbot, tmp_path, vector_store, tmp_app_paths
):
    """Clicking Previous updates the metadata panel to show the previous image."""
    img1_path, db_image1 = _make_test_image(tmp_path, "green", 10400, 500)
    img2_path, db_image2 = _make_test_image(tmp_path, "yellow", 10401, 600)
    db_image1.faces = []
    db_image2.metadata_rel = ImageMetadata(taken_at=datetime(2022, 5, 10))
    db_image2.faces = []

    results = [
        SearchResult(
            image_id=10400, file_path=str(img1_path), thumb_path=tmp_path / "t1.jpg"
        ),
        SearchResult(
            image_id=10401, file_path=str(img2_path), thumb_path=tmp_path / "t2.jpg"
        ),
    ]

    mock_sf = _make_mock_session_factory(db_image1)

    dialog = ImageDetailDialog(
        db_image2,
        mock_sf,
        vector_store,
        tmp_app_paths,
        current_index=1,
        all_results=results,
    )
    qtbot.add_widget(dialog)

    assert dialog._metadata_widgets["id"].text() == "10401"

    dialog._show_previous_image()

    assert dialog._metadata_widgets["id"].text() == "10400"
