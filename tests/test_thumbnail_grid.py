from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage
from PySide6 import QtCore, QtWidgets

from photoaident.core.search import SearchResult, SortOrder
from photoaident.db.database import (
    Face,
    FaceState,
    Image as DBImage,
    get_engine,
    get_session_factory,
)
from photoaident.db.migrate import apply_migrations
from photoaident.db.vector_store import VectorStore
from photoaident.paths import AppPaths
from photoaident.ui.widgets.thumbnail_grid import (
    PAGE_SIZE,
    ThumbnailGrid,
    ThumbnailWidget,
    _get_scaled_size,
    _has_unidentified_faces,
    _read_pixmap,
)


@pytest.fixture
def sample_image(tmp_path):
    img_path = tmp_path / "test.jpg"
    img = PILImage.new("RGB", (800, 600), "blue")
    img.save(img_path, "JPEG")
    return img_path


@pytest.fixture
def mock_session_factory():
    factory = MagicMock()
    # Mocking the context manager session_factory() as session:
    session = factory.return_value.__enter__.return_value
    # Mocking session.scalar(...) to return 0 by default
    session.scalar.return_value = 0
    return factory


@pytest.fixture
def mock_vector_store():
    return MagicMock(spec=VectorStore)


def test_thumbnail_widget_init(qtbot, sample_image, tmp_path, mock_session_factory):
    thumb_path = tmp_path / "thumb.jpg"
    widget = ThumbnailWidget(
        image_id=1,
        file_path=str(sample_image),
        thumb_path=thumb_path,
        session_factory=mock_session_factory,
    )
    qtbot.addWidget(widget)

    assert widget.image_id == 1
    assert widget.file_path == str(sample_image)
    assert widget.thumb_path == thumb_path

    # Check if thumbnail was generated because it didn't exist
    assert thumb_path.exists()
    assert widget.image_label.pixmap() is not None
    assert not widget.image_label.pixmap().isNull()


def test_thumbnail_widget_with_existing_thumb(
    qtbot, sample_image, tmp_path, mock_session_factory
):
    thumb_path = tmp_path / "thumb_exists.jpg"
    # Create a dummy thumbnail
    PILImage.new("RGB", (150, 150), "red").save(thumb_path, "JPEG")

    widget = ThumbnailWidget(
        image_id=1,
        file_path=str(sample_image),
        thumb_path=thumb_path,
        session_factory=mock_session_factory,
    )
    qtbot.addWidget(widget)

    # It should use the existing thumbnail
    assert thumb_path.exists()
    # Check if it loaded (label should have pixmap)
    assert widget.image_label.pixmap() is not None


def test_thumbnail_widget_click(qtbot, sample_image, tmp_path, mock_session_factory):
    """Clicking the view button on the overlay emits the clicked signal."""
    thumb_path = tmp_path / "thumb_click.jpg"
    widget = ThumbnailWidget(
        1, str(sample_image), thumb_path, session_factory=mock_session_factory
    )
    qtbot.addWidget(widget)

    widget._overlay.show()

    with qtbot.waitSignal(widget.clicked) as blocker:
        qtbot.mouseClick(widget._overlay.view_btn, QtCore.Qt.MouseButton.LeftButton)

    assert blocker.args == [1]


def test_thumbnail_widget_error_loading(qtbot, tmp_path, mock_session_factory):
    # Pass a non-existent file path and non-existent thumb path
    thumb_path = tmp_path / "non_existent_thumb.jpg"
    widget = ThumbnailWidget(
        1, "non_existent_file.jpg", thumb_path, session_factory=mock_session_factory
    )
    qtbot.addWidget(widget)

    # Should show error text
    assert widget.image_label.text() == "Error loading image"


# --- hover overlay tests ---


def test_hover_overlay_hidden_initially(
    qtbot, sample_image, tmp_path, mock_session_factory
):
    """The overlay is hidden before any hover event."""
    widget = ThumbnailWidget(
        1, str(sample_image), tmp_path / "t.jpg", session_factory=mock_session_factory
    )
    qtbot.addWidget(widget)

    assert widget._overlay.isHidden()


def test_hover_overlay_shows_on_hoverenter(
    qtbot, sample_image, tmp_path, mock_session_factory
):
    """Sending a HoverEnter event shows the overlay."""
    widget = ThumbnailWidget(
        1, str(sample_image), tmp_path / "t.jpg", session_factory=mock_session_factory
    )
    qtbot.addWidget(widget)
    widget.show()

    ev = QtCore.QEvent(QtCore.QEvent.Type.HoverEnter)
    QtWidgets.QApplication.sendEvent(widget, ev)

    assert not widget._overlay.isHidden()


def test_hover_overlay_hides_on_hoverleave(
    qtbot, sample_image, tmp_path, mock_session_factory
):
    """Sending a HoverLeave event hides the overlay."""
    widget = ThumbnailWidget(
        1, str(sample_image), tmp_path / "t.jpg", session_factory=mock_session_factory
    )
    qtbot.addWidget(widget)
    widget.show()

    # Show first
    ev_enter = QtCore.QEvent(QtCore.QEvent.Type.HoverEnter)
    QtWidgets.QApplication.sendEvent(widget, ev_enter)
    assert not widget._overlay.isHidden()

    # Then leave
    ev_leave = QtCore.QEvent(QtCore.QEvent.Type.HoverLeave)
    QtWidgets.QApplication.sendEvent(widget, ev_leave)
    assert widget._overlay.isHidden()


def test_label_button_disabled_without_faces(
    qtbot, sample_image, tmp_path, mock_session_factory
):
    """label_btn is disabled when there are no unidentified faces."""
    widget = ThumbnailWidget(
        1, str(sample_image), tmp_path / "t.jpg", mock_session_factory
    )
    qtbot.addWidget(widget)
    widget.show()  # parent must be visible for showEvent to fire on child

    widget._overlay.show()
    QtWidgets.QApplication.processEvents()

    assert not widget._overlay.label_btn.isEnabled()


def test_label_button_enabled_with_unidentified_face(
    qtbot, sample_image, tmp_app_paths
):
    """label_btn is enabled when the image has an unidentified face."""
    session_factory = _make_session_factory(tmp_app_paths)

    with session_factory() as session:
        img = DBImage(file_path=str(sample_image), file_size=100)
        session.add(img)
        session.flush()
        face = Face(
            image_id=img.id,
            bbox_x=10,
            bbox_y=10,
            bbox_w=50,
            bbox_h=50,
            detection_confidence=0.99,
            model_version="test",
            state=FaceState.UNIDENTIFIED,
        )
        session.add(face)
        session.commit()
        img_id = img.id

    widget = ThumbnailWidget(
        img_id, str(sample_image), tmp_app_paths.thumbs_dir / "t.jpg", session_factory
    )
    qtbot.addWidget(widget)
    widget.show()

    widget._overlay.show()
    QtWidgets.QApplication.processEvents()

    assert widget._overlay.label_btn.isEnabled()


def test_navigate_to_labelling_from_overlay(qtbot, sample_image, tmp_app_paths):
    """Clicking label_btn causes the grid to emit navigate_to_labelling."""
    session_factory = _make_session_factory(tmp_app_paths)

    with session_factory() as session:
        img = DBImage(file_path=str(sample_image), file_size=100)
        session.add(img)
        session.flush()
        face = Face(
            image_id=img.id,
            bbox_x=10,
            bbox_y=10,
            bbox_w=50,
            bbox_h=50,
            detection_confidence=0.99,
            model_version="test",
            state=FaceState.UNIDENTIFIED,
        )
        session.add(face)
        session.commit()
        img_id = img.id

    grid = ThumbnailGrid(session_factory, VectorStore(), tmp_app_paths)
    qtbot.addWidget(grid)

    grid.add_thumbnail(img_id, str(sample_image), tmp_app_paths.thumbs_dir / "t.jpg")
    thumb = grid.thumbnails[0]
    thumb._overlay.show()

    received: list[int] = []
    grid.navigate_to_labelling.connect(received.append)

    with qtbot.waitSignal(grid.navigate_to_labelling):
        qtbot.mouseClick(thumb._overlay.label_btn, QtCore.Qt.MouseButton.LeftButton)

    assert received == [img_id]


def test_thumbnail_grid_init(
    qtbot, mock_session_factory, mock_vector_store, tmp_app_paths
):
    grid = ThumbnailGrid(mock_session_factory, mock_vector_store, tmp_app_paths)
    qtbot.addWidget(grid)
    assert len(grid.thumbnails) == 0


def test_thumbnail_grid_add_thumbnail(
    qtbot,
    sample_image,
    tmp_path,
    mock_session_factory,
    mock_vector_store,
    tmp_app_paths,
):
    grid = ThumbnailGrid(mock_session_factory, mock_vector_store, tmp_app_paths)
    qtbot.addWidget(grid)

    thumb_path = tmp_path / "thumb_grid.jpg"
    grid.add_thumbnail(1, str(sample_image), thumb_path)

    assert len(grid.thumbnails) == 1
    # Check if the widget was added to layout
    assert grid.grid_layout.count() == 1


def test_thumbnail_grid_clear(
    qtbot,
    sample_image,
    tmp_path,
    mock_session_factory,
    mock_vector_store,
    tmp_app_paths,
):
    grid = ThumbnailGrid(mock_session_factory, mock_vector_store, tmp_app_paths)
    qtbot.addWidget(grid)

    grid.add_thumbnail(1, str(sample_image), tmp_path / "t1.jpg")
    grid.add_thumbnail(2, str(sample_image), tmp_path / "t2.jpg")

    assert len(grid.thumbnails) == 2
    assert grid.grid_layout.count() == 2

    grid.clear()

    assert len(grid.thumbnails) == 0
    assert grid.grid_layout.count() == 0


def test_thumbnail_grid_resize(
    qtbot,
    sample_image,
    tmp_path,
    mock_session_factory,
    mock_vector_store,
    tmp_app_paths,
):
    grid = ThumbnailGrid(mock_session_factory, mock_vector_store, tmp_app_paths)
    grid.show()  # Need to show to have a valid width
    qtbot.addWidget(grid)

    # Force a width that should result in a specific number of columns
    # 170 * 2 + 20 = 360
    grid.resize(400, 600)
    assert grid.cols == 2

    grid.add_thumbnail(1, str(sample_image), tmp_path / "t1.jpg")
    grid.add_thumbnail(2, str(sample_image), tmp_path / "t2.jpg")
    grid.add_thumbnail(3, str(sample_image), tmp_path / "t3.jpg")

    # Check positions
    pos0 = grid.grid_layout.getItemPosition(
        grid.grid_layout.indexOf(grid.thumbnails[0])
    )
    assert isinstance(pos0, tuple)
    assert pos0[:2] == (0, 0)
    pos1 = grid.grid_layout.getItemPosition(
        grid.grid_layout.indexOf(grid.thumbnails[1])
    )
    assert isinstance(pos1, tuple)
    assert pos1[:2] == (0, 1)
    pos2 = grid.grid_layout.getItemPosition(
        grid.grid_layout.indexOf(grid.thumbnails[2])
    )
    assert isinstance(pos2, tuple)
    assert pos2[:2] == (1, 0)

    # Resize to 3 columns: 170 * 3 + 20 = 530
    grid.resize(600, 600)
    assert grid.cols == 3

    pos0 = grid.grid_layout.getItemPosition(
        grid.grid_layout.indexOf(grid.thumbnails[0])
    )
    assert isinstance(pos0, tuple)
    assert pos0[:2] == (0, 0)
    pos1 = grid.grid_layout.getItemPosition(
        grid.grid_layout.indexOf(grid.thumbnails[1])
    )
    assert isinstance(pos1, tuple)
    assert pos1[:2] == (0, 1)
    pos2 = grid.grid_layout.getItemPosition(
        grid.grid_layout.indexOf(grid.thumbnails[2])
    )
    assert isinstance(pos2, tuple)
    assert pos2[:2] == (0, 2)


# --- set_results / infinite scroll tests ---


def test_set_results_loads_first_page_only(
    qtbot,
    sample_image,
    tmp_path,
    mock_session_factory,
    mock_vector_store,
    tmp_app_paths,
):
    grid = ThumbnailGrid(mock_session_factory, mock_vector_store, tmp_app_paths)
    qtbot.addWidget(grid)

    results = [
        SearchResult(i, str(sample_image), tmp_path / f"t{i}.jpg") for i in range(100)
    ]
    grid.set_results(results)

    # Only the first page is loaded synchronously; deferred fill not yet triggered
    assert len(grid.thumbnails) == PAGE_SIZE
    assert grid._loaded_count == PAGE_SIZE


def test_set_results_fewer_than_page_size(
    qtbot,
    sample_image,
    tmp_path,
    mock_session_factory,
    mock_vector_store,
    tmp_app_paths,
):
    grid = ThumbnailGrid(mock_session_factory, mock_vector_store, tmp_app_paths)
    qtbot.addWidget(grid)

    results = [
        SearchResult(i, str(sample_image), tmp_path / f"t{i}.jpg") for i in range(10)
    ]
    grid.set_results(results)

    assert len(grid.thumbnails) == 10
    assert grid._hint_label.isHidden()


def test_set_results_empty(
    qtbot, mock_session_factory, mock_vector_store, tmp_app_paths
):
    grid = ThumbnailGrid(mock_session_factory, mock_vector_store, tmp_app_paths)
    qtbot.addWidget(grid)

    grid.set_results([])

    assert len(grid.thumbnails) == 0
    assert grid._hint_label.isHidden()


def test_set_results_emits_results_changed(
    qtbot,
    sample_image,
    tmp_path,
    mock_session_factory,
    mock_vector_store,
    tmp_app_paths,
):
    grid = ThumbnailGrid(mock_session_factory, mock_vector_store, tmp_app_paths)
    qtbot.addWidget(grid)

    results = [
        SearchResult(i, str(sample_image), tmp_path / f"t{i}.jpg") for i in range(50)
    ]

    received = []
    grid.results_changed.connect(lambda n: received.append(n))
    grid.set_results(results)

    assert received == [50]


def test_page_loaded_signal(
    qtbot,
    sample_image,
    tmp_path,
    mock_session_factory,
    mock_vector_store,
    tmp_app_paths,
):
    grid = ThumbnailGrid(mock_session_factory, mock_vector_store, tmp_app_paths)
    qtbot.addWidget(grid)

    results = [
        SearchResult(i, str(sample_image), tmp_path / f"t{i}.jpg") for i in range(100)
    ]

    signals: list[tuple[int, int]] = []
    grid.page_loaded.connect(lambda loaded, total: signals.append((loaded, total)))
    grid.set_results(results)

    assert len(signals) >= 1
    assert signals[0] == (PAGE_SIZE, 100)


def test_set_results_resets_previous(
    qtbot,
    sample_image,
    tmp_path,
    mock_session_factory,
    mock_vector_store,
    tmp_app_paths,
):
    grid = ThumbnailGrid(mock_session_factory, mock_vector_store, tmp_app_paths)
    qtbot.addWidget(grid)

    results1 = [
        SearchResult(i, str(sample_image), tmp_path / f"t{i}.jpg") for i in range(100)
    ]
    grid.set_results(results1)
    assert len(grid.thumbnails) == PAGE_SIZE

    results2 = [
        SearchResult(i, str(sample_image), tmp_path / f"s{i}.jpg") for i in range(100)
    ]
    grid.set_results(results2)

    assert len(grid.thumbnails) == PAGE_SIZE
    assert grid._loaded_count == PAGE_SIZE


def test_hint_label_shown_when_more_remain(
    qtbot,
    sample_image,
    tmp_path,
    mock_session_factory,
    mock_vector_store,
    tmp_app_paths,
):
    grid = ThumbnailGrid(mock_session_factory, mock_vector_store, tmp_app_paths)
    qtbot.addWidget(grid)

    results = [
        SearchResult(i, str(sample_image), tmp_path / f"t{i}.jpg") for i in range(100)
    ]
    grid.set_results(results)

    assert not grid._hint_label.isHidden()
    # 100 - PAGE_SIZE (30) = 70 remaining
    assert str(100 - PAGE_SIZE) in grid._hint_label.text()
    assert "remaining" in grid._hint_label.text()


def test_hint_label_hidden_when_all_loaded(
    qtbot,
    sample_image,
    tmp_path,
    mock_session_factory,
    mock_vector_store,
    tmp_app_paths,
):
    grid = ThumbnailGrid(mock_session_factory, mock_vector_store, tmp_app_paths)
    qtbot.addWidget(grid)

    results = [
        SearchResult(i, str(sample_image), tmp_path / f"t{i}.jpg") for i in range(10)
    ]
    grid.set_results(results)

    assert grid._hint_label.isHidden()


def test_scroll_triggers_load_more(
    qtbot,
    sample_image,
    tmp_path,
    mock_session_factory,
    mock_vector_store,
    tmp_app_paths,
):
    grid = ThumbnailGrid(mock_session_factory, mock_vector_store, tmp_app_paths)
    qtbot.addWidget(grid)

    results = [
        SearchResult(i, str(sample_image), tmp_path / f"t{i}.jpg") for i in range(100)
    ]
    grid.set_results(results)

    initial_count = grid._loaded_count
    assert initial_count == PAGE_SIZE

    # Directly invoke scroll handler with a value that satisfies the condition
    grid._on_scroll_changed(grid.scroll_area.verticalScrollBar().maximum())

    assert grid._loaded_count == initial_count + PAGE_SIZE


# --- _has_unidentified_faces tests ---


def test_has_unidentified_faces_true(tmp_app_paths):
    session_factory = _make_session_factory(tmp_app_paths)
    with session_factory() as session:
        img = DBImage(file_path=str(tmp_app_paths.thumbs_dir / "img.jpg"), file_size=1)
        session.add(img)
        session.flush()
        face = Face(
            image_id=img.id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=10,
            bbox_h=10,
            detection_confidence=0.9,
            model_version="v1",
            state=FaceState.UNIDENTIFIED,
        )
        session.add(face)
        session.commit()
        img_id = img.id
    assert _has_unidentified_faces(session_factory, img_id)


def test_has_unidentified_faces_false_no_faces(tmp_app_paths):
    session_factory = _make_session_factory(tmp_app_paths)
    with session_factory() as session:
        img = DBImage(file_path=str(tmp_app_paths.thumbs_dir / "img.jpg"), file_size=1)
        session.add(img)
        session.commit()
        img_id = img.id
    assert not _has_unidentified_faces(session_factory, img_id)


def test_has_unidentified_faces_false_identified(tmp_app_paths):
    session_factory = _make_session_factory(tmp_app_paths)
    with session_factory() as session:
        img = DBImage(file_path=str(tmp_app_paths.thumbs_dir / "img.jpg"), file_size=1)
        session.add(img)
        session.flush()
        face = Face(
            image_id=img.id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=10,
            bbox_h=10,
            detection_confidence=0.9,
            model_version="v1",
            state=FaceState.IDENTIFIED,
        )
        session.add(face)
        session.commit()
        img_id = img.id
    assert not _has_unidentified_faces(session_factory, img_id)


# --- _get_scaled_size edge case tests ---


def test_get_scaled_size_unreadable_file(tmp_path):
    """Non-image file → canRead() returns False → invalid QSize."""
    txt = tmp_path / "not_an_image.txt"
    txt.write_text("hello")
    size = _get_scaled_size(txt)
    assert not size.isValid()


def test_get_scaled_size_zero_dimension(monkeypatch):
    """QImageReader returns valid but zero-dimension size → invalid QSize returned, no ZeroDivisionError."""  # noqa: E501

    from PySide6 import QtCore, QtGui

    mock_reader = MagicMock(spec=QtGui.QImageReader)
    mock_reader.canRead.return_value = True
    mock_reader.size.return_value = QtCore.QSize(0, 0)
    monkeypatch.setattr(
        "photoaident.ui.widgets.thumbnail_grid.QtGui.QImageReader",
        lambda _path: mock_reader,
    )
    size = _get_scaled_size(Path("/fake/image.jpg"))
    assert not size.isValid()


# --- _read_pixmap edge case tests ---


def test_read_pixmap_null_image(tmp_path):
    """Corrupted JPEG → image.isNull() → null QPixmap."""
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not a jpeg")
    valid_size = QtCore.QSize(100, 100)
    pixmap = _read_pixmap(bad, valid_size)
    assert pixmap.isNull()


def test_read_pixmap_invalid_size(sample_image):
    """Invalid scaled_size → early return with null QPixmap."""
    pixmap = _read_pixmap(sample_image, QtCore.QSize())
    assert pixmap.isNull()


# --- overlay showEvent with session_factory tests ---


def test_label_button_disabled_with_session_factory_but_no_faces(
    qtbot, sample_image, tmp_app_paths
):
    """label_btn is disabled when session_factory is set but image has no faces."""
    session_factory = _make_session_factory(tmp_app_paths)
    with session_factory() as session:
        img = DBImage(file_path=str(sample_image), file_size=100)
        session.add(img)
        session.commit()
        img_id = img.id

    widget = ThumbnailWidget(
        img_id, str(sample_image), tmp_app_paths.thumbs_dir / "t.jpg", session_factory
    )
    qtbot.addWidget(widget)
    widget.show()

    widget._overlay.show()
    QtWidgets.QApplication.processEvents()

    assert not widget._overlay.label_btn.isEnabled()


# --- image detail dialog tests ---


def _make_session_factory(paths: AppPaths):
    apply_migrations(f"sqlite:///{paths.db_path}")
    engine = get_engine(str(paths.db_path))
    return get_session_factory(engine)


def test_on_image_selected_opens_dialog(qtbot, tmp_app_paths, mock_vector_store):
    """Clicking a thumbnail opens ImageDetailDialog."""
    session_factory = _make_session_factory(tmp_app_paths)

    with session_factory() as session:
        img = DBImage(
            file_path=str(tmp_app_paths.thumbs_dir / "img.jpg"), file_size=100
        )
        session.add(img)
        session.commit()
        img_id = img.id

    grid = ThumbnailGrid(session_factory, mock_vector_store, tmp_app_paths)
    qtbot.addWidget(grid)

    with patch("photoaident.ui.widgets.thumbnail_grid.ImageDetailDialog") as MockDlg:
        MockDlg.return_value.exec.return_value = None
        grid._on_image_selected(img_id)
        MockDlg.assert_called_once()
        MockDlg.return_value.exec.assert_called_once()


def test_on_image_selected_nonexistent_id_skips_dialog(
    qtbot, tmp_app_paths, mock_vector_store
):
    """Selecting a non-existent image id does not open the dialog."""
    session_factory = _make_session_factory(tmp_app_paths)
    grid = ThumbnailGrid(session_factory, mock_vector_store, tmp_app_paths)
    qtbot.addWidget(grid)

    with patch("photoaident.ui.widgets.thumbnail_grid.ImageDetailDialog") as MockDlg:
        grid._on_image_selected(99999)
        MockDlg.assert_not_called()


def test_on_image_selected_without_image_in_db_is_noop(
    qtbot, mock_session_factory, mock_vector_store, tmp_app_paths
):
    """If image is not found in DB, clicking a thumbnail does nothing."""
    session = mock_session_factory.return_value.__enter__.return_value
    (
        session.execute.return_value.unique.return_value.scalar_one_or_none.return_value
    ) = None
    grid = ThumbnailGrid(mock_session_factory, mock_vector_store, tmp_app_paths)
    qtbot.addWidget(grid)

    with patch("photoaident.ui.widgets.thumbnail_grid.ImageDetailDialog") as MockDlg:
        grid._on_image_selected(1)
        MockDlg.assert_not_called()


def test_navigate_to_labelling_signal_forwarded(
    qtbot, tmp_app_paths, mock_vector_store
):
    """navigate_to_labelling from ImageDetailDialog is forwarded by the grid."""
    session_factory = _make_session_factory(tmp_app_paths)

    with session_factory() as session:
        img = DBImage(
            file_path=str(tmp_app_paths.thumbs_dir / "nav.jpg"), file_size=100
        )
        session.add(img)
        session.commit()
        img_id = img.id

    grid = ThumbnailGrid(session_factory, mock_vector_store, tmp_app_paths)
    qtbot.addWidget(grid)

    received: list[int] = []
    grid.navigate_to_labelling.connect(received.append)

    with patch("photoaident.ui.widgets.thumbnail_grid.ImageDetailDialog") as MockDlg:
        MockDlg.return_value.exec.return_value = None
        # Capture the slot connected to navigate_to_labelling
        captured_slots: list[Callable] = []

        def capture_connect(slot: Callable) -> None:
            captured_slots.append(slot)

        MockDlg.return_value.navigate_to_labelling.connect.side_effect = capture_connect
        grid._on_image_selected(img_id)

        assert (
            len(captured_slots) == 1
        ), "Expected exactly one slot connected to navigate_to_labelling"
        captured_slots[0](img_id)

    assert received == [img_id]


# ---------------------------------------------------------------------------
# Sort dropdown tests
# ---------------------------------------------------------------------------


def test_sort_combo_default_is_relevance_desc(
    qtbot, mock_session_factory, mock_vector_store, tmp_app_paths
):
    """Default sort is RELEVANCE_DESC and combo is enabled."""
    grid = ThumbnailGrid(mock_session_factory, mock_vector_store, tmp_app_paths)
    qtbot.addWidget(grid)

    assert grid._current_sort == SortOrder.RELEVANCE_DESC
    assert grid._sort_combo.isEnabled()


def test_set_sort_locked_disables_combo(
    qtbot, mock_session_factory, mock_vector_store, tmp_app_paths
):
    """set_sort_locked sets the given order and disables the combo."""
    grid = ThumbnailGrid(mock_session_factory, mock_vector_store, tmp_app_paths)
    qtbot.addWidget(grid)

    grid.set_sort_locked(SortOrder.FILENAME_ASC)

    assert not grid._sort_combo.isEnabled()
    assert grid._sort_entries[grid._sort_combo.currentIndex()] == SortOrder.FILENAME_ASC


def test_set_relevance_available_false_disables_relevance_items(
    qtbot, mock_session_factory, mock_vector_store, tmp_app_paths
):
    """set_relevance_available(False) disables the two relevance combo items."""
    from PySide6.QtGui import QStandardItemModel

    grid = ThumbnailGrid(mock_session_factory, mock_vector_store, tmp_app_paths)
    qtbot.addWidget(grid)

    grid.set_relevance_available(False)

    model = grid._sort_combo.model()
    assert isinstance(model, QStandardItemModel)
    for i, order in enumerate(grid._sort_entries):
        item = model.item(i)
        assert item is not None
        enabled = bool(item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled)
        if order in (SortOrder.RELEVANCE_DESC, SortOrder.RELEVANCE_ASC):
            assert not enabled, f"Expected {order} to be disabled"
        else:
            assert enabled, f"Expected {order} to be enabled"


def test_set_relevance_available_true_reenables_relevance_items(
    qtbot, mock_session_factory, mock_vector_store, tmp_app_paths
):
    """set_relevance_available(True) re-enables relevance items after disabling."""
    from PySide6.QtGui import QStandardItemModel

    grid = ThumbnailGrid(mock_session_factory, mock_vector_store, tmp_app_paths)
    qtbot.addWidget(grid)

    grid.set_relevance_available(False)
    grid.set_relevance_available(True)

    model = grid._sort_combo.model()
    assert isinstance(model, QStandardItemModel)
    for i in range(grid._sort_combo.count()):
        item = model.item(i)
        assert item is not None
        assert bool(item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled)


def test_set_relevance_unavailable_switches_from_relevance_sort(
    qtbot, mock_session_factory, mock_vector_store, tmp_app_paths
):
    """When relevance sort is active and becomes unavailable, falls back to TAKEN_AT_DESC."""  # noqa: E501
    grid = ThumbnailGrid(mock_session_factory, mock_vector_store, tmp_app_paths)
    qtbot.addWidget(grid)

    assert grid._current_sort == SortOrder.RELEVANCE_DESC
    grid.set_relevance_available(False)

    assert grid._current_sort == SortOrder.TAKEN_AT_DESC
    assert (
        grid._sort_entries[grid._sort_combo.currentIndex()] == SortOrder.TAKEN_AT_DESC
    )


def test_sort_order_applied_on_set_results(
    qtbot,
    tmp_path,
    mock_session_factory,
    mock_vector_store,
    tmp_app_paths,
):
    """Results are sorted by _current_sort when set_results() is called."""
    grid = ThumbnailGrid(mock_session_factory, mock_vector_store, tmp_app_paths)
    qtbot.addWidget(grid)

    # Switch to FILENAME_ASC before setting results
    grid._sort_combo.setCurrentIndex(grid._sort_entries.index(SortOrder.FILENAME_ASC))

    results = [
        SearchResult(1, "/z_img.jpg", tmp_path / "z.jpg"),
        SearchResult(2, "/a_img.jpg", tmp_path / "a.jpg"),
        SearchResult(3, "/m_img.jpg", tmp_path / "m.jpg"),
    ]
    grid.set_results(results)

    # After sorting by FILENAME_ASC: /a_img.jpg, /m_img.jpg, /z_img.jpg → ids 2, 3, 1
    assert [r.image_id for r in grid._all_results] == [2, 3, 1]


def test_changing_sort_combo_reorders_displayed_thumbnails(
    qtbot,
    tmp_path,
    mock_session_factory,
    mock_vector_store,
    tmp_app_paths,
):
    """Changing the combo re-sorts and reloads the grid."""
    grid = ThumbnailGrid(mock_session_factory, mock_vector_store, tmp_app_paths)
    qtbot.addWidget(grid)

    results = [
        SearchResult(1, "/z_img.jpg", tmp_path / "z.jpg"),
        SearchResult(2, "/a_img.jpg", tmp_path / "a.jpg"),
    ]
    grid.set_results(results)

    # Change to FILENAME_ASC
    grid._sort_combo.setCurrentIndex(grid._sort_entries.index(SortOrder.FILENAME_ASC))

    assert grid._all_results[0].image_id == 2  # /a_img.jpg first
