"""Tests for LibraryPage: persistent right-column person filter, image selection."""

from unittest.mock import MagicMock, patch

import pytest
from PySide6 import QtWidgets, QtGui

from photoaident.core.date_range import DateRange
from photoaident.core.geo import GpsBoundingBox
from photoaident.db.cluster_means import recompute_cluster_mean
from photoaident.db.database import (
    EmbeddingCluster,
    Face,
    FaceState,
    Image,
    Person,
    get_engine,
    get_session_factory,
)
from photoaident.db.migrate import apply_migrations
from photoaident.ui.pages.library import LibraryPage
from tests.search_helpers import _unit_emb


@pytest.fixture
def session_factory(tmp_path):
    db_path = tmp_path / "lib_test.db"
    engine = get_engine(str(db_path))
    apply_migrations(f"sqlite:///{db_path}")
    return get_session_factory(engine)


def _add_image(
    session_factory, file_path="/img.jpg", file_hash="abc123", file_size=1000
):
    with session_factory() as session:
        img = Image(file_path=file_path, file_size=file_size, file_hash=file_hash)
        session.add(img)
        session.commit()
        return img.id


def _add_person_with_face(session_factory, vector_store, name, file_path):
    """Add person+cluster, image, and identified face. Return (person_id, image_id)."""
    with session_factory() as session:
        p = Person(name=name)
        session.add(p)
        session.flush()
        c = EmbeddingCluster(person_id=p.id)
        session.add(c)
        session.commit()
        person_id = p.id
        cluster_id = c.id

    emb = _unit_emb(0)

    with session_factory() as session:
        img = Image(file_path=file_path, file_size=100, file_hash=file_path.strip("/"))
        session.add(img)
        session.flush()
        face = Face(
            image_id=img.id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=50,
            bbox_h=50,
            detection_confidence=0.9,
            person_id=person_id,
            cluster_id=cluster_id,
            state=FaceState.IDENTIFIED,
            model_version="test",
        )
        session.add(face)
        session.flush()
        vector_store.add(face.id, emb)
        session.commit()
        image_id = img.id
    recompute_cluster_mean(cluster_id, session_factory, vector_store)
    return person_id, image_id


# --- Filter panel always visible ---


def test_filter_panel_always_visible(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)
    assert not page.filter_panel.isHidden()


# --- Person list structure ---


def test_person_list_items_are_selectable(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    with session_factory() as session:
        session.add(Person(name="Bob"))
        session.commit()

    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)

    assert page.person_list_widget.count() == 1
    assert (
        page.person_list_widget.selectionMode()
        == QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
    )


# --- Search filter ---


def test_search_filter_hides_non_matching_items(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    with session_factory() as session:
        session.add(Person(name="Alice"))
        session.add(Person(name="Bob"))
        session.commit()

    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)

    page.search_edit.setText("ali")

    bob_item = None
    for i in range(page.person_list_widget.count()):
        item = page.person_list_widget.item(i)
        if item and item.text() == "Bob":
            bob_item = item
    assert bob_item is not None
    assert bob_item.isHidden()


def test_search_filter_case_insensitive(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    with session_factory() as session:
        session.add(Person(name="Alice"))
        session.commit()

    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)

    page.search_edit.setText("ALICE")

    item = page.person_list_widget.item(0)
    assert item is not None
    assert not item.isHidden()


# --- load_images: no selection → empty grid ---


def test_no_person_selected_shows_empty_grid(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    _add_image(session_factory, "/img1.jpg", file_hash="h1")
    _add_image(session_factory, "/img2.jpg", file_hash="h2")

    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)

    assert len(page.grid.thumbnails) == 0


def test_no_person_checked_shows_all_images(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    _add_image(session_factory, "/img1.jpg", file_hash="h1")
    _add_image(session_factory, "/img2.jpg", file_hash="h2")

    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)

    assert len(page.grid.thumbnails) == 0


def test_no_person_checked_with_images_in_db(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    for i in range(5):
        _add_image(session_factory, f"/img{i}.jpg", file_hash=f"hash{i}")

    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)

    assert len(page.grid.thumbnails) == 0


def test_image_without_hash_uses_unknown_thumb(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    with session_factory() as session:
        img = Image(file_path="/no_hash.jpg", file_size=500, file_hash=None)
        session.add(img)
        session.commit()

    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)

    # No person selected → empty grid
    assert len(page.grid.thumbnails) == 0


def test_image_with_metadata_appears_in_grid(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """An image containing a selected person appears in the grid."""
    _add_person_with_face(session_factory, vector_store, "Meta", "/meta.jpg")

    page = LibraryPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)

    item = page.person_list_widget.item(0)
    assert item is not None
    item.setSelected(True)

    assert len(page.grid.thumbnails) == 1
    assert page.grid.thumbnails[0].thumb_path.name == "meta.jpg.jpg"


# --- load_images: person filter ---


def test_select_one_person_shows_matched_images(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    _, _ = _add_person_with_face(session_factory, vector_store, "Eve", "/eve.jpg")

    page = LibraryPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)

    item = page.person_list_widget.item(0)
    assert item is not None
    item.setSelected(True)

    assert len(page.grid.thumbnails) >= 1


def test_select_person_with_no_faces_shows_empty(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    with session_factory() as session:
        p = Person(name="Nobody")
        session.add(p)
        session.flush()
        session.add(EmbeddingCluster(person_id=p.id))
        session.commit()

    page = LibraryPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)

    item = page.person_list_widget.item(0)
    assert item is not None
    item.setSelected(True)

    assert len(page.grid.thumbnails) == 0


def test_select_multiple_persons_calls_search(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """UI test: selecting multiple persons triggers search_images with correct IDs."""
    with session_factory() as session:
        p1 = Person(name="Alice")
        p2 = Person(name="Bob")
        session.add_all([p1, p2])
        session.commit()
        p1_id, p2_id = p1.id, p2.id

    page = LibraryPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)

    with patch("photoaident.ui.pages.library.search_images") as mock_search:
        mock_search.return_value = []
        # Select both
        for i in range(page.person_list_widget.count()):
            page.person_list_widget.item(i).setSelected(True)

        # load_images is called by itemSelectionChanged signal
        mock_search.assert_called()
        # Verify it was called with both IDs
        _, kwargs = mock_search.call_args
        assert set(kwargs["person_ids"]) == {p1_id, p2_id}


def test_select_person_with_no_faces_shows_empty_grid(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    _add_image(session_factory, "/img1.jpg", file_hash="h1")
    _add_image(session_factory, "/img2.jpg", file_hash="h2")
    with session_factory() as session:
        p = Person(name="Alice")
        session.add(p)
        session.commit()

    # We now REQUIRE vector_store in LibraryPage and search_images.
    # If we pass a vector_store, it works.
    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)

    item = page.person_list_widget.item(0)
    assert item is not None
    item.setSelected(True)

    # Since there are no faces for Alice, it should show 0 images, NOT fall back to all.
    assert len(page.grid.thumbnails) == 0


def test_gps_filter_calls_search(qtbot, session_factory, tmp_app_paths, vector_store):
    """UI test: setting GPS filter triggers search_images with correct bbox."""
    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)

    bbox = GpsBoundingBox(south=52.0, west=13.0, north=53.0, east=14.0)
    with patch("photoaident.ui.pages.library.search_images") as mock_search:
        mock_search.return_value = []
        page._gps_bbox = bbox
        page.load_images()

        mock_search.assert_called_once()
        assert mock_search.call_args[1]["gps_bbox"] == bbox

    # Clear filter
    page._on_location_cleared()
    assert page._gps_bbox is None


def test_show_event_populates_list(qtbot, session_factory, tmp_app_paths, vector_store):
    with session_factory() as session:
        session.add(Person(name="Alice"))
        session.commit()

    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)

    # Clear the list to ensure showEvent repopulates it
    page.person_list_widget.clear()
    assert page.person_list_widget.count() == 0

    # Trigger showEvent
    event = QtGui.QShowEvent()
    page.showEvent(event)
    assert page.person_list_widget.count() == 1
    assert page.person_list_widget.item(0).text() == "Alice"


def test_populate_person_list_preserves_selection(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    with session_factory() as session:
        p1 = Person(name="Alice")
        session.add(p1)
        session.commit()
        p1_id = p1.id

    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)

    item = page.person_list_widget.item(0)
    item.setSelected(True)
    assert p1_id in page._selected_person_ids()

    # Repopulate
    page._populate_person_list()

    new_item = page.person_list_widget.item(0)
    assert new_item.isSelected()
    assert p1_id in page._selected_person_ids()


def test_apply_search_filter_handles_none_item(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)

    # Mock item() to return None at some index
    with patch.object(page.person_list_widget, "item", side_effect=[None]):
        with patch.object(page.person_list_widget, "count", return_value=1):
            # This should not raise an AttributeError when calling item.text()
            page._apply_search_filter("test")


def test_open_map_dialog_accepted(qtbot, session_factory, tmp_app_paths, vector_store):
    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)

    bbox = GpsBoundingBox(south=10, west=10, north=20, east=20)

    with patch("photoaident.ui.pages.library.MapLocationDialog") as MockDialog:
        mock_dialog = MockDialog.return_value
        mock_dialog.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
        mock_dialog.selected_bbox.return_value = bbox

        page._open_map_dialog()

        assert page._gps_bbox == bbox
        assert not page.clear_location_btn.isHidden()


def test_load_images_empty_per_person_scores(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    # This happens if find_images_by_person returns nothing
    with session_factory() as session:
        p1 = Person(name="Alice")
        session.add(p1)
        session.commit()

    page = LibraryPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)

    # Select Alice
    item = page.person_list_widget.item(0)
    item.setSelected(True)

    # Mock search_images to return empty
    with patch("photoaident.ui.pages.library.search_images", return_value=[]):
        page.load_images()
        assert len(page.grid.thumbnails) == 0


def test_navigate_to_labelling(qtbot, session_factory, tmp_app_paths, vector_store):
    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)

    mock_main_window = MagicMock()
    # We need to mock self.window() to return our mock_main_window
    with patch.object(page, "window", return_value=mock_main_window):
        # We also need to make sure isinstance(mock_main_window, MainWindow) is true
        # But MainWindow is imported locally in the method.
        # Actually, if we mock the class itself in the module where it's imported:
        with patch("photoaident.app.MainWindow", new=MagicMock) as MockMW:
            # Re-patching window to return an instance of MockMW
            mw_instance = MockMW()
            with patch.object(page, "window", return_value=mw_instance):
                page._on_navigate_to_labelling(123)
                mw_instance.go_to_labelling.assert_called_once_with(123)


# --- Date filter tests ---


def test_time_filter_button_present(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Time filter button exists in the filter panel."""
    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)
    assert hasattr(page, "date_filter_btn")
    assert not page.date_filter_btn.isHidden()


def test_clear_time_button_initially_hidden(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Clear Time button is hidden until a date range is active."""
    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)
    assert page.clear_time_btn.isHidden()


def test_open_date_dialog_accepted(qtbot, session_factory, tmp_app_paths, vector_store):
    """Accepting DateFilterDialog sets _date_range and shows Clear Time button."""
    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)

    date_range = DateRange(start_year=2020, end_year=2023)

    with patch("photoaident.ui.pages.library.DateFilterDialog") as MockDialog:
        mock_dialog = MockDialog.return_value
        mock_dialog.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
        mock_dialog.selected_range.return_value = date_range

        page._open_date_dialog()

        assert page._date_range == date_range
        assert not page.clear_time_btn.isHidden()
        assert page.date_filter_btn.isChecked()


def test_on_time_cleared_resets_filter(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_on_time_cleared() removes date range and updates button state."""
    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)

    page._date_range = DateRange(start_year=2020)
    page._update_time_button()
    assert not page.clear_time_btn.isHidden()

    page._on_time_cleared()

    assert page._date_range is None
    assert page.clear_time_btn.isHidden()
    assert not page.date_filter_btn.isChecked()


def test_has_filters_with_date_range(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_has_filters() returns True when only a date range is set."""
    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)

    assert not page._has_filters()
    page._date_range = DateRange(start_year=2020)
    assert page._has_filters()


def test_date_filter_calls_search_with_date_range(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """load_images passes date_range to search_images."""
    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)

    date_range = DateRange(start_year=2020, end_year=2023)
    page._date_range = date_range

    with patch("photoaident.ui.pages.library.search_images") as mock_search:
        mock_search.return_value = []
        page.load_images()

        mock_search.assert_called_once()
        assert mock_search.call_args[1]["date_range"] == date_range


# --- Keyword/filename search tests ---


def test_keyword_search_counts_as_filter(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """A non-empty keyword search text activates the filter."""
    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)

    assert not page._has_filters()
    page.filepath_search_edit.setText("vacation")
    assert page._has_filters()

    page.filepath_search_edit.clear()
    assert not page._has_filters()


def test_keyword_search_passes_query_to_search(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """load_images passes filename_query to search_images."""
    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)

    page.filepath_search_edit.setText("New York")
    page._keyword_debounce_timer.stop()  # prevent delayed fire inside patch context

    with patch("photoaident.ui.pages.library.search_images") as mock_search:
        mock_search.return_value = []
        page.load_images()

        mock_search.assert_called_once()
        assert mock_search.call_args[1]["filename_query"] == "New York"


def test_keyword_search_empty_passes_none(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """load_images passes filename_query=None when keyword field is empty."""
    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)

    # Need at least one other filter active so load_images actually calls search
    page._date_range = DateRange(start_year=2020)
    page.filepath_search_edit.clear()
    page._keyword_debounce_timer.stop()  # prevent delayed fire inside patch context

    with patch("photoaident.ui.pages.library.search_images") as mock_search:
        mock_search.return_value = []
        page.load_images()

        mock_search.assert_called_once()
        assert mock_search.call_args[1]["filename_query"] is None


def test_keyword_search_debounce_timer_configured(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """The debounce timer exists, is single-shot, and has a 300 ms interval."""
    page = LibraryPage(session_factory, tmp_app_paths, vector_store=vector_store)
    qtbot.addWidget(page)

    assert hasattr(page, "_keyword_debounce_timer")
    assert page._keyword_debounce_timer.isSingleShot()
    assert page._keyword_debounce_timer.interval() == 300
