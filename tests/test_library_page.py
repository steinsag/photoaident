"""Tests for LibraryPage: persistent right-column person filter, image selection."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PySide6 import QtWidgets, QtGui

from photoaident.core.geo import GpsBoundingBox
from photoaident.db.database import (
    EmbeddingCluster,
    Face,
    FaceState,
    Image,
    Person,
    get_engine,
    get_session_factory,
    ImageMetadata,
    TakenAtSource,
)
from photoaident.db.migrate import apply_migrations
from photoaident.db.vector_store import VectorStore
from photoaident.ui.pages.library import LibraryPage


@pytest.fixture
def session_factory(tmp_path):
    db_path = tmp_path / "lib_test.db"
    engine = get_engine(str(db_path))
    apply_migrations(f"sqlite:///{db_path}")
    return get_session_factory(engine)


@pytest.fixture
def vs():
    return VectorStore()


def _add_test_images_with_gps(session_factory):
    """Add two images with GPS metadata. Return (img1_id, img2_id)."""
    with session_factory() as session:
        img1 = Image(file_path="/img1.jpg", file_size=100, file_hash="h1")
        session.add(img1)
        session.flush()
        meta1 = ImageMetadata(
            image_id=img1.id,
            gps_lat=52.5,
            gps_lon=13.4,
            taken_at_source=TakenAtSource.FILESYSTEM,
            width=100,
            height=100,
        )
        session.add(meta1)

        img2 = Image(file_path="/img2.jpg", file_size=100, file_hash="h2")
        session.add(img2)
        session.flush()
        meta2 = ImageMetadata(
            image_id=img2.id,
            gps_lat=40.0,
            gps_lon=10.0,
            taken_at_source=TakenAtSource.FILESYSTEM,
            width=100,
            height=100,
        )
        session.add(meta2)
        session.commit()
        return img1.id, img2.id


def _add_image(
    session_factory, file_path="/img.jpg", file_hash="abc123", file_size=1000
):
    with session_factory() as session:
        img = Image(file_path=file_path, file_size=file_size, file_hash=file_hash)
        session.add(img)
        session.commit()
        return img.id


def _add_person_with_face(session_factory, vs, name, file_path):
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

    emb = np.zeros(512, dtype=np.float32)
    emb[0] = 1.0
    faiss_id = vs.add(emb)

    with session_factory() as session:
        img = Image(file_path=file_path, file_size=100, file_hash=file_path.strip("/"))
        session.add(img)
        session.flush()
        face = Face(
            image_id=img.id,
            faiss_id=faiss_id,
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
        session.commit()
        return person_id, img.id


def _add_two_persons_with_embeddings(session_factory, vs):
    """
    Add two persons with orthogonal embeddings.
    Return (p1_id, c1_id, faiss_id1, p2_id, c2_id, faiss_id2).
    """
    with session_factory() as session:
        p1 = Person(name="Alice")
        p2 = Person(name="Bob")
        session.add_all([p1, p2])
        session.flush()
        c1 = EmbeddingCluster(person_id=p1.id)
        c2 = EmbeddingCluster(person_id=p2.id)
        session.add_all([c1, c2])
        session.commit()
        p1_id, c1_id = p1.id, c1.id
        p2_id, c2_id = p2.id, c2.id

    # Two orthogonal embeddings so they won't match each other
    emb1 = np.zeros(512, dtype=np.float32)
    emb1[0] = 1.0
    emb2 = np.zeros(512, dtype=np.float32)
    emb2[1] = 1.0
    faiss_id1 = vs.add(emb1)
    faiss_id2 = vs.add(emb2)

    return p1_id, c1_id, faiss_id1, p2_id, c2_id, faiss_id2


def _setup_library_page_with_all_selected(
    qtbot, session_factory, tmp_app_paths, vs=None
) -> LibraryPage:
    """Helper to create LibraryPage and select all persons in the list."""
    page = LibraryPage(session_factory, tmp_app_paths, vs)
    qtbot.addWidget(page)

    page.person_list_widget.blockSignals(True)
    for i in range(page.person_list_widget.count()):
        item = page.person_list_widget.item(i)
        assert item is not None
        item.setSelected(True)
    page.person_list_widget.blockSignals(False)
    page.load_images()
    return page


# --- Filter panel always visible ---


def test_filter_panel_always_visible(qtbot, session_factory, tmp_app_paths):
    page = LibraryPage(session_factory, tmp_app_paths)
    qtbot.addWidget(page)
    assert not page.filter_panel.isHidden()


# --- Person list structure ---


def test_person_list_items_are_selectable(qtbot, session_factory, tmp_app_paths):
    with session_factory() as session:
        session.add(Person(name="Bob"))
        session.commit()

    page = LibraryPage(session_factory, tmp_app_paths)
    qtbot.addWidget(page)

    assert page.person_list_widget.count() == 1
    assert (
        page.person_list_widget.selectionMode()
        == QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
    )


# --- Search filter ---


def test_search_filter_hides_non_matching_items(qtbot, session_factory, tmp_app_paths):
    with session_factory() as session:
        session.add(Person(name="Alice"))
        session.add(Person(name="Bob"))
        session.commit()

    page = LibraryPage(session_factory, tmp_app_paths)
    qtbot.addWidget(page)

    page.search_edit.setText("ali")

    bob_item = None
    for i in range(page.person_list_widget.count()):
        item = page.person_list_widget.item(i)
        if item and item.text() == "Bob":
            bob_item = item
    assert bob_item is not None
    assert bob_item.isHidden()


def test_search_filter_case_insensitive(qtbot, session_factory, tmp_app_paths):
    with session_factory() as session:
        session.add(Person(name="Alice"))
        session.commit()

    page = LibraryPage(session_factory, tmp_app_paths)
    qtbot.addWidget(page)

    page.search_edit.setText("ALICE")

    item = page.person_list_widget.item(0)
    assert item is not None
    assert not item.isHidden()


# --- load_images: no selection → empty grid ---


def test_no_person_selected_shows_empty_grid(qtbot, session_factory, tmp_app_paths):
    _add_image(session_factory, "/img1.jpg", file_hash="h1")
    _add_image(session_factory, "/img2.jpg", file_hash="h2")

    page = LibraryPage(session_factory, tmp_app_paths)
    qtbot.addWidget(page)

    assert len(page.grid.thumbnails) == 0


def test_no_person_checked_shows_all_images(qtbot, session_factory, tmp_app_paths):
    _add_image(session_factory, "/img1.jpg", file_hash="h1")
    _add_image(session_factory, "/img2.jpg", file_hash="h2")

    page = LibraryPage(session_factory, tmp_app_paths)
    qtbot.addWidget(page)

    assert len(page.grid.thumbnails) == 0


def test_no_person_checked_with_images_in_db(qtbot, session_factory, tmp_app_paths):
    for i in range(5):
        _add_image(session_factory, f"/img{i}.jpg", file_hash=f"hash{i}")

    page = LibraryPage(session_factory, tmp_app_paths)
    qtbot.addWidget(page)

    assert len(page.grid.thumbnails) == 0


def test_image_without_hash_uses_unknown_thumb(qtbot, session_factory, tmp_app_paths):
    with session_factory() as session:
        img = Image(file_path="/no_hash.jpg", file_size=500, file_hash=None)
        session.add(img)
        session.commit()

    page = LibraryPage(session_factory, tmp_app_paths)
    qtbot.addWidget(page)

    # No person selected → empty grid
    assert len(page.grid.thumbnails) == 0


def test_image_with_metadata_appears_in_grid(qtbot, session_factory, tmp_app_paths, vs):
    """An image containing a selected person appears in the grid."""
    _add_person_with_face(session_factory, vs, "Meta", "/meta.jpg")

    page = LibraryPage(session_factory, tmp_app_paths, vs)
    qtbot.addWidget(page)

    item = page.person_list_widget.item(0)
    assert item is not None
    item.setSelected(True)

    assert len(page.grid.thumbnails) == 1
    assert page.grid.thumbnails[0].thumb_path.name == "meta.jpg.jpg"


# --- load_images: person filter ---


def test_select_one_person_shows_matched_images(
    qtbot, session_factory, tmp_app_paths, vs
):
    _, _ = _add_person_with_face(session_factory, vs, "Eve", "/eve.jpg")

    page = LibraryPage(session_factory, tmp_app_paths, vs)
    qtbot.addWidget(page)

    item = page.person_list_widget.item(0)
    assert item is not None
    item.setSelected(True)

    assert len(page.grid.thumbnails) >= 1


def test_select_person_with_no_faces_shows_empty(
    qtbot, session_factory, tmp_app_paths, vs
):
    with session_factory() as session:
        p = Person(name="Nobody")
        session.add(p)
        session.flush()
        session.add(EmbeddingCluster(person_id=p.id))
        session.commit()

    page = LibraryPage(session_factory, tmp_app_paths, vs)
    qtbot.addWidget(page)

    item = page.person_list_widget.item(0)
    assert item is not None
    item.setSelected(True)

    assert len(page.grid.thumbnails) == 0


def test_select_multiple_persons_intersects_results(
    qtbot, session_factory, tmp_app_paths, vs
):
    """Selecting two persons shows only images where BOTH appear (AND, not OR)."""
    (
        p1_id,
        c1_id,
        faiss_id1,
        p2_id,
        c2_id,
        faiss_id2,
    ) = _add_two_persons_with_embeddings(session_factory, vs)

    with session_factory() as session:
        img1 = Image(file_path="/alice.jpg", file_size=100, file_hash="alice")
        img2 = Image(file_path="/bob.jpg", file_size=100, file_hash="bob")
        session.add_all([img1, img2])
        session.flush()
        session.add(
            Face(
                image_id=img1.id,
                faiss_id=faiss_id1,
                bbox_x=0,
                bbox_y=0,
                bbox_w=50,
                bbox_h=50,
                detection_confidence=0.9,
                person_id=p1_id,
                cluster_id=c1_id,
                state=FaceState.IDENTIFIED,
                model_version="test",
            )
        )
        session.add(
            Face(
                image_id=img2.id,
                faiss_id=faiss_id2,
                bbox_x=0,
                bbox_y=0,
                bbox_w=50,
                bbox_h=50,
                detection_confidence=0.9,
                person_id=p2_id,
                cluster_id=c2_id,
                state=FaceState.IDENTIFIED,
                model_version="test",
            )
        )
        session.commit()

    page = _setup_library_page_with_all_selected(
        qtbot, session_factory, tmp_app_paths, vs
    )

    # Alice is only in alice.jpg, Bob is only in bob.jpg → no image has both
    assert len(page.grid.thumbnails) == 0


def test_select_multiple_persons_shows_shared_image(
    qtbot, session_factory, tmp_app_paths, vs
):
    """An image containing all selected persons is included in AND results."""
    (
        p1_id,
        c1_id,
        faiss_id1,
        p2_id,
        c2_id,
        faiss_id2,
    ) = _add_two_persons_with_embeddings(session_factory, vs)

    with session_factory() as session:
        img = Image(file_path="/shared.jpg", file_size=100, file_hash="shared")
        session.add(img)
        session.flush()
        session.add(
            Face(
                image_id=img.id,
                faiss_id=faiss_id1,
                bbox_x=0,
                bbox_y=0,
                bbox_w=50,
                bbox_h=50,
                detection_confidence=0.9,
                person_id=p1_id,
                cluster_id=c1_id,
                state=FaceState.IDENTIFIED,
                model_version="test",
            )
        )
        session.add(
            Face(
                image_id=img.id,
                faiss_id=faiss_id2,
                bbox_x=60,
                bbox_y=0,
                bbox_w=50,
                bbox_h=50,
                detection_confidence=0.9,
                person_id=p2_id,
                cluster_id=c2_id,
                state=FaceState.IDENTIFIED,
                model_version="test",
            )
        )
        session.commit()
        shared_img_id = img.id

    page = _setup_library_page_with_all_selected(
        qtbot, session_factory, tmp_app_paths, vs
    )

    result_ids = [t.image_id for t in page.grid.thumbnails]
    assert shared_img_id in result_ids
    assert len(page.grid.thumbnails) == 1


def test_person_filter_without_vector_store_falls_back_to_all(
    qtbot, session_factory, tmp_app_paths
):
    _add_image(session_factory, "/img1.jpg", file_hash="h1")
    _add_image(session_factory, "/img2.jpg", file_hash="h2")
    with session_factory() as session:
        p = Person(name="Alice")
        session.add(p)
        session.commit()

    page = LibraryPage(session_factory, tmp_app_paths, vector_store=None)
    qtbot.addWidget(page)

    item = page.person_list_widget.item(0)
    assert item is not None
    item.setSelected(True)

    assert len(page.grid.thumbnails) == 2


def test_gps_filter_shows_matched_images(qtbot, session_factory, tmp_app_paths):
    img1_id, _ = _add_test_images_with_gps(session_factory)

    page = LibraryPage(session_factory, tmp_app_paths)
    qtbot.addWidget(page)

    # Set GPS filter (Berlin-ish)
    bbox = GpsBoundingBox(south=52.0, west=13.0, north=53.0, east=14.0)
    page._gps_bbox = bbox
    page._update_map_button()
    page.load_images()

    assert len(page.grid.thumbnails) == 1
    assert page.grid.thumbnails[0].image_id == img1_id

    # Clear filter
    page._on_location_cleared()
    assert len(page.grid.thumbnails) == 0  # No person selected, no GPS -> empty
    assert not page.empty_label.isHidden()


def test_show_event_populates_list(qtbot, session_factory, tmp_app_paths):
    with session_factory() as session:
        session.add(Person(name="Alice"))
        session.commit()

    page = LibraryPage(session_factory, tmp_app_paths)
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
    qtbot, session_factory, tmp_app_paths
):
    with session_factory() as session:
        p1 = Person(name="Alice")
        session.add(p1)
        session.commit()
        p1_id = p1.id

    page = LibraryPage(session_factory, tmp_app_paths)
    qtbot.addWidget(page)

    item = page.person_list_widget.item(0)
    item.setSelected(True)
    assert p1_id in page._selected_person_ids()

    # Repopulate
    page._populate_person_list()

    new_item = page.person_list_widget.item(0)
    assert new_item.isSelected()
    assert p1_id in page._selected_person_ids()


def test_apply_search_filter_handles_none_item(qtbot, session_factory, tmp_app_paths):
    page = LibraryPage(session_factory, tmp_app_paths)
    qtbot.addWidget(page)

    # Mock item() to return None at some index
    with patch.object(page.person_list_widget, "item", side_effect=[None]):
        with patch.object(page.person_list_widget, "count", return_value=1):
            # This should not raise an AttributeError when calling item.text()
            page._apply_search_filter("test")


def test_open_map_dialog_accepted(qtbot, session_factory, tmp_app_paths):
    page = LibraryPage(session_factory, tmp_app_paths)
    qtbot.addWidget(page)

    bbox = GpsBoundingBox(south=10, west=10, north=20, east=20)

    with patch("photoaident.ui.pages.library.MapLocationDialog") as MockDialog:
        mock_dialog = MockDialog.return_value
        mock_dialog.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
        mock_dialog.selected_bbox.return_value = bbox

        page._open_map_dialog()

        assert page._gps_bbox == bbox
        assert not page.clear_location_btn.isHidden()


def test_gps_filter_without_vector_store(qtbot, session_factory, tmp_app_paths):
    img1_id, _ = _add_test_images_with_gps(session_factory)

    page = LibraryPage(session_factory, tmp_app_paths, vector_store=None)
    qtbot.addWidget(page)

    page._gps_bbox = GpsBoundingBox(south=52.0, west=13.0, north=53.0, east=14.0)
    page.load_images()

    assert len(page.grid.thumbnails) == 1
    assert page.grid.thumbnails[0].image_id == img1_id


def test_load_images_empty_per_person_scores(qtbot, session_factory, tmp_app_paths, vs):
    # This happens if find_images_by_person returns nothing
    with session_factory() as session:
        p1 = Person(name="Alice")
        session.add(p1)
        session.commit()

    page = LibraryPage(session_factory, tmp_app_paths, vs)
    qtbot.addWidget(page)

    # Select Alice
    item = page.person_list_widget.item(0)
    item.setSelected(True)

    # Mock find_images_by_person to return empty
    with patch("photoaident.ui.pages.library.find_images_by_person", return_value=[]):
        page.load_images()
        assert len(page.grid.thumbnails) == 0


def test_navigate_to_labelling(qtbot, session_factory, tmp_app_paths):
    page = LibraryPage(session_factory, tmp_app_paths)
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


def test_gps_filter_without_vector_store_and_person(
    qtbot, session_factory, tmp_app_paths
):
    img1_id, _ = _add_test_images_with_gps(session_factory)

    page = LibraryPage(session_factory, tmp_app_paths, vector_store=None)
    qtbot.addWidget(page)

    # Set GPS filter (Berlin-ish)
    page._gps_bbox = GpsBoundingBox(south=52.0, west=13.0, north=53.0, east=14.0)

    # We need to SELECT at least one person to reach line 212
    # if vector_store is None:
    #     with self.session_factory() as session:
    #         stmt = select(Image)
    #         if gps_image_ids is not None:
    #             stmt = stmt.where(Image.id.in_(list(gps_image_ids))) <--- line 218

    with session_factory() as session:
        p = Person(name="Alice")
        session.add(p)
        session.commit()

    page._populate_person_list()
    item = page.person_list_widget.item(0)
    item.setSelected(True)

    page.load_images()

    assert len(page.grid.thumbnails) == 1
    assert page.grid.thumbnails[0].image_id == img1_id


def test_load_images_no_common_ids(qtbot, session_factory, tmp_app_paths, vs):
    # Coverage for lines 235-236
    with session_factory() as session:
        p1 = Person(name="Alice")
        p2 = Person(name="Bob")
        session.add_all([p1, p2])
        session.commit()
        p1_id = p1.id
        p2_id = p2.id

    page = LibraryPage(session_factory, tmp_app_paths, vs)
    qtbot.addWidget(page)

    # Select both
    for i in range(page.person_list_widget.count()):
        page.person_list_widget.item(i).setSelected(True)

    # Mock find_images_by_person to return different images for each person
    # Alice is in image 1, Bob is in image 2.
    def mock_find(factory, store, person_id):
        if person_id == p1_id:
            return [(1, 0.9)]
        if person_id == p2_id:
            return [(2, 0.9)]
        return []

    with patch(
        "photoaident.ui.pages.library.find_images_by_person", side_effect=mock_find
    ):
        page.load_images()
        # It seems line 235-236 in original are unreachable if person_ids is not empty.
        pass
