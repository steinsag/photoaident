"""Tests for LibraryPage: collapsible person filter panel, image selection."""

from unittest.mock import patch

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
from sqlalchemy import create_engine

from photoaident.db.database import (
    EmbeddingCluster,
    Face,
    FaceState,
    Image,
    ImageMetadata,
    Person,
    TakenAtSource,
    get_engine,
    get_session_factory,
)
from photoaident.db.migrate import apply_migrations
from photoaident.db.vector_store import VectorStore
from photoaident.paths import AppPaths
from photoaident.ui.pages.library import LibraryPage


@pytest.fixture
def session_factory(tmp_path):
    db_path = tmp_path / "lib_test.db"
    engine = get_engine(str(db_path))
    apply_migrations(f"sqlite:///{db_path}")
    return get_session_factory(engine)


@pytest.fixture
def test_paths(tmp_path):
    paths = AppPaths(
        base_data=tmp_path / "data",
        base_cache=tmp_path / "cache",
        base_config=tmp_path / "config",
    )
    paths.ensure_dirs()
    return paths


@pytest.fixture
def vs():
    return VectorStore()


def _add_image(
    session_factory, file_path="/img.jpg", file_hash="abc123", file_size=1000
):
    with session_factory() as session:
        img = Image(file_path=file_path, file_size=file_size, file_hash=file_hash)
        session.add(img)
        session.commit()
        return img.id


def _add_image_with_metadata(session_factory, file_path="/meta.jpg"):
    with session_factory() as session:
        img = Image(file_path=file_path, file_size=2000, file_hash="hashwithmeta")
        session.add(img)
        session.flush()
        meta = ImageMetadata(
            image_id=img.id,
            taken_at_source=TakenAtSource.FILESYSTEM,
            width=800,
            height=600,
        )
        session.add(meta)
        session.commit()
        return img.id


def _add_face(session_factory, image_id, faiss_id=0, state=FaceState.UNIDENTIFIED):
    with session_factory() as session:
        face = Face(
            image_id=image_id,
            faiss_id=faiss_id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=50,
            bbox_h=50,
            detection_confidence=0.9,
            state=state,
            model_version="test",
        )
        session.add(face)
        session.commit()
        return face.id


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


# --- Filter button state ---


def test_person_filter_btn_disabled_when_no_persons(qtbot, session_factory, test_paths):
    page = LibraryPage(session_factory, test_paths)
    qtbot.addWidget(page)
    assert not page.person_filter_btn.isEnabled()


def test_person_filter_btn_enabled_when_persons_exist(
    qtbot, session_factory, test_paths
):
    with session_factory() as session:
        session.add(Person(name="Alice"))
        session.commit()

    page = LibraryPage(session_factory, test_paths)
    qtbot.addWidget(page)
    assert page.person_filter_btn.isEnabled()


# --- Filter panel visibility ---


def test_filter_panel_hidden_initially(qtbot, session_factory, test_paths):
    page = LibraryPage(session_factory, test_paths)
    qtbot.addWidget(page)
    assert page.filter_panel.isHidden()


def test_filter_panel_shows_when_button_checked(qtbot, session_factory, test_paths):
    with session_factory() as session:
        session.add(Person(name="Alice"))
        session.commit()

    page = LibraryPage(session_factory, test_paths)
    qtbot.addWidget(page)
    page.person_filter_btn.setChecked(True)
    assert not page.filter_panel.isHidden()


# --- Person list structure ---


def test_person_list_items_are_checkable(qtbot, session_factory, test_paths):
    with session_factory() as session:
        session.add(Person(name="Bob"))
        session.commit()

    page = LibraryPage(session_factory, test_paths)
    qtbot.addWidget(page)

    assert page.person_list_widget.count() == 1
    item = page.person_list_widget.item(0)
    assert item is not None
    assert item.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable


def test_person_list_unchecked_by_default(qtbot, session_factory, test_paths):
    with session_factory() as session:
        session.add(Person(name="Carol"))
        session.commit()

    page = LibraryPage(session_factory, test_paths)
    qtbot.addWidget(page)

    item = page.person_list_widget.item(0)
    assert item is not None
    assert item.checkState() == QtCore.Qt.CheckState.Unchecked


# --- Search filter ---


def test_search_filter_hides_non_matching_items(qtbot, session_factory, test_paths):
    with session_factory() as session:
        session.add(Person(name="Alice"))
        session.add(Person(name="Bob"))
        session.commit()

    page = LibraryPage(session_factory, test_paths)
    qtbot.addWidget(page)

    page.search_edit.setText("ali")

    bob_item = None
    for i in range(page.person_list_widget.count()):
        item = page.person_list_widget.item(i)
        if item and item.text() == "Bob":
            bob_item = item
    assert bob_item is not None
    assert bob_item.isHidden()


def test_search_filter_case_insensitive(qtbot, session_factory, test_paths):
    with session_factory() as session:
        session.add(Person(name="Alice"))
        session.commit()

    page = LibraryPage(session_factory, test_paths)
    qtbot.addWidget(page)

    page.search_edit.setText("ALICE")

    item = page.person_list_widget.item(0)
    assert item is not None
    assert not item.isHidden()


# --- load_images: no selection → show all ---


def test_no_person_checked_shows_all_images(qtbot, session_factory, test_paths):
    _add_image(session_factory, "/img1.jpg", file_hash="h1")
    _add_image(session_factory, "/img2.jpg", file_hash="h2")

    page = LibraryPage(session_factory, test_paths)
    qtbot.addWidget(page)

    assert len(page.grid.thumbnails) == 2


def test_no_person_checked_with_images_in_db(qtbot, session_factory, test_paths):
    for i in range(5):
        _add_image(session_factory, f"/img{i}.jpg", file_hash=f"hash{i}")

    page = LibraryPage(session_factory, test_paths)
    qtbot.addWidget(page)

    assert len(page.grid.thumbnails) == 5


def test_image_without_hash_uses_unknown_thumb(qtbot, session_factory, test_paths):
    with session_factory() as session:
        img = Image(file_path="/no_hash.jpg", file_size=500, file_hash=None)
        session.add(img)
        session.commit()

    page = LibraryPage(session_factory, test_paths)
    qtbot.addWidget(page)

    assert len(page.grid.thumbnails) == 1
    assert page.grid.thumbnails[0].thumb_path.name == "unknown.jpg"


def test_image_with_metadata_appears_in_grid(qtbot, session_factory, test_paths):
    """An image with metadata is loaded into the grid with the correct thumb_path."""
    _add_image_with_metadata(session_factory, "/meta.jpg")

    page = LibraryPage(session_factory, test_paths)
    qtbot.addWidget(page)

    assert len(page.grid.thumbnails) == 1
    assert page.grid.thumbnails[0].thumb_path.name == "hashwithmeta.jpg"


# --- load_images: person filter ---


def test_select_one_person_shows_matched_images(qtbot, session_factory, test_paths, vs):
    _, _ = _add_person_with_face(session_factory, vs, "Eve", "/eve.jpg")

    page = LibraryPage(session_factory, test_paths, vs)
    qtbot.addWidget(page)

    item = page.person_list_widget.item(0)
    assert item is not None
    page.person_list_widget.blockSignals(False)
    item.setCheckState(QtCore.Qt.CheckState.Checked)

    assert len(page.grid.thumbnails) >= 1


def test_select_person_with_no_faces_shows_empty(
    qtbot, session_factory, test_paths, vs
):
    with session_factory() as session:
        p = Person(name="Nobody")
        session.add(p)
        session.flush()
        session.add(EmbeddingCluster(person_id=p.id))
        session.commit()

    page = LibraryPage(session_factory, test_paths, vs)
    qtbot.addWidget(page)

    item = page.person_list_widget.item(0)
    assert item is not None
    page.person_list_widget.blockSignals(False)
    item.setCheckState(QtCore.Qt.CheckState.Checked)

    assert len(page.grid.thumbnails) == 0


def test_select_multiple_persons_intersects_results(
    qtbot, session_factory, test_paths, vs
):
    """Selecting two persons shows only images where BOTH appear (AND, not OR)."""
    # Use orthogonal embeddings so Alice's search won't match Bob's face and vice versa
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

    emb1 = np.zeros(512, dtype=np.float32)
    emb1[0] = 1.0
    emb2 = np.zeros(512, dtype=np.float32)
    emb2[1] = 1.0
    faiss_id1 = vs.add(emb1)
    faiss_id2 = vs.add(emb2)

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

    page = LibraryPage(session_factory, test_paths, vs)
    qtbot.addWidget(page)

    page.person_list_widget.blockSignals(True)
    for i in range(page.person_list_widget.count()):
        item = page.person_list_widget.item(i)
        assert item is not None
        item.setCheckState(QtCore.Qt.CheckState.Checked)
    page.person_list_widget.blockSignals(False)
    page.load_images()

    # Alice is only in alice.jpg, Bob is only in bob.jpg → no image has both
    assert len(page.grid.thumbnails) == 0


def test_select_multiple_persons_shows_shared_image(
    qtbot, session_factory, test_paths, vs
):
    """An image containing all selected persons is included in AND results."""
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

    page = LibraryPage(session_factory, test_paths, vs)
    qtbot.addWidget(page)

    page.person_list_widget.blockSignals(True)
    for i in range(page.person_list_widget.count()):
        item = page.person_list_widget.item(i)
        assert item is not None
        item.setCheckState(QtCore.Qt.CheckState.Checked)
    page.person_list_widget.blockSignals(False)
    page.load_images()

    result_ids = [t.image_id for t in page.grid.thumbnails]
    assert shared_img_id in result_ids
    assert len(page.grid.thumbnails) == 1


def test_person_filter_without_vector_store_falls_back_to_all(
    qtbot, session_factory, test_paths
):
    _add_image(session_factory, "/img1.jpg", file_hash="h1")
    _add_image(session_factory, "/img2.jpg", file_hash="h2")
    with session_factory() as session:
        p = Person(name="Alice")
        session.add(p)
        session.commit()

    page = LibraryPage(session_factory, test_paths, vector_store=None)
    qtbot.addWidget(page)

    item = page.person_list_widget.item(0)
    assert item is not None
    page.person_list_widget.blockSignals(False)
    item.setCheckState(QtCore.Qt.CheckState.Checked)

    assert len(page.grid.thumbnails) == 2


# --- Deselect All ---


def test_deselect_all_clears_selection_and_shows_all(
    qtbot, session_factory, test_paths, vs
):
    _add_person_with_face(session_factory, vs, "Alice", "/alice.jpg")
    _add_image(session_factory, "/extra.jpg", file_hash="extra")

    page = LibraryPage(session_factory, test_paths, vs)
    qtbot.addWidget(page)

    # Select Alice
    page.person_list_widget.blockSignals(True)
    item = page.person_list_widget.item(0)
    assert item is not None
    item.setCheckState(QtCore.Qt.CheckState.Checked)
    page.person_list_widget.blockSignals(False)
    page.load_images()

    # Deselect all → should show all images
    page._deselect_all()
    assert len(page.grid.thumbnails) == 2


# --- Button text ---


def test_button_text_updates_when_person_selected(
    qtbot, session_factory, test_paths, vs
):
    _add_person_with_face(session_factory, vs, "Alice", "/alice.jpg")

    page = LibraryPage(session_factory, test_paths, vs)
    qtbot.addWidget(page)

    page.person_list_widget.blockSignals(True)
    item = page.person_list_widget.item(0)
    assert item is not None
    item.setCheckState(QtCore.Qt.CheckState.Checked)
    page.person_list_widget.blockSignals(False)
    page._update_button_text()

    assert "1" in page.person_filter_btn.text()


def test_button_text_resets_after_deselect_all(qtbot, session_factory, test_paths, vs):
    _add_person_with_face(session_factory, vs, "Alice", "/alice.jpg")

    page = LibraryPage(session_factory, test_paths, vs)
    qtbot.addWidget(page)

    page.person_list_widget.blockSignals(True)
    item = page.person_list_widget.item(0)
    assert item is not None
    item.setCheckState(QtCore.Qt.CheckState.Checked)
    page.person_list_widget.blockSignals(False)
    page._update_button_text()

    page._deselect_all()
    assert page.person_filter_btn.text() == page.tr("Filter by Person")


# --- _on_image_selected ---


def test_on_image_selected_opens_dialog(qtbot, session_factory, test_paths):
    img_id = _add_image(session_factory, "/sel.jpg")

    page = LibraryPage(session_factory, test_paths)
    qtbot.addWidget(page)

    with patch("photoaident.ui.pages.library.ImageDetailDialog") as MockDlg:
        MockDlg.return_value.exec.return_value = None
        page._on_image_selected(img_id)
        MockDlg.assert_called_once()
        MockDlg.return_value.exec.assert_called_once()


def test_on_image_selected_nonexistent_id_skips_dialog(
    qtbot, session_factory, test_paths
):
    page = LibraryPage(session_factory, test_paths)
    qtbot.addWidget(page)

    with patch("photoaident.ui.pages.library.ImageDetailDialog") as MockDlg:
        page._on_image_selected(99999)
        MockDlg.assert_not_called()
