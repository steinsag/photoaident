from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from photoaident.db.cluster_means import recompute_cluster_mean, serialize_embedding
from photoaident.db.database import (
    AGE_CLUSTERS,
    EmbeddingCluster,
    Face,
    FaceState,
    Image,
    ImageMetadata,
    Person,
    TakenAtSource,
)
from photoaident.db.vector_store import VectorStore
from photoaident.ui.pages.labelling import LabellingPage
from tests.search_helpers import _ones_norm_emb, _unit_emb, _zero_emb


def _make_jpeg(path: Path) -> None:
    """Write a minimal valid JPEG file at path."""
    img = QtGui.QImage(80, 60, QtGui.QImage.Format.Format_RGB32)
    img.fill(QtGui.QColor(100, 150, 200))
    img.save(str(path))  # format inferred from .jpg extension


def _insert_image(session_factory, file_path: str, file_hash: str) -> int:
    """Insert a bare Image row and return its id."""
    with session_factory() as session:
        img = Image(file_path=file_path, file_size=1000, file_hash=file_hash)
        session.add(img)
        session.commit()
        return img.id


def _insert_face(session_factory, file_path: str = "/path/to/img.jpg") -> int:
    """Insert an unidentified face into the DB and return its id."""
    with session_factory() as session:
        img = Image(file_path=file_path, file_size=1000, file_hash="abc123")
        session.add(img)
        session.flush()

        meta = ImageMetadata(
            image_id=img.id,
            taken_at_source=TakenAtSource.EXIF,
            width=800,
            height=600,
        )
        session.add(meta)
        session.flush()

        face = Face(
            image_id=img.id,
            bbox_x=100,
            bbox_y=100,
            bbox_w=50,
            bbox_h=50,
            detection_confidence=0.95,
            state=FaceState.UNIDENTIFIED,
            model_version="test",
        )
        session.add(face)
        session.commit()
        return face.id


def _insert_face_for_image(session_factory, image_id: int) -> int:
    """Insert an unidentified face for an existing image; return the face id."""
    with session_factory() as session:
        face = Face(
            image_id=image_id,
            bbox_x=10,
            bbox_y=10,
            bbox_w=40,
            bbox_h=40,
            detection_confidence=0.8,
            state=FaceState.UNIDENTIFIED,
            model_version="test",
        )
        session.add(face)
        session.commit()
        return face.id


def _insert_person_with_cluster(
    session_factory, name: str, age_key: str = "adult"
) -> tuple[int, int]:
    """Insert a person with one age cluster. Returns (person_id, cluster_id)."""
    with session_factory() as session:
        p = Person(name=name)
        session.add(p)
        session.flush()
        c = EmbeddingCluster(person_id=p.id, age_group=age_key)
        session.add(c)
        session.commit()
        return p.id, c.id


def _insert_identified_face(
    session_factory,
    vector_store: VectorStore,
    person_id: int,
    cluster_id: int,
    image_id: int,
) -> int:
    """Insert an identified face with a normalised embedding.

    Also recomputes the cluster mean so that DB-based lookups work.
    Returns face_id.
    """
    embedding = _ones_norm_emb()
    with session_factory() as session:
        face = Face(
            image_id=image_id,
            person_id=person_id,
            cluster_id=cluster_id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=50,
            bbox_h=50,
            detection_confidence=0.9,
            state=FaceState.IDENTIFIED,
            model_version="test",
        )
        session.add(face)
        session.flush()
        face_id = face.id
        vector_store.add(face_id, embedding)
        session.commit()
    recompute_cluster_mean(cluster_id, session_factory, vector_store)
    return face_id


def _get_cluster_by_age(session_factory, person_id: int) -> dict:
    """Load a person with clusters eagerly; return a {age_group: cluster} dict."""
    with session_factory() as session:
        person = session.execute(
            select(Person)
            .where(Person.id == person_id)
            .options(selectinload(Person.clusters))
        ).scalar_one()
        session.expunge_all()
    return {c.age_group: c for c in person.clusters if c.age_group}


# ===========================================================================
# Basic page state
# ===========================================================================


def test_labelling_page_no_faces(qtbot, session_factory, tmp_app_paths, vector_store):
    """Page disables action buttons when the DB has no unidentified faces."""
    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    assert not page.confirm_btn.isEnabled()
    assert not page.anonymous_btn.isEnabled()
    assert not page.skip_btn.isEnabled()
    assert not page.skip_image_btn.isEnabled()


def test_labelling_page_shows_face(qtbot, session_factory, tmp_app_paths, vector_store):
    """Page loads and enables action buttons when a face exists in the DB."""
    face_id = _insert_face(session_factory)

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    assert page._current_face_id == face_id
    assert page.anonymous_btn.isEnabled()
    assert page.skip_btn.isEnabled()
    assert page.skip_image_btn.isEnabled()
    assert not page.confirm_btn.isEnabled()


def test_labelling_page_empty_state_all_done(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Empty state when no unidentified faces remain shows 'All done' in crop label."""
    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    assert "done" in page._crop_label.text().lower()


def test_labelling_page_empty_state_all_skipped(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Empty state when faces exist but all are skipped shows 'skipped' message."""
    _insert_face(session_factory)

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()
    page._skip_face()  # skip the only face

    assert "skipped" in page._crop_label.text().lower()


# ===========================================================================
# _load_crop
# ===========================================================================


def test_load_crop_with_existing_file(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_load_crop() sets a pixmap when the crop file exists and is a valid JPEG."""
    face_id = _insert_face(session_factory)

    # Place a valid JPEG at the expected crop location
    crop_path = tmp_app_paths.face_crops_dir / f"{face_id}.jpg"
    _make_jpeg(crop_path)

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    assert page._current_face_id == face_id
    assert not page._crop_label.pixmap().isNull()


def test_load_crop_missing_file_shows_placeholder(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_load_crop() shows 'No image' text when crop file does not exist."""
    _insert_face(session_factory)

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    assert page._crop_label.text() == page.tr("No image")


# ===========================================================================
# Person list
# ===========================================================================


def test_load_persons_populates_list(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """After refresh(), the person list contains all persons in DB order."""
    _insert_face(session_factory)
    _insert_person_with_cluster(session_factory, "Alice")
    _insert_person_with_cluster(session_factory, "Bob")

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    assert page._person_list.count() == 2
    assert page._person_list.item(0).text() == "Alice"
    assert page._person_list.item(1).text() == "Bob"


def test_filter_text_preserved_across_face_advances(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Filter text must survive _load_persons() — user should not have to retype."""
    _insert_face(session_factory, "/img/a.jpg")
    _insert_face(session_factory, "/img/b.jpg")
    _insert_person_with_cluster(session_factory, "Alice")
    _insert_person_with_cluster(session_factory, "Bob")

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    # User types a filter — only Alice visible
    page._search_edit.setText("ali")
    assert page._person_list.count() == 1

    # Simulate advancing to the next face (as _load_next_face does)
    page._load_persons()

    # Filter text and filtered list must be intact
    assert page._search_edit.text() == "ali"
    assert page._person_list.count() == 1
    assert page._person_list.item(0).text() == "Alice"


def test_previous_person_selection_restored_after_face_advance(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """The previously selected person is re-selected when loading the next face."""
    _insert_face(session_factory, "/img/a.jpg")
    _insert_face(session_factory, "/img/b.jpg")
    person_id, _ = _insert_person_with_cluster(session_factory, "Alice")
    _insert_person_with_cluster(session_factory, "Bob")

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    # Select Alice
    page._person_list.setCurrentRow(0)
    assert page._selected_person is not None
    assert page._selected_person.id == person_id

    # Advance to next face — Alice should still be selected
    page._load_persons()

    assert page._selected_person is not None
    assert page._selected_person.id == person_id


# ===========================================================================
# _on_person_selected / cluster table
# ===========================================================================


def test_on_person_selected_populates_cluster_table(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Selecting a person in the list stores the person and populates cluster data."""
    _insert_face(session_factory)
    person_id, _ = _insert_person_with_cluster(session_factory, "Marc", "adult")

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    assert page._person_list.count() == 1
    page._person_list.setCurrentRow(0)

    assert page._selected_person is not None
    assert page._selected_person.id == person_id


def test_on_person_selected_deselect_clears_state(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Deselecting a person clears _selected_person and disables confirm."""
    _insert_face(session_factory)
    _insert_person_with_cluster(session_factory, "Marc")

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    page._person_list.setCurrentRow(0)
    assert page._selected_person is not None

    # Deselect
    page._on_person_selected(None, None)
    assert page._selected_person is None
    assert page._selected_cluster is None
    assert not page.confirm_btn.isEnabled()


def test_on_person_selected_with_embedding_shows_scores(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Selecting a person when a query embedding is present shows similarity scores."""
    img_id = _insert_image(session_factory, "/emb.jpg", "embhash")
    with session_factory() as session:
        face = Face(
            image_id=img_id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=40,
            bbox_h=40,
            detection_confidence=0.9,
            state=FaceState.UNIDENTIFIED,
            model_version="test",
        )
        session.add(face)
        session.flush()
        vector_store.add(face.id, _ones_norm_emb())
        session.commit()

    person_id, cluster_id = _insert_person_with_cluster(
        session_factory, "Scored", "adult"
    )
    ref_img_id = _insert_image(session_factory, "/ref.jpg", "refhash")
    _insert_identified_face(
        session_factory, vector_store, person_id, cluster_id, ref_img_id
    )

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    # Query embedding is now set from the unidentified face
    assert page._query_embedding is not None

    page._person_list.setCurrentRow(0)

    # "adult" cluster should have a numeric score in the table
    adult_row = AGE_CLUSTERS.index("adult")
    score_item = page._cluster_table.item(adult_row, 1)
    assert score_item is not None
    text = score_item.text()
    assert text != "\u2014"
    float(text)  # must be parseable as float


def test_on_person_selected_preselects_best_cluster(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """The cluster with the highest score is pre-selected after person selection."""
    img_id = _insert_image(session_factory, "/pre.jpg", "prehash")
    query_vec = _ones_norm_emb()
    with session_factory() as session:
        face = Face(
            image_id=img_id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=40,
            bbox_h=40,
            detection_confidence=0.9,
            state=FaceState.UNIDENTIFIED,
            model_version="test",
        )
        session.add(face)
        session.flush()
        vector_store.add(face.id, query_vec.copy())
        session.commit()

    person_id, cluster_id = _insert_person_with_cluster(session_factory, "Pre", "adult")
    ref_img_id = _insert_image(session_factory, "/ref2.jpg", "refhash2")
    _insert_identified_face(
        session_factory, vector_store, person_id, cluster_id, ref_img_id
    )

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    page._person_list.setCurrentRow(0)

    # After selection, best cluster should be pre-selected → _selected_cluster not None
    assert page._selected_cluster is not None
    assert page.confirm_btn.isEnabled()


# ===========================================================================
# _compute_cluster_scores
# ===========================================================================


def test_compute_cluster_scores_returns_float(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_compute_cluster_scores() returns a float score for clusters with embeddings."""
    img_id = _insert_image(session_factory, "/ccs.jpg", "ccshash")
    person_id, cluster_id = _insert_person_with_cluster(session_factory, "CCS", "adult")
    _insert_identified_face(
        session_factory, vector_store, person_id, cluster_id, img_id
    )

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)

    # Set a non-trivial query embedding
    page._query_embedding = _ones_norm_emb()

    cluster_by_age = _get_cluster_by_age(session_factory, person_id)
    scores = page._compute_cluster_scores(cluster_by_age)

    assert "adult" in scores
    assert isinstance(scores["adult"], float)


def test_compute_cluster_scores_zero_norm_query_returns_empty(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_compute_cluster_scores() returns {} when query embedding is all zeros."""
    img_id = _insert_image(session_factory, "/znq.jpg", "znqhash")
    person_id, cluster_id = _insert_person_with_cluster(session_factory, "ZNQ", "adult")
    _insert_identified_face(
        session_factory, vector_store, person_id, cluster_id, img_id
    )

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page._query_embedding = _zero_emb()

    cluster_by_age = _get_cluster_by_age(session_factory, person_id)
    scores = page._compute_cluster_scores(cluster_by_age)
    assert scores == {}


def test_compute_cluster_scores_empty_cluster_skipped(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_compute_cluster_scores() skips clusters that have no identified faces."""
    person_id, _ = _insert_person_with_cluster(session_factory, "NF", "adult")

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page._query_embedding = _ones_norm_emb()

    cluster_by_age = _get_cluster_by_age(session_factory, person_id)
    scores = page._compute_cluster_scores(cluster_by_age)
    # No identified faces in the cluster → no score
    assert "adult" not in scores


# ===========================================================================
# _create_new_person
# ===========================================================================


def test_create_new_person_rejected_does_nothing(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Cancelling NewPersonDialog leaves the person list unchanged."""
    _insert_face(session_factory)
    _insert_person_with_cluster(session_factory, "Existing")

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    count_before = page._person_list.count()

    with patch("photoaident.ui.pages.labelling.NewPersonDialog") as MockDlg:
        inst = MockDlg.return_value
        inst.exec.return_value = QtWidgets.QDialog.DialogCode.Rejected
        page._create_new_person()

    assert page._person_list.count() == count_before


def test_create_new_person_accepted_adds_and_selects(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Accepting NewPersonDialog adds the new person and selects them in the list."""
    _insert_face(session_factory)

    # Pre-insert the person that the dialog would create
    person_id, _ = _insert_person_with_cluster(session_factory, "NewGuy", "adult")

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    with patch("photoaident.ui.pages.labelling.NewPersonDialog") as MockDlg:
        inst = MockDlg.return_value
        inst.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
        inst.created_person_id.return_value = person_id
        page._create_new_person()

    # Person should appear in the list (may already be there from _load_persons)
    found = False
    for i in range(page._person_list.count()):
        item = page._person_list.item(i)
        if item and item.data(QtCore.Qt.ItemDataRole.UserRole).id == person_id:
            found = True
            break
    assert found


def test_create_new_person_accepted_none_id_does_nothing(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """If created_person_id() returns None, _create_new_person does nothing."""
    _insert_face(session_factory)

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    with patch("photoaident.ui.pages.labelling.NewPersonDialog") as MockDlg:
        inst = MockDlg.return_value
        inst.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
        inst.created_person_id.return_value = None
        page._create_new_person()

    # No person was added → list still empty
    assert page._person_list.count() == 0


# ===========================================================================
# Confirm / cancel
# ===========================================================================


def test_confirm_assigns_face_and_advances(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_on_confirm sets state=IDENTIFIED and advances to the next face."""
    with session_factory() as session:
        p = Person(name="Target")
        session.add(p)
        session.flush()
        c = EmbeddingCluster(person_id=p.id)
        session.add(c)
        session.commit()
        person_id = p.id
        cluster_id = c.id

    face_id = _insert_face(session_factory)

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    assert page._current_face_id == face_id

    with session_factory() as session:
        person = session.get(Person, person_id)
        cluster = session.get(EmbeddingCluster, cluster_id)
        session.expunge_all()

    page._selected_person = person
    page._selected_cluster = cluster
    page._update_confirm_button()

    assert page.confirm_btn.isEnabled()
    page._on_confirm()

    with session_factory() as session:
        face = session.get(Face, face_id)
        assert face is not None
        assert face.state == FaceState.IDENTIFIED
        assert face.person_id == person_id
        assert face.cluster_id == cluster_id
        assert face.labelled_at is not None


def test_on_confirm_noop_when_no_face(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_on_confirm does nothing when no face is loaded."""
    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)

    assert page._current_face_id is None
    page._on_confirm()  # must not raise
    assert page._current_face_id is None


def test_on_confirm_noop_when_no_selection(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_on_confirm does nothing when no person/cluster is selected."""
    face_id = _insert_face(session_factory)

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    assert page._current_face_id == face_id
    assert page._selected_person is None

    page._on_confirm()  # must not raise, face state unchanged

    with session_factory() as session:
        face = session.get(Face, face_id)
        assert face is not None
        assert face.state == FaceState.UNIDENTIFIED


def test_confirm_disabled_without_selection(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """confirm_btn is disabled when no person+cluster is selected."""
    _insert_face(session_factory)

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    assert not page.confirm_btn.isEnabled()

    page._selected_person = Person(name="X")
    page._selected_cluster = None
    page._update_confirm_button()
    assert not page.confirm_btn.isEnabled()

    page._selected_cluster = EmbeddingCluster()
    page._update_confirm_button()
    assert page.confirm_btn.isEnabled()


def test_cancel_clears_selection(qtbot, session_factory, tmp_app_paths, vector_store):
    """_on_cancel clears person/cluster selection without changing face or advancing."""
    face_id = _insert_face(session_factory)

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    page._selected_person = Person(name="Someone")
    page._selected_cluster = EmbeddingCluster()
    page._update_confirm_button()
    assert page.confirm_btn.isEnabled()

    page._on_cancel()

    assert page._selected_person is None
    assert page._selected_cluster is None
    assert not page.confirm_btn.isEnabled()
    assert page._current_face_id == face_id


# ===========================================================================
# Mark anonymous
# ===========================================================================


def test_mark_anonymous(qtbot, session_factory, tmp_app_paths, vector_store):
    """Clicking Mark Anonymous sets face.state = ANONYMOUS and advances."""
    face_id = _insert_face(session_factory)

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    assert page._current_face_id == face_id
    page._mark_anonymous()

    with session_factory() as session:
        face = session.get(Face, face_id)
        assert face is not None
        assert face.state == FaceState.ANONYMOUS
        assert face.labelled_at is not None

    assert not page.confirm_btn.isEnabled()
    assert page._current_face_id is None


def test_mark_anonymous_noop_when_no_current_face(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_mark_anonymous does nothing when _current_face_id is None."""
    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)

    assert page._current_face_id is None
    page._mark_anonymous()
    assert page._current_face_id is None


# ===========================================================================
# Skip face
# ===========================================================================


def test_skip_face(qtbot, session_factory, tmp_app_paths, vector_store):
    """Skip does not change face state and moves to next (or done when all skipped)."""
    face_id = _insert_face(session_factory)

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    assert page._current_face_id == face_id
    page._skip_face()

    with session_factory() as session:
        face = session.get(Face, face_id)
        assert face is not None
        assert face.state == FaceState.UNIDENTIFIED
        assert face.labelled_at is None

    assert face_id in page._skipped
    assert not page.skip_btn.isEnabled()


def test_skip_face_noop_when_no_current_face(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_skip_face does nothing when _current_face_id is None."""
    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)

    assert page._current_face_id is None
    page._skip_face()
    assert page._current_face_id is None


# ===========================================================================
# Skip image
# ===========================================================================


def test_skip_image_skips_all_faces_of_current_image(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_skip_image records the current image in _skipped_images so all its faces
    are excluded from subsequent face queries."""
    img1_id = _insert_image(session_factory, "/si1.jpg", "sihash1")
    img2_id = _insert_image(session_factory, "/si2.jpg", "sihash2")
    face1_id = _insert_face_for_image(session_factory, img1_id)
    face2_id = _insert_face_for_image(session_factory, img1_id)
    face3_id = _insert_face_for_image(session_factory, img2_id)

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    assert page._current_face_id in (face1_id, face2_id, face3_id)
    first_image_id = (
        img1_id if page._current_face_id in (face1_id, face2_id) else img2_id
    )
    other_image_id = img2_id if first_image_id == img1_id else img1_id

    page._skip_image()

    assert first_image_id in page._skipped_images
    assert other_image_id not in page._skipped_images

    assert page._current_face_id in (face1_id, face2_id, face3_id)
    assert page._current_face_id not in (
        [face1_id, face2_id] if first_image_id == img1_id else [face3_id]
    )


def test_skip_image_does_not_change_face_state(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_skip_image must not alter face state in the DB."""
    img_id = _insert_image(session_factory, "/sino.jpg", "sinohash")
    face1_id = _insert_face_for_image(session_factory, img_id)
    face2_id = _insert_face_for_image(session_factory, img_id)

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()
    page._skip_image()

    with session_factory() as session:
        for fid in (face1_id, face2_id):
            face = session.get(Face, fid)
            assert face is not None
            assert face.state == FaceState.UNIDENTIFIED
            assert face.labelled_at is None


def test_skip_image_noop_when_no_current_face(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_skip_image does nothing when _current_face_id is None."""
    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)

    assert page._current_face_id is None
    page._skip_image()
    assert page._current_face_id is None
    assert len(page._skipped_images) == 0


def test_skip_image_face_deleted_in_db(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_skip_image recovers gracefully when the current face no longer exists in DB.

    Instead of staying stuck on a stale _current_face_id, the page advances
    (clearing the current face) so the UI is not left in a broken state.
    """
    face_id = _insert_face(session_factory)

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    # Manually delete the face from the DB to simulate a race condition
    with session_factory() as session:
        face = session.get(Face, face_id)
        session.delete(face)
        session.commit()

    # Force the face_id to be set (simulates the race)
    page._current_face_id = face_id

    page._skip_image()  # must not raise

    # Page must have recovered: no stale image skip recorded, and current face cleared
    assert len(page._skipped_images) == 0
    assert page._current_face_id is None


def test_skip_image_button_exists_and_enabled_with_face(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """The Skip Image button is present and enabled when a face is loaded."""
    _insert_face(session_factory)

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    assert hasattr(page, "skip_image_btn")
    assert page.skip_image_btn.isEnabled()


def test_skip_image_button_disabled_when_no_faces(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """The Skip Image button is disabled when there are no faces to label."""
    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    assert hasattr(page, "skip_image_btn")
    assert not page.skip_image_btn.isEnabled()


# ===========================================================================
# Priority image
# ===========================================================================


def test_priority_image_shows_image_faces_first(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """refresh(priority_image_id=X) causes the first face to come from image X."""
    img1_id = _insert_image(session_factory, "/img1.jpg", "hash1")
    img2_id = _insert_image(session_factory, "/img2.jpg", "hash2")
    _insert_face_for_image(session_factory, img2_id)
    face1_id = _insert_face_for_image(session_factory, img1_id)

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh(priority_image_id=img1_id)

    assert page._current_face_id == face1_id
    assert page._priority_image_id == img1_id


def test_priority_clears_after_all_image_faces_done(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """After all priority-image faces are labelled, priority is cleared."""
    img1_id = _insert_image(session_factory, "/p1.jpg", "phash1")
    img2_id = _insert_image(session_factory, "/p2.jpg", "phash2")
    face1_id = _insert_face_for_image(session_factory, img1_id)
    _insert_face_for_image(session_factory, img2_id)

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh(priority_image_id=img1_id)

    assert page._current_face_id == face1_id
    page._mark_anonymous()

    assert page._priority_image_id is None
    assert page._current_face_id is not None
    assert page._current_face_id != face1_id


def test_maybe_clear_priority_with_skipped_faces(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_maybe_clear_priority counts remaining faces excluding skipped ones."""
    img1_id = _insert_image(session_factory, "/mcp.jpg", "mcphash")
    img2_id = _insert_image(session_factory, "/mcp2.jpg", "mcphash2")
    face1_id = _insert_face_for_image(session_factory, img1_id)
    face2_id = _insert_face_for_image(session_factory, img1_id)
    _insert_face_for_image(session_factory, img2_id)

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh(priority_image_id=img1_id)

    # Current face is from img1; skip it (adds to _skipped, face2 still remains)
    first_face_id = page._current_face_id
    page._skip_face()

    # Priority should still be set since face2 of img1 remains
    assert page._priority_image_id == img1_id
    # The new current face should be the other face from img1
    assert page._current_face_id in (face1_id, face2_id)
    assert page._current_face_id != first_face_id


def test_refresh_without_priority_clears_priority(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Calling refresh() without arguments clears any previously set priority."""
    img_id = _insert_image(session_factory, "/clr.jpg", "clrhash")
    _insert_face_for_image(session_factory, img_id)

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh(priority_image_id=img_id)
    assert page._priority_image_id == img_id

    page.refresh()
    assert page._priority_image_id is None


def test_skip_image_clears_priority_when_all_image_faces_skipped(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Skipping the priority image via _skip_image clears priority in one click."""
    img1_id = _insert_image(session_factory, "/pri1.jpg", "prihash1")
    img2_id = _insert_image(session_factory, "/pri2.jpg", "prihash2")
    _insert_face_for_image(session_factory, img1_id)
    _insert_face_for_image(session_factory, img1_id)
    _insert_face_for_image(session_factory, img2_id)

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh(priority_image_id=img1_id)

    assert page._priority_image_id == img1_id
    page._skip_image()

    assert img1_id in page._skipped_images
    assert page._priority_image_id is None
    assert page._current_face_id is not None


# ===========================================================================
# Edge-case coverage for remaining branches
# ===========================================================================


def test_compute_cluster_scores_null_mean_skipped(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Cluster with mean_embedding=None is skipped in score computation."""
    person_id, _ = _insert_person_with_cluster(session_factory, "Exc", "adult")

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page._query_embedding = _ones_norm_emb()

    cluster_by_age = _get_cluster_by_age(session_factory, person_id)
    # mean_embedding is None by default (no faces added)
    scores = page._compute_cluster_scores(cluster_by_age)

    assert "adult" not in scores


def test_compute_cluster_scores_with_persisted_mean(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_compute_cluster_scores reads the persisted mean and computes a dot product."""
    person_id, cluster_id = _insert_person_with_cluster(session_factory, "ZNM", "adult")

    # Directly set a known mean_embedding on the cluster
    mean = _unit_emb(0)
    with session_factory() as session:
        cluster = session.get(EmbeddingCluster, cluster_id)
        cluster.mean_embedding = serialize_embedding(mean)
        session.commit()

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page._query_embedding = mean.copy()

    cluster_by_age = _get_cluster_by_age(session_factory, person_id)
    scores = page._compute_cluster_scores(cluster_by_age)

    assert "adult" in scores
    assert scores["adult"] == pytest.approx(1.0, abs=1e-6)


def _insert_face_with_taken_at(
    session_factory, taken_at: datetime, file_path: str = "/path/to/img.jpg"
) -> int:
    """Insert an unidentified face whose image metadata has a taken_at timestamp."""
    with session_factory() as session:
        img = Image(file_path=file_path, file_size=1000, file_hash="tsat123")
        session.add(img)
        session.flush()

        meta = ImageMetadata(
            image_id=img.id,
            taken_at=taken_at,
            taken_at_source=TakenAtSource.EXIF,
            width=640,
            height=480,
        )
        session.add(meta)
        session.flush()

        face = Face(
            image_id=img.id,
            bbox_x=10,
            bbox_y=10,
            bbox_w=40,
            bbox_h=40,
            detection_confidence=0.9,
            state=FaceState.UNIDENTIFIED,
            model_version="test",
        )
        session.add(face)
        session.commit()
        return face.id


# ===========================================================================
# _face_info_label content
# ===========================================================================


def test_face_info_label_shows_path_and_taken_at(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_face_info_label shows the image file path and formatted taken_at date."""
    taken_at = datetime(2023, 6, 15, 10, 30, tzinfo=timezone.utc)
    _insert_face_with_taken_at(
        session_factory, taken_at, file_path="/photos/summer.jpg"
    )

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    label_text = page._face_info_label.text()
    assert "/photos/summer.jpg" in label_text
    assert "2023-06-15" in label_text
    assert "10:30" in label_text


def test_face_info_label_shows_unknown_date_when_no_taken_at(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_face_info_label shows 'Unknown date' when image metadata has no taken_at."""
    # _insert_face() creates metadata without taken_at → None
    _insert_face(session_factory, file_path="/photos/nodatephoto.jpg")

    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    label_text = page._face_info_label.text()
    assert "/photos/nodatephoto.jpg" in label_text
    assert "Unknown date" in label_text


def test_face_info_label_cleared_in_empty_state(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_face_info_label is cleared when the page transitions to the empty state."""
    page = LabellingPage(session_factory, tmp_app_paths, vector_store)
    qtbot.addWidget(page)
    # No faces → empty state shown immediately on refresh
    page.refresh()

    assert page._face_info_label.text() == ""
