import re
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
from photoaident.ui.widgets.labelling_dialog import LabellingDialog
from tests.search_helpers import _ones_norm_emb, _unit_emb, _zero_emb


def _make_jpeg(path: Path) -> None:
    """Write a minimal valid JPEG file at path."""
    img = QtGui.QImage(80, 60, QtGui.QImage.Format.Format_RGB32)
    img.fill(QtGui.QColor(100, 150, 200))
    img.save(str(path))


def _insert_image(session_factory, file_path: str, file_hash: str) -> int:
    """Insert a bare Image row and return its id."""
    with session_factory() as session:
        img = Image(file_path=file_path, file_size=1000, file_hash=file_hash)
        session.add(img)
        session.commit()
        return img.id


def _insert_face(
    session_factory, file_path: str = "/path/to/img.jpg"
) -> tuple[int, int]:
    """Insert an unidentified face into the DB and return (face_id, image_id)."""
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
        return face.id, img.id


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


def _make_dialog(session_factory, tmp_app_paths, vector_store, image_id):
    """Convenience constructor for LabellingDialog."""
    return LabellingDialog(image_id, session_factory, tmp_app_paths, vector_store)


# ===========================================================================
# Basic dialog state
# ===========================================================================


def test_labelling_dialog_no_faces(qtbot, session_factory, tmp_app_paths, vector_store):
    """Dialog disables action buttons when the image has no unidentified faces."""
    image_id = _insert_image(session_factory, "/empty.jpg", "emptyhash")
    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    assert not dialog.confirm_btn.isEnabled()
    assert not dialog.anonymous_btn.isEnabled()
    assert not dialog.skip_btn.isEnabled()


def test_labelling_dialog_shows_face(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Dialog loads and enables action buttons when a face exists for the image."""
    face_id, image_id = _insert_face(session_factory)

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    assert dialog._current_face_id == face_id
    assert dialog.anonymous_btn.isEnabled()
    assert dialog.skip_btn.isEnabled()
    assert not dialog.confirm_btn.isEnabled()


def test_labelling_dialog_empty_state_all_done(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Completion state is shown when the image has no unidentified faces."""
    image_id = _insert_image(session_factory, "/done.jpg", "donehash")
    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    assert "done" in dialog._crop_label.text().lower()


# ===========================================================================
# _load_crop
# ===========================================================================


def test_load_crop_with_existing_file(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Dialog sets a pixmap when the crop file exists and is a valid JPEG."""
    face_id, image_id = _insert_face(session_factory)

    crop_path = tmp_app_paths.face_crops_dir / f"{face_id}.jpg"
    _make_jpeg(crop_path)

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    assert dialog._current_face_id == face_id
    assert not dialog._crop_label.pixmap().isNull()


def test_load_crop_missing_file_shows_placeholder(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Dialog shows 'No image' text when crop file does not exist."""
    _, image_id = _insert_face(session_factory)

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    assert dialog._crop_label.text() == dialog.tr("No image")


# ===========================================================================
# Person list
# ===========================================================================


def test_load_persons_populates_list(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """After opening the dialog, the person list contains all persons in DB order."""
    _, image_id = _insert_face(session_factory)
    _insert_person_with_cluster(session_factory, "Alice")
    _insert_person_with_cluster(session_factory, "Bob")

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    assert dialog._person_list.count() == 2
    assert dialog._person_list.item(0).text() == "Alice"
    assert dialog._person_list.item(1).text() == "Bob"


def test_filter_text_preserved_across_face_advances(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Filter text must survive _load_persons() — user should not have to retype."""
    _, image_id = _insert_face(session_factory)
    _insert_person_with_cluster(session_factory, "Alice")
    _insert_person_with_cluster(session_factory, "Bob")

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    dialog._search_edit.setText("ali")
    assert dialog._person_list.count() == 1

    # Simulate advancing (as _load_persons does during face transitions)
    dialog._load_persons()

    # Filter text and filtered list must be intact
    assert dialog._search_edit.text() == "ali"
    assert dialog._person_list.count() == 1
    assert dialog._person_list.item(0).text() == "Alice"


def test_previous_person_selection_restored_after_face_advance(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """The previously selected person is re-selected when _load_persons() is called."""
    _, image_id = _insert_face(session_factory)
    person_id, _ = _insert_person_with_cluster(session_factory, "Alice")
    _insert_person_with_cluster(session_factory, "Bob")

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    dialog._person_list.setCurrentRow(0)
    assert dialog._selected_person is not None
    assert dialog._selected_person.id == person_id

    # Re-load persons (as happens after a face transition)
    dialog._load_persons()

    assert dialog._selected_person is not None
    assert dialog._selected_person.id == person_id


# ===========================================================================
# _on_person_selected / cluster table
# ===========================================================================


def test_on_person_selected_populates_cluster_table(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Selecting a person in the list stores the person and populates cluster data."""
    _, image_id = _insert_face(session_factory)
    person_id, _ = _insert_person_with_cluster(session_factory, "Marc", "adult")

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    assert dialog._person_list.count() == 1
    dialog._person_list.setCurrentRow(0)

    assert dialog._selected_person is not None
    assert dialog._selected_person.id == person_id


def test_on_person_selected_deselect_clears_state(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Deselecting a person clears _selected_person and disables confirm."""
    _, image_id = _insert_face(session_factory)
    _insert_person_with_cluster(session_factory, "Marc")

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    dialog._person_list.setCurrentRow(0)
    assert dialog._selected_person is not None

    dialog._on_person_selected(None, None)
    assert dialog._selected_person is None
    assert dialog._selected_cluster is None
    assert not dialog.confirm_btn.isEnabled()


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

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, img_id)
    qtbot.addWidget(dialog)

    assert dialog._query_embedding is not None

    dialog._person_list.setCurrentRow(0)

    adult_row = AGE_CLUSTERS.index("adult")
    score_item = dialog._cluster_table.item(adult_row, 1)
    assert score_item is not None
    text = score_item.text()
    assert re.fullmatch(
        r"-?\d+%", text
    ), f"Expected numeric score ending with '%', got {text!r}"


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

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, img_id)
    qtbot.addWidget(dialog)

    dialog._person_list.setCurrentRow(0)

    assert dialog._selected_cluster is not None
    assert dialog.confirm_btn.isEnabled()


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

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, img_id)
    qtbot.addWidget(dialog)

    dialog._query_embedding = _ones_norm_emb()

    cluster_by_age = _get_cluster_by_age(session_factory, person_id)
    scores = dialog._compute_cluster_scores(cluster_by_age)

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

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, img_id)
    qtbot.addWidget(dialog)
    dialog._query_embedding = _zero_emb()

    cluster_by_age = _get_cluster_by_age(session_factory, person_id)
    scores = dialog._compute_cluster_scores(cluster_by_age)
    assert scores == {}


def test_compute_cluster_scores_empty_cluster_skipped(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_compute_cluster_scores() skips clusters that have no identified faces."""
    person_id, _ = _insert_person_with_cluster(session_factory, "NF", "adult")
    image_id = _insert_image(session_factory, "/nf.jpg", "nfhash")

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)
    dialog._query_embedding = _ones_norm_emb()

    cluster_by_age = _get_cluster_by_age(session_factory, person_id)
    scores = dialog._compute_cluster_scores(cluster_by_age)
    assert "adult" not in scores


# ===========================================================================
# _create_new_person
# ===========================================================================


def test_create_new_person_rejected_does_nothing(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Cancelling NewPersonDialog leaves the person list unchanged."""
    _, image_id = _insert_face(session_factory)
    _insert_person_with_cluster(session_factory, "Existing")

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    count_before = dialog._person_list.count()

    with patch("photoaident.ui.widgets.labelling_dialog.NewPersonDialog") as MockDlg:
        inst = MockDlg.return_value
        inst.exec.return_value = QtWidgets.QDialog.DialogCode.Rejected
        dialog._create_new_person()

    assert dialog._person_list.count() == count_before


def test_create_new_person_accepted_adds_and_selects(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Accepting NewPersonDialog adds the new person and selects them in the list."""
    _, image_id = _insert_face(session_factory)

    person_id, _ = _insert_person_with_cluster(session_factory, "NewGuy", "adult")

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    with patch("photoaident.ui.widgets.labelling_dialog.NewPersonDialog") as MockDlg:
        inst = MockDlg.return_value
        inst.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
        inst.created_person_id.return_value = person_id
        dialog._create_new_person()

    found = False
    for i in range(dialog._person_list.count()):
        item = dialog._person_list.item(i)
        if item and item.data(QtCore.Qt.ItemDataRole.UserRole).id == person_id:
            found = True
            break
    assert found


def test_create_new_person_accepted_none_id_does_nothing(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """If created_person_id() returns None, _create_new_person does nothing."""
    _, image_id = _insert_face(session_factory)

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    with patch("photoaident.ui.widgets.labelling_dialog.NewPersonDialog") as MockDlg:
        inst = MockDlg.return_value
        inst.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
        inst.created_person_id.return_value = None
        dialog._create_new_person()

    assert dialog._person_list.count() == 0


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

    face_id, image_id = _insert_face(session_factory)

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    assert dialog._current_face_id == face_id

    with session_factory() as session:
        person = session.get(Person, person_id)
        cluster = session.get(EmbeddingCluster, cluster_id)
        session.expunge_all()

    dialog._selected_person = person
    dialog._selected_cluster = cluster
    dialog._update_confirm_button()

    assert dialog.confirm_btn.isEnabled()
    dialog._on_confirm()

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
    image_id = _insert_image(session_factory, "/noop.jpg", "noophash")
    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    assert dialog._current_face_id is None
    dialog._on_confirm()  # must not raise
    assert dialog._current_face_id is None


def test_on_confirm_noop_when_no_selection(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_on_confirm does nothing when no person/cluster is selected."""
    face_id, image_id = _insert_face(session_factory)

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    assert dialog._current_face_id == face_id
    assert dialog._selected_person is None

    dialog._on_confirm()  # must not raise, face state unchanged

    with session_factory() as session:
        face = session.get(Face, face_id)
        assert face is not None
        assert face.state == FaceState.UNIDENTIFIED


def test_confirm_disabled_without_selection(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """confirm_btn is disabled when no person+cluster is selected."""
    _, image_id = _insert_face(session_factory)

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    assert not dialog.confirm_btn.isEnabled()

    dialog._selected_person = Person(name="X")
    dialog._selected_cluster = None
    dialog._update_confirm_button()
    assert not dialog.confirm_btn.isEnabled()

    dialog._selected_cluster = EmbeddingCluster()
    dialog._update_confirm_button()
    assert dialog.confirm_btn.isEnabled()


def test_cancel_clears_selection(qtbot, session_factory, tmp_app_paths, vector_store):
    """_on_cancel clears person/cluster selection without changing face or advancing."""
    face_id, image_id = _insert_face(session_factory)

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    dialog._selected_person = Person(name="Someone")
    dialog._selected_cluster = EmbeddingCluster()
    dialog._update_confirm_button()
    assert dialog.confirm_btn.isEnabled()

    dialog._on_cancel()

    assert dialog._selected_person is None
    assert dialog._selected_cluster is None
    assert not dialog.confirm_btn.isEnabled()
    assert dialog._current_face_id == face_id


# ===========================================================================
# Mark anonymous
# ===========================================================================


def test_mark_anonymous(qtbot, session_factory, tmp_app_paths, vector_store):
    """Clicking Mark Anonymous sets face.state = ANONYMOUS and advances."""
    face_id, image_id = _insert_face(session_factory)

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    assert dialog._current_face_id == face_id
    dialog._mark_anonymous()

    with session_factory() as session:
        face = session.get(Face, face_id)
        assert face is not None
        assert face.state == FaceState.ANONYMOUS
        assert face.labelled_at is not None

    assert not dialog.confirm_btn.isEnabled()
    assert dialog._current_face_id is None


def test_mark_anonymous_noop_when_no_current_face(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_mark_anonymous does nothing when _current_face_id is None."""
    image_id = _insert_image(session_factory, "/anonnoop.jpg", "anonnoop")
    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    assert dialog._current_face_id is None
    dialog._mark_anonymous()
    assert dialog._current_face_id is None


# ===========================================================================
# Skip face
# ===========================================================================


def test_skip_face(qtbot, session_factory, tmp_app_paths, vector_store):
    """Skip does not change face state. When the only face is skipped, buttons
    become disabled (completion state)."""
    face_id, image_id = _insert_face(session_factory)

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    assert dialog._current_face_id == face_id
    dialog._skip_face()

    with session_factory() as session:
        face = session.get(Face, face_id)
        assert face is not None
        assert face.state == FaceState.UNIDENTIFIED
        assert face.labelled_at is None

    assert not dialog.skip_btn.isEnabled()


def test_skip_face_noop_when_no_current_face(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_skip_face does nothing when _current_face_id is None."""
    image_id = _insert_image(session_factory, "/skipnoop.jpg", "skipnoop")
    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    assert dialog._current_face_id is None
    dialog._skip_face()
    assert dialog._current_face_id is None


def test_skip_face_advances_to_next(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Skip advances to the next face in the image without removing it."""
    image_id = _insert_image(session_factory, "/skipadv.jpg", "skipadvhash")
    face1_id = _insert_face_for_image(session_factory, image_id)
    face2_id = _insert_face_for_image(session_factory, image_id)

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    # First face shown (ordered by bbox_x, bbox_y — both at 10,10 so stable by insert)
    first_shown = dialog._current_face_id
    assert first_shown in (face1_id, face2_id)

    dialog._skip_face()

    # After skip, the other face is shown and skip button still enabled
    assert dialog._current_face_id != first_shown
    assert dialog._current_face_id in (face1_id, face2_id)
    assert dialog.skip_btn.isEnabled()


# ===========================================================================
# Auto-advance after confirm / anonymous
# ===========================================================================


def test_auto_advance_after_confirm(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """After confirming face 1, the dialog auto-advances to face 2."""
    with session_factory() as session:
        p = Person(name="Multi")
        session.add(p)
        session.flush()
        c = EmbeddingCluster(person_id=p.id)
        session.add(c)
        session.commit()
        person_id = p.id
        cluster_id = c.id

    image_id = _insert_image(session_factory, "/advance.jpg", "advancehash")
    face1_id = _insert_face_for_image(session_factory, image_id)
    face2_id = _insert_face_for_image(session_factory, image_id)

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    with session_factory() as session:
        person = session.get(Person, person_id)
        _ = list(person.clusters)  # force-load clusters before detaching
        cluster = session.get(EmbeddingCluster, cluster_id)
        session.expunge_all()

    first_face_id = dialog._current_face_id
    assert first_face_id in (face1_id, face2_id)

    dialog._selected_person = person
    dialog._selected_cluster = cluster
    dialog._on_confirm()

    # Must advance to the other face
    assert dialog._current_face_id is not None
    assert dialog._current_face_id != first_face_id
    assert dialog._current_face_id in (face1_id, face2_id)


def test_auto_advance_after_anonymous(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """After marking face 1 anonymous, the dialog auto-advances to face 2."""
    image_id = _insert_image(session_factory, "/anon2.jpg", "anon2hash")
    face1_id = _insert_face_for_image(session_factory, image_id)
    face2_id = _insert_face_for_image(session_factory, image_id)

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    first_face_id = dialog._current_face_id
    assert first_face_id in (face1_id, face2_id)

    dialog._mark_anonymous()

    assert dialog._current_face_id is not None
    assert dialog._current_face_id != first_face_id
    assert dialog._current_face_id in (face1_id, face2_id)


def test_completion_state_when_all_labelled(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """When all faces have been labelled, completion state is shown."""
    _, image_id = _insert_face(session_factory)

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    dialog._mark_anonymous()

    # All faces labelled → completion state
    assert dialog._current_face_id is None
    assert not dialog.skip_btn.isEnabled()
    assert not dialog.anonymous_btn.isEnabled()
    assert "done" in dialog._crop_label.text().lower()


# ===========================================================================
# Progress label
# ===========================================================================


def test_progress_label_shows_face_count(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """The face info label shows 'Face X of Y' progress indicator."""
    image_id = _insert_image(session_factory, "/prog.jpg", "proghash")
    _insert_face_for_image(session_factory, image_id)
    _insert_face_for_image(session_factory, image_id)

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    label_text = dialog._face_info_label.text()
    assert "1" in label_text  # "Face 1 of 2"
    assert "2" in label_text


# ===========================================================================
# Edge-case coverage for remaining branches
# ===========================================================================


def test_compute_cluster_scores_null_mean_skipped(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Cluster with mean_embedding=None is skipped in score computation."""
    person_id, _ = _insert_person_with_cluster(session_factory, "Exc", "adult")
    image_id = _insert_image(session_factory, "/exc.jpg", "exchash")

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)
    dialog._query_embedding = _ones_norm_emb()

    cluster_by_age = _get_cluster_by_age(session_factory, person_id)
    scores = dialog._compute_cluster_scores(cluster_by_age)

    assert "adult" not in scores


def test_compute_cluster_scores_with_persisted_mean(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_compute_cluster_scores reads the persisted mean and computes a dot product."""
    person_id, cluster_id = _insert_person_with_cluster(session_factory, "ZNM", "adult")
    image_id = _insert_image(session_factory, "/znm.jpg", "znmhash")

    mean = _unit_emb(0)
    with session_factory() as session:
        cluster = session.get(EmbeddingCluster, cluster_id)
        cluster.mean_embedding = serialize_embedding(mean)
        session.commit()

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)
    dialog._query_embedding = mean.copy()

    cluster_by_age = _get_cluster_by_age(session_factory, person_id)
    scores = dialog._compute_cluster_scores(cluster_by_age)

    assert "adult" in scores
    assert scores["adult"] == pytest.approx(1.0, abs=1e-6)


def _insert_face_with_taken_at(
    session_factory, taken_at: datetime, file_path: str = "/path/to/img.jpg"
) -> tuple[int, int]:
    """Insert an unidentified face with a taken_at timestamp.

    Returns (face_id, image_id).
    """
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
        return face.id, img.id


# ===========================================================================
# _face_info_label content
# ===========================================================================


def test_face_info_label_shows_path_and_taken_at(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_face_info_label shows the image file path and formatted taken_at date."""
    taken_at = datetime(2023, 6, 15, 10, 30, tzinfo=timezone.utc)
    _, image_id = _insert_face_with_taken_at(
        session_factory, taken_at, file_path="/photos/summer.jpg"
    )

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    label_text = dialog._face_info_label.text()
    assert "/photos/summer.jpg" in label_text
    assert "2023-06-15" in label_text
    assert "10:30" in label_text


def test_face_info_label_shows_unknown_date_when_no_taken_at(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_face_info_label shows 'Unknown date' when image metadata has no taken_at."""
    _, image_id = _insert_face(session_factory, file_path="/photos/nodatephoto.jpg")

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    label_text = dialog._face_info_label.text()
    assert "/photos/nodatephoto.jpg" in label_text
    assert "Unknown date" in label_text


def test_face_info_label_cleared_in_empty_state(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """_face_info_label is cleared when the dialog has no unidentified faces."""
    image_id = _insert_image(session_factory, "/empty2.jpg", "empty2hash")
    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    assert dialog._face_info_label.text() == ""


# ===========================================================================
# Auto-preselect best person + cluster
# ===========================================================================


def test_preselect_best_person_and_cluster_on_open(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """Opening the dialog auto-selects the person whose cluster mean best matches
    the face embedding, and pre-selects the best-scoring age cluster, so Confirm
    is enabled without any user interaction."""
    # Create an unidentified face whose embedding points along axis 1 (unit_emb(1))
    img_id = _insert_image(session_factory, "/presel.jpg", "preselhash")
    face_emb = _unit_emb(1)
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
        vector_store.add(face.id, face_emb)
        session.commit()

    # Person A — cluster mean along axis 0 (orthogonal to face_emb)
    _, cluster_a_id = _insert_person_with_cluster(session_factory, "Alice", "adult")
    with session_factory() as session:
        ca = session.get(EmbeddingCluster, cluster_a_id)
        ca.mean_embedding = serialize_embedding(_unit_emb(0))
        session.commit()

    # Person B — cluster mean along axis 1 (identical direction to face_emb → score 1.0)
    person_b_id, cluster_b_id = _insert_person_with_cluster(
        session_factory, "Bob", "adult"
    )
    with session_factory() as session:
        cb = session.get(EmbeddingCluster, cluster_b_id)
        cb.mean_embedding = serialize_embedding(_unit_emb(1))
        session.commit()

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, img_id)
    qtbot.addWidget(dialog)

    # Bob should be auto-selected because his cluster mean is closest to face_emb
    assert dialog._selected_person is not None
    assert dialog._selected_person.id == person_b_id
    # Best-matching cluster (adult) should also be selected, enabling Confirm
    assert dialog._selected_cluster is not None
    assert dialog.confirm_btn.isEnabled()


def test_no_preselection_when_no_cluster_means(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """When no cluster means have been persisted yet, the dialog opens with no
    person or cluster pre-selected (existing fall-through behaviour)."""
    face_id, image_id = _insert_face(session_factory)
    vector_store.add(face_id, _ones_norm_emb())

    # Person exists but has no identified faces → mean_embedding is NULL
    _insert_person_with_cluster(session_factory, "Zara", "adult")

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    assert dialog._selected_person is None
    assert dialog._selected_cluster is None
    assert not dialog.confirm_btn.isEnabled()


def test_preselect_recomputed_on_face_advance(
    qtbot, session_factory, tmp_app_paths, vector_store
):
    """After advancing to the next face, preselection is re-evaluated for that
    face's embedding, not the previous face's."""
    image_id = _insert_image(session_factory, "/advpre.jpg", "advprehash")

    # Face 1 — embedding along axis 0; shown first (lower bbox_x)
    with session_factory() as session:
        face1 = Face(
            image_id=image_id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=40,
            bbox_h=40,
            detection_confidence=0.9,
            state=FaceState.UNIDENTIFIED,
            model_version="test",
        )
        session.add(face1)
        session.flush()
        face1_id = face1.id
        vector_store.add(face1_id, _unit_emb(0))
        session.commit()

    # Face 2 — embedding along axis 1; shown second (higher bbox_x)
    with session_factory() as session:
        face2 = Face(
            image_id=image_id,
            bbox_x=50,
            bbox_y=0,
            bbox_w=40,
            bbox_h=40,
            detection_confidence=0.9,
            state=FaceState.UNIDENTIFIED,
            model_version="test",
        )
        session.add(face2)
        session.flush()
        face2_id = face2.id
        vector_store.add(face2_id, _unit_emb(1))
        session.commit()

    # Person B — cluster mean along axis 1, only close to face 2
    person_b_id, cluster_b_id = _insert_person_with_cluster(
        session_factory, "Bob2", "adult"
    )
    with session_factory() as session:
        cb = session.get(EmbeddingCluster, cluster_b_id)
        cb.mean_embedding = serialize_embedding(_unit_emb(1))
        session.commit()

    dialog = _make_dialog(session_factory, tmp_app_paths, vector_store, image_id)
    qtbot.addWidget(dialog)

    # Face 1 is shown first; no cluster mean is close (axis 0 vs axis 1 = score 0)
    # But B is still the best available option, so it gets preselected
    assert dialog._current_face_id == face1_id

    # Mark face 1 anonymous → dialog advances to face 2
    dialog._mark_anonymous()

    assert dialog._current_face_id == face2_id
    # Face 2 embedding aligns perfectly with Bob2's cluster mean
    assert dialog._selected_person is not None
    assert dialog._selected_person.id == person_b_id
    assert dialog._selected_cluster is not None
    assert dialog.confirm_btn.isEnabled()
