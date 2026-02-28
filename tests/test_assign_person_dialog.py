from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PySide6 import QtWidgets
from sqlalchemy import select

from photoaident.db.database import (
    AGE_CLUSTERS,
    EmbeddingCluster,
    Face,
    FaceState,
    Image,
    Person,
    get_engine,
    get_session_factory,
)
from photoaident.db.migrate import apply_migrations
from photoaident.db.vector_store import VectorStore
from photoaident.ui.widgets.assign_person_dialog import AssignPersonDialog

# ── helpers ──────────────────────────────────────────────────────────────────

_COL_SCORE = 2


@pytest.fixture
def session_factory(tmp_path):
    db_path = tmp_path / "test.db"
    engine = get_engine(str(db_path))
    apply_migrations(f"sqlite:///{db_path}")
    return get_session_factory(engine)


@pytest.fixture
def vector_store():
    return VectorStore()


def _add_person_with_age_clusters(session_factory, name: str) -> int:
    """Insert a person with all 5 age clusters and return the person id."""
    with session_factory() as session:
        person = Person(name=name)
        session.add(person)
        session.flush()
        for age_key in AGE_CLUSTERS:
            session.add(
                EmbeddingCluster(person_id=person.id, label=age_key, age_group=age_key)
            )
        session.commit()
        return person.id


def _add_identified_face_to_cluster(
    session_factory, vector_store: VectorStore, cluster_id: int, embedding: np.ndarray
) -> int:
    """Insert an identified face with given embedding into a cluster; return face id."""
    faiss_id = vector_store.add(embedding)
    with session_factory() as session:
        img = Image(file_path=f"/img_{faiss_id}.jpg", file_size=1000)
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
            state=FaceState.IDENTIFIED,
            model_version="test",
            cluster_id=cluster_id,
        )
        session.add(face)
        session.commit()
        return face.id


def _get_cluster_id_for_age(session_factory, person_id: int, age_key: str) -> int:
    """Return the cluster id for a given person and age_group key."""
    with session_factory() as session:
        cluster = session.execute(
            select(EmbeddingCluster).where(
                EmbeddingCluster.person_id == person_id,
                EmbeddingCluster.age_group == age_key,
            )
        ).scalar_one()
        return cluster.id


# ── tests ─────────────────────────────────────────────────────────────────────


def test_existing_persons_listed(qtbot, session_factory):
    """Persons in the DB appear in the person list widget."""
    _add_person_with_age_clusters(session_factory, "Alice")
    _add_person_with_age_clusters(session_factory, "Bob")

    dialog = AssignPersonDialog(session_factory)
    qtbot.addWidget(dialog)

    assert dialog.person_list.count() == 2
    names = {
        dialog.person_list.item(i).text() for i in range(dialog.person_list.count())
    }
    assert "Alice" in names
    assert "Bob" in names


def test_search_filters_persons(qtbot, session_factory):
    """Typing in the search field filters the person list in real-time."""
    _add_person_with_age_clusters(session_factory, "Alice")
    _add_person_with_age_clusters(session_factory, "Alicia")
    _add_person_with_age_clusters(session_factory, "Bob")

    dialog = AssignPersonDialog(session_factory)
    qtbot.addWidget(dialog)

    dialog.search_edit.setText("ali")

    assert dialog.person_list.count() == 2
    names = {
        dialog.person_list.item(i).text() for i in range(dialog.person_list.count())
    }
    assert "Alice" in names
    assert "Alicia" in names
    assert "Bob" not in names


def test_create_new_person(qtbot, session_factory):
    """'New person…' opens NewPersonDialog; on accept, person appears in the list."""
    dialog = AssignPersonDialog(session_factory)
    qtbot.addWidget(dialog)

    assert dialog.person_list.count() == 0

    # Create the person directly so we have a real id to return
    with session_factory() as session:
        person = Person(name="Charlie")
        session.add(person)
        session.flush()
        new_person_id = person.id
        for age_key in AGE_CLUSTERS:
            session.add(
                EmbeddingCluster(
                    person_id=new_person_id, label=age_key, age_group=age_key
                )
            )
        session.commit()

    mock_dlg = MagicMock()
    mock_dlg.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
    mock_dlg.created_person_id.return_value = new_person_id

    with patch(
        "photoaident.ui.widgets.assign_person_dialog.NewPersonDialog",
        return_value=mock_dlg,
    ):
        dialog._create_new_person()

    assert dialog.person_list.count() == 1
    assert dialog.person_list.item(0).text() == "Charlie"

    # Verify persistence in the DB: 1 person with 5 age clusters
    with session_factory() as session:
        persons = session.execute(select(Person)).scalars().all()
        assert len(persons) == 1
        assert persons[0].name == "Charlie"
        clusters = (
            session.execute(
                select(EmbeddingCluster).where(
                    EmbeddingCluster.person_id == persons[0].id
                )
            )
            .scalars()
            .all()
        )
        assert len(clusters) == len(AGE_CLUSTERS)
        age_groups = {c.age_group for c in clusters}
        assert age_groups == set(AGE_CLUSTERS)


def test_cluster_group_shows_5_rows(qtbot, session_factory):
    """After selecting a person, the cluster table always has 5 rows."""
    _add_person_with_age_clusters(session_factory, "Dave")

    dialog = AssignPersonDialog(session_factory)
    qtbot.addWidget(dialog)

    dialog.person_list.setCurrentRow(0)

    assert not dialog.cluster_group.isHidden()
    assert dialog.cluster_table.rowCount() == 5


def test_ok_disabled_until_cluster_row_selected(qtbot, session_factory):
    """OK button is disabled until a cluster row is selected."""
    _add_person_with_age_clusters(session_factory, "Eve")

    dialog = AssignPersonDialog(session_factory)
    qtbot.addWidget(dialog)

    dialog.person_list.setCurrentRow(0)

    ok_btn = dialog.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    assert ok_btn is not None

    # No row pre-selected (no embedding provided) → OK disabled
    dialog.cluster_table.clearSelection()
    # Trigger selection-changed manually (clearSelection may not fire when empty)
    dialog._on_cluster_row_selected()
    assert not ok_btn.isEnabled()

    # Selecting a row enables OK
    dialog.cluster_table.selectRow(3)  # adult row
    assert ok_btn.isEnabled()


def test_cluster_group_hidden_before_person_selected(qtbot, session_factory):
    """Cluster group is hidden until a person is selected."""
    _add_person_with_age_clusters(session_factory, "Frank")

    dialog = AssignPersonDialog(session_factory)
    qtbot.addWidget(dialog)

    # Before any selection
    assert dialog.cluster_group.isHidden()

    dialog.person_list.setCurrentRow(0)
    assert not dialog.cluster_group.isHidden()


def test_deselect_person_clears_selection(qtbot, session_factory):
    """Calling _on_person_selected(None, …) clears person + cluster selection."""
    _add_person_with_age_clusters(session_factory, "Heidi")

    dialog = AssignPersonDialog(session_factory)
    qtbot.addWidget(dialog)

    dialog.person_list.setCurrentRow(0)
    assert dialog._selected_person is not None

    dialog._on_person_selected(None, None)

    assert dialog._selected_person is None
    assert dialog._selected_cluster is None
    assert dialog.cluster_group.isHidden()
    ok_btn = dialog.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    assert ok_btn is not None
    assert not ok_btn.isEnabled()


def test_deselect_cluster_clears_cluster_selection(qtbot, session_factory):
    """Clearing the cluster table selection resets _selected_cluster."""
    _add_person_with_age_clusters(session_factory, "Jack")

    dialog = AssignPersonDialog(session_factory)
    qtbot.addWidget(dialog)

    dialog.person_list.setCurrentRow(0)
    dialog.cluster_table.selectRow(0)
    assert dialog._selected_cluster is not None

    dialog.cluster_table.clearSelection()
    dialog._on_cluster_row_selected()

    assert dialog._selected_cluster is None
    ok_btn = dialog.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    assert ok_btn is not None
    assert not ok_btn.isEnabled()


def test_create_new_person_cancelled(qtbot, session_factory):
    """Cancelling NewPersonDialog does not create a person."""
    dialog = AssignPersonDialog(session_factory)
    qtbot.addWidget(dialog)

    mock_dlg = MagicMock()
    mock_dlg.exec.return_value = QtWidgets.QDialog.DialogCode.Rejected

    with patch(
        "photoaident.ui.widgets.assign_person_dialog.NewPersonDialog",
        return_value=mock_dlg,
    ):
        dialog._create_new_person()

    assert dialog.person_list.count() == 0


def test_create_new_person_none_id_is_noop(qtbot, session_factory):
    """If NewPersonDialog returns None from created_person_id(), nothing is added."""
    dialog = AssignPersonDialog(session_factory)
    qtbot.addWidget(dialog)

    mock_dlg = MagicMock()
    mock_dlg.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
    mock_dlg.created_person_id.return_value = None

    with patch(
        "photoaident.ui.widgets.assign_person_dialog.NewPersonDialog",
        return_value=mock_dlg,
    ):
        dialog._create_new_person()

    assert dialog.person_list.count() == 0


def test_result_person_cluster_returns_none_without_selection(qtbot, session_factory):
    """result_person_cluster returns None when nothing is selected."""
    dialog = AssignPersonDialog(session_factory)
    qtbot.addWidget(dialog)

    assert dialog.result_person_cluster() is None


def test_no_preselection_without_embedding(qtbot, session_factory):
    """Without a query_embedding no row is pre-selected after person selection."""
    _add_person_with_age_clusters(session_factory, "NoEmbed")

    dialog = AssignPersonDialog(session_factory)
    qtbot.addWidget(dialog)

    dialog.person_list.setCurrentRow(0)

    selected = dialog.cluster_table.selectedItems()
    assert len(selected) == 0
    assert dialog._selected_cluster is None


def test_cluster_preselection_with_embedding(qtbot, session_factory, vector_store):
    """Cluster whose mean embedding is closest to query gets pre-selected."""
    person_id = _add_person_with_age_clusters(session_factory, "EmbedPerson")

    # Add an identified face to the "adult" cluster
    adult_cluster_id = _get_cluster_id_for_age(session_factory, person_id, "adult")

    # Create a distinctive adult embedding and a different query embedding close to it
    adult_emb = np.zeros(512, dtype=np.float32)
    adult_emb[0] = 1.0  # unit vector in dimension 0
    _add_identified_face_to_cluster(
        session_factory, vector_store, adult_cluster_id, adult_emb
    )

    # Query embedding is very close to adult cluster
    query_emb = adult_emb.copy()

    dialog = AssignPersonDialog(
        session_factory, query_embedding=query_emb, vector_store=vector_store
    )
    qtbot.addWidget(dialog)

    dialog.person_list.setCurrentRow(0)

    # Adult is row index 3 (infant=0, youngster=1, teenager=2, adult=3, senior=4)
    assert dialog.cluster_table.currentRow() == 3
    assert dialog._selected_cluster is not None
    assert dialog._selected_cluster.age_group == "adult"


def test_similarity_scores_displayed(qtbot, session_factory, vector_store):
    """Clusters with identified faces show numeric scores; empty clusters show '—'."""
    person_id = _add_person_with_age_clusters(session_factory, "ScorePerson")

    infant_cluster_id = _get_cluster_id_for_age(session_factory, person_id, "infant")
    infant_emb = np.ones(512, dtype=np.float32)
    infant_emb /= np.linalg.norm(infant_emb)
    _add_identified_face_to_cluster(
        session_factory, vector_store, infant_cluster_id, infant_emb
    )

    query_emb = infant_emb.copy()
    dialog = AssignPersonDialog(
        session_factory, query_embedding=query_emb, vector_store=vector_store
    )
    qtbot.addWidget(dialog)

    dialog.person_list.setCurrentRow(0)

    # Infant row (row 0) should have a numeric score
    infant_item = dialog.cluster_table.item(0, _COL_SCORE)
    assert infant_item is not None
    infant_score_text = infant_item.text()
    assert infant_score_text != "\u2014"
    float(infant_score_text)  # must be parseable as float

    # Remaining rows should show '—' (no identified faces)
    for row in range(1, 5):
        score_item = dialog.cluster_table.item(row, _COL_SCORE)
        assert score_item is not None
        score_text = score_item.text()
        assert score_text == "\u2014", f"Row {row} expected '—', got {score_text!r}"
