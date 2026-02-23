from unittest.mock import patch

import pytest
from PySide6 import QtWidgets
from sqlalchemy import select

from photoaident.db.database import (
    EmbeddingCluster,
    Person,
    get_engine,
    get_session_factory,
)
from photoaident.db.migrate import apply_migrations
from photoaident.ui.widgets.assign_person_dialog import AssignPersonDialog

# ── helpers ──────────────────────────────────────────────────────────────────


def _person_cluster_counts(session_factory, person_id):
    with session_factory() as session:
        clusters = (
            session.execute(
                select(EmbeddingCluster).where(EmbeddingCluster.person_id == person_id)
            )
            .scalars()
            .all()
        )
        return len(clusters)


@pytest.fixture
def session_factory(tmp_path):
    db_path = tmp_path / "test.db"
    engine = get_engine(str(db_path))
    apply_migrations(f"sqlite:///{db_path}")
    return get_session_factory(engine)


def _add_person(
    session_factory,
    name: str,
    cluster_labels: list[str | None] | None = None,
) -> int:
    """Insert a person with clusters and return the person id."""
    if cluster_labels is None:
        cluster_labels = [None]  # one unlabelled cluster by default
    with session_factory() as session:
        person = Person(name=name)
        session.add(person)
        session.flush()
        for label in cluster_labels:
            session.add(EmbeddingCluster(person_id=person.id, label=label))
        session.commit()
        return person.id


def test_existing_persons_listed(qtbot, session_factory):
    """Persons in the DB appear in the person list widget."""
    _add_person(session_factory, "Alice")
    _add_person(session_factory, "Bob")

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
    _add_person(session_factory, "Alice")
    _add_person(session_factory, "Alicia")
    _add_person(session_factory, "Bob")

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
    """'New person…' creates a Person and a default unlabelled EmbeddingCluster."""
    dialog = AssignPersonDialog(session_factory)
    qtbot.addWidget(dialog)

    assert dialog.person_list.count() == 0

    with patch.object(
        QtWidgets.QInputDialog, "getText", return_value=("Charlie", True)
    ):
        dialog._create_new_person()

    assert dialog.person_list.count() == 1
    assert dialog.person_list.item(0).text() == "Charlie"

    # Verify persistence in the DB
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
        assert len(clusters) == 1


def test_single_cluster_auto_selected(qtbot, session_factory):
    """If a person has exactly one cluster, it is auto-selected and OK is enabled."""
    _add_person(session_factory, "Dave", cluster_labels=[None])

    dialog = AssignPersonDialog(session_factory)
    qtbot.addWidget(dialog)

    dialog.person_list.setCurrentRow(0)

    # Cluster group should be hidden (single cluster auto-selected)
    assert dialog.cluster_group.isHidden()

    ok_btn = dialog.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    assert ok_btn is not None
    assert ok_btn.isEnabled()


def test_multiple_clusters_require_selection(qtbot, session_factory):
    """If a person has multiple clusters, cluster group is visible and OK disabled."""
    _add_person(session_factory, "Eve", cluster_labels=["childhood", "adult"])

    dialog = AssignPersonDialog(session_factory)
    qtbot.addWidget(dialog)

    dialog.person_list.setCurrentRow(0)

    # Cluster group must be explicitly set visible (not hidden)
    assert not dialog.cluster_group.isHidden()

    ok_btn = dialog.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    assert ok_btn is not None
    assert not ok_btn.isEnabled()

    # Selecting a cluster enables OK
    dialog.cluster_list.setCurrentRow(0)
    assert ok_btn.isEnabled()


def test_deselect_person_clears_selection(qtbot, session_factory):
    """Calling _on_person_selected(None, …) clears person + cluster selection."""
    _add_person(session_factory, "Heidi", cluster_labels=[None])

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


def test_person_with_no_clusters_hides_cluster_group(qtbot, session_factory):
    """Person with zero clusters shows cluster group as hidden (else branch)."""
    with session_factory() as session:
        person = Person(name="Ivan")
        session.add(person)
        session.commit()

    dialog = AssignPersonDialog(session_factory)
    qtbot.addWidget(dialog)

    dialog.person_list.setCurrentRow(0)

    assert dialog.cluster_group.isHidden()
    assert dialog._selected_cluster is None


def test_deselect_cluster_clears_cluster_selection(qtbot, session_factory):
    """Calling _on_cluster_selected(None, …) resets _selected_cluster."""
    _add_person(session_factory, "Jack", cluster_labels=["childhood", "adult"])

    dialog = AssignPersonDialog(session_factory)
    qtbot.addWidget(dialog)

    dialog.person_list.setCurrentRow(0)
    dialog.cluster_list.setCurrentRow(0)
    assert dialog._selected_cluster is not None

    dialog._on_cluster_selected(None, None)

    assert dialog._selected_cluster is None
    ok_btn = dialog.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    assert ok_btn is not None
    assert not ok_btn.isEnabled()


def test_create_new_person_cancelled(qtbot, session_factory):
    """Cancelling the input dialog does not create a person."""
    dialog = AssignPersonDialog(session_factory)
    qtbot.addWidget(dialog)

    with patch.object(QtWidgets.QInputDialog, "getText", return_value=("", False)):
        dialog._create_new_person()

    assert dialog.person_list.count() == 0


def test_create_new_person_empty_name_is_noop(qtbot, session_factory):
    """Accepting with whitespace-only name does not create a person."""
    dialog = AssignPersonDialog(session_factory)
    qtbot.addWidget(dialog)

    with patch.object(QtWidgets.QInputDialog, "getText", return_value=("   ", True)):
        dialog._create_new_person()

    assert dialog.person_list.count() == 0


def test_create_new_cluster(qtbot, session_factory):
    """Creating a new cluster adds it to the DB and updates the cluster list."""
    _add_person(session_factory, "Karen", cluster_labels=[None])

    dialog = AssignPersonDialog(session_factory)
    qtbot.addWidget(dialog)

    dialog.person_list.setCurrentRow(0)
    person_id = dialog._selected_person.id  # type: ignore[union-attr]

    with patch.object(QtWidgets.QInputDialog, "getText", return_value=("adult", True)):
        dialog._create_new_cluster()

    assert _person_cluster_counts(session_factory, person_id) == 2
    assert dialog.cluster_list.count() == 2
    assert not dialog.cluster_group.isHidden()


def test_create_new_cluster_cancelled(qtbot, session_factory):
    """Cancelling the new-cluster dialog leaves the DB unchanged."""
    _add_person(session_factory, "Leo", cluster_labels=[None])

    dialog = AssignPersonDialog(session_factory)
    qtbot.addWidget(dialog)

    dialog.person_list.setCurrentRow(0)
    person_id = dialog._selected_person.id  # type: ignore[union-attr]

    with patch.object(QtWidgets.QInputDialog, "getText", return_value=("", False)):
        dialog._create_new_cluster()

    assert _person_cluster_counts(session_factory, person_id) == 1


def test_result_person_cluster_returns_none_without_selection(qtbot, session_factory):
    """result_person_cluster returns None when nothing is selected."""
    dialog = AssignPersonDialog(session_factory)
    qtbot.addWidget(dialog)

    assert dialog.result_person_cluster() is None
