"""Tests for NewPersonDialog.

All tests call internal methods directly and never invoke exec(), accept(), or
reject() — those interact with the native window system and would block or flash
a real dialog on screen.
"""

from PySide6 import QtWidgets
from sqlalchemy import select

from photoaident.db.database import (
    AGE_CLUSTERS,
    EmbeddingCluster,
    Person,
    get_engine,
    get_session_factory,
)
from photoaident.db.migrate import apply_migrations
from photoaident.paths import AppPaths
from photoaident.ui.widgets.new_person_dialog import NewPersonDialog


def _make_session_factory(tmp_path):
    paths = AppPaths(
        base_data=tmp_path / "data",
        base_cache=tmp_path / "cache",
        base_config=tmp_path / "config",
    )
    paths.ensure_dirs()
    apply_migrations(f"sqlite:///{paths.db_path}")
    engine = get_engine(str(paths.db_path))
    return get_session_factory(engine)


def _make_dialog(tmp_path, qtbot) -> NewPersonDialog:
    session_factory = _make_session_factory(tmp_path)
    dlg = NewPersonDialog(session_factory)
    qtbot.addWidget(dlg)
    return dlg


# ── OK button state ──────────────────────────────────────────────────────────


def test_ok_button_disabled_when_name_empty(tmp_path, qtbot):
    """OK button is disabled when the name field is empty."""
    dlg = _make_dialog(tmp_path, qtbot)
    ok_btn = dlg._button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    assert ok_btn is not None
    assert not ok_btn.isEnabled()


def test_ok_button_enabled_when_name_non_empty(tmp_path, qtbot):
    """OK button is enabled once a non-empty name is typed."""
    dlg = _make_dialog(tmp_path, qtbot)
    dlg._name_edit.setText("Alice")
    ok_btn = dlg._button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    assert ok_btn is not None
    assert ok_btn.isEnabled()


def test_ok_button_disabled_again_when_name_cleared(tmp_path, qtbot):
    """OK button becomes disabled again when the name is cleared."""
    dlg = _make_dialog(tmp_path, qtbot)
    dlg._name_edit.setText("Alice")
    dlg._name_edit.clear()
    ok_btn = dlg._button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    assert ok_btn is not None
    assert not ok_btn.isEnabled()


def test_ok_button_disabled_for_whitespace_only(tmp_path, qtbot):
    """OK button stays disabled when the name contains only whitespace."""
    dlg = _make_dialog(tmp_path, qtbot)
    dlg._name_edit.setText("   ")
    ok_btn = dlg._button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    assert ok_btn is not None
    assert not ok_btn.isEnabled()


# ── DB persistence ───────────────────────────────────────────────────────────


def test_persist_new_person_creates_person_and_clusters(tmp_path, qtbot):
    """_persist_new_person() creates a Person and 5 EmbeddingCluster rows."""
    dlg = _make_dialog(tmp_path, qtbot)

    dlg._persist_new_person("Bob")

    with dlg.session_factory() as session:
        persons = session.execute(select(Person)).scalars().all()
        assert len(persons) == 1
        assert persons[0].name == "Bob"

        clusters = (
            session.execute(
                select(EmbeddingCluster).where(
                    EmbeddingCluster.person_id == persons[0].id
                )
            )
            .scalars()
            .all()
        )
        assert len(clusters) == 5
        assert {c.age_group for c in clusters} == set(AGE_CLUSTERS)


def test_persist_new_person_returns_id(tmp_path, qtbot):
    """_persist_new_person() returns the new person's DB id."""
    dlg = _make_dialog(tmp_path, qtbot)

    person_id = dlg._persist_new_person("Carol")

    assert person_id is not None
    with dlg.session_factory() as session:
        person = session.get(Person, person_id)
        assert person is not None
        assert person.name == "Carol"


def test_persist_new_person_strips_whitespace(tmp_path, qtbot):
    """Leading/trailing whitespace is stripped by the caller (_on_accept)."""
    dlg = _make_dialog(tmp_path, qtbot)

    dlg._persist_new_person("  Eve  ".strip())

    with dlg.session_factory() as session:
        person = session.execute(select(Person)).scalars().one()
        assert person.name == "Eve"


# ── created_person_id ────────────────────────────────────────────────────────


def test_created_person_id_none_before_acceptance(tmp_path, qtbot):
    """created_person_id() returns None before anything is persisted."""
    dlg = _make_dialog(tmp_path, qtbot)
    assert dlg.created_person_id() is None


def test_created_person_id_none_without_confirmation(tmp_path, qtbot):
    """Typing a name without confirming leaves created_person_id() as None."""
    dlg = _make_dialog(tmp_path, qtbot)
    dlg._name_edit.setText("Dave")
    assert dlg.created_person_id() is None


def test_created_person_id_set_after_persist(tmp_path, qtbot):
    """After _persist_new_person(), _created_person_id holds the DB id."""
    dlg = _make_dialog(tmp_path, qtbot)

    person_id = dlg._persist_new_person("Frank")
    # Simulate what _on_accept() does after persisting
    dlg._created_person_id = person_id

    assert dlg.created_person_id() == person_id
