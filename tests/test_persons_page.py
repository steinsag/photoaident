"""Tests for PersonsPage and ReferenceFaceWidget."""

from PySide6 import QtWidgets

from photoaident.app import MainWindow
from photoaident.db.database import (
    AGE_CLUSTERS,
    EmbeddingCluster,
    Face,
    FaceState,
    Image,
    Person,
)
from photoaident.db.migrate import apply_migrations
from photoaident.paths import AppPaths
from photoaident.ui.pages.persons import PersonsPage, ReferenceFaceWidget, _PendingKind

# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_window(tmp_path, qtbot, collection_path: str = "") -> MainWindow:
    paths = AppPaths(
        base_data=tmp_path / "data",
        base_cache=tmp_path / "cache",
        base_config=tmp_path / "config",
    )
    paths.ensure_dirs()
    apply_migrations(f"sqlite:///{paths.db_path}")
    window = MainWindow(paths, check_gpu=False, enable_onboarding=False)
    window.settings.collection_path = collection_path
    qtbot.addWidget(window)
    return window


def _make_page(tmp_path, qtbot) -> PersonsPage:
    paths = AppPaths(
        base_data=tmp_path / "data",
        base_cache=tmp_path / "cache",
        base_config=tmp_path / "config",
    )
    paths.ensure_dirs()
    apply_migrations(f"sqlite:///{paths.db_path}")
    from photoaident.db.database import get_engine, get_session_factory

    engine = get_engine(str(paths.db_path))
    session_factory = get_session_factory(engine)
    page = PersonsPage(session_factory, paths)
    qtbot.addWidget(page)
    return page


def _add_person_with_clusters(session_factory, name: str) -> int:
    """Create a person with all 5 age clusters. Returns person_id."""
    with session_factory() as session:
        person = Person(name=name)
        session.add(person)
        session.flush()
        person_id = person.id
        for age_key in AGE_CLUSTERS:
            session.add(
                EmbeddingCluster(person_id=person_id, label=age_key, age_group=age_key)
            )
        session.commit()
    return person_id


def _get_cluster_id(session_factory, person_id: int, age_key: str) -> int:
    from sqlalchemy import select

    with session_factory() as session:
        cluster_id = session.execute(
            select(EmbeddingCluster.id).where(
                EmbeddingCluster.person_id == person_id,
                EmbeddingCluster.age_group == age_key,
            )
        ).scalar_one()
    return cluster_id


def _add_image(session_factory, path: str = "/fake/photo.jpg") -> int:
    with session_factory() as session:
        img = Image(file_path=path, file_size=1000)
        session.add(img)
        session.flush()
        image_id = img.id
        session.commit()
    return image_id


def _add_identified_face(
    session_factory,
    person_id: int,
    cluster_id: int,
    image_id: int,
    faiss_id: int = 0,
) -> int:
    with session_factory() as session:
        face = Face(
            image_id=image_id,
            faiss_id=faiss_id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=10,
            bbox_h=10,
            detection_confidence=0.99,
            model_version="v1",
            state=FaceState.IDENTIFIED,
            person_id=person_id,
            cluster_id=cluster_id,
        )
        session.add(face)
        session.flush()
        face_id = face.id
        session.commit()
    return face_id


# ─── Tests ──────────────────────────────────────────────────────────────────


def test_empty_db(tmp_path, qtbot):
    """No persons → list is empty, placeholder shown, buttons disabled."""
    page = _make_page(tmp_path, qtbot)
    page.refresh()

    assert page._person_list.count() == 0
    assert not page._placeholder_label.isHidden()
    assert not page._confirm_btn.isEnabled()
    assert not page._cancel_btn.isEnabled()


def test_persons_listed_sorted(tmp_path, qtbot):
    """'Zebra' and 'Alice' → Alice appears first in the list."""
    page = _make_page(tmp_path, qtbot)
    _add_person_with_clusters(page.session_factory, "Zebra")
    _add_person_with_clusters(page.session_factory, "Alice")
    page.refresh()

    assert page._person_list.count() == 2
    assert page._person_list.item(0).text() == "Alice"
    assert page._person_list.item(1).text() == "Zebra"


def test_person_selection_shows_five_clusters(tmp_path, qtbot):
    """Selecting a person shows 5 QGroupBoxes in AGE_CLUSTERS order."""
    page = _make_page(tmp_path, qtbot)
    _add_person_with_clusters(page.session_factory, "Alice")
    page.refresh()

    page._person_list.setCurrentRow(0)

    # Count QGroupBox children of the clusters widget
    group_boxes = page._clusters_widget.findChildren(QtWidgets.QGroupBox)
    assert len(group_boxes) == 5


def test_reference_faces_shown(tmp_path, qtbot):
    """2 IDENTIFIED faces in adult cluster → 2 ReferenceFaceWidgets."""
    page = _make_page(tmp_path, qtbot)
    person_id = _add_person_with_clusters(page.session_factory, "Bob")
    cluster_id = _get_cluster_id(page.session_factory, person_id, "adult")
    img_id = _add_image(page.session_factory)
    _add_identified_face(
        page.session_factory, person_id, cluster_id, img_id, faiss_id=0
    )
    _add_identified_face(
        page.session_factory, person_id, cluster_id, img_id, faiss_id=1
    )
    page.refresh()

    page._person_list.setCurrentRow(0)

    widgets = page._clusters_widget.findChildren(ReferenceFaceWidget)
    assert len(widgets) == 2


def test_empty_cluster_rendered(tmp_path, qtbot):
    """A cluster with 0 faces shows the '(No faces)' placeholder label."""
    page = _make_page(tmp_path, qtbot)
    _add_person_with_clusters(page.session_factory, "Carol")
    page.refresh()

    page._person_list.setCurrentRow(0)

    # Each of the 5 group boxes should contain the placeholder text
    labels = [
        lbl
        for lbl in page._clusters_widget.findChildren(QtWidgets.QLabel)
        if lbl.text() == "(No faces)"
    ]
    assert len(labels) == 5


def test_remove_marks_pending(tmp_path, qtbot):
    """_on_remove_requested adds a REMOVE entry; Confirm becomes enabled."""
    page = _make_page(tmp_path, qtbot)
    person_id = _add_person_with_clusters(page.session_factory, "Dave")
    cluster_id = _get_cluster_id(page.session_factory, person_id, "adult")
    img_id = _add_image(page.session_factory)
    face_id = _add_identified_face(page.session_factory, person_id, cluster_id, img_id)
    page.refresh()
    page._person_list.setCurrentRow(0)

    page._on_remove_requested(face_id)

    assert face_id in page._pending
    assert page._pending[face_id].kind == _PendingKind.REMOVE
    assert page._confirm_btn.isEnabled()
    assert page._cancel_btn.isEnabled()

    # status label on the widget should not be hidden
    widget = page._face_widgets[face_id]
    assert not widget._status_label.isHidden()


def test_remove_twice_undoes_pending(tmp_path, qtbot):
    """Calling _on_remove_requested twice removes the entry (undo)."""
    page = _make_page(tmp_path, qtbot)
    person_id = _add_person_with_clusters(page.session_factory, "Eve")
    cluster_id = _get_cluster_id(page.session_factory, person_id, "adult")
    img_id = _add_image(page.session_factory)
    face_id = _add_identified_face(page.session_factory, person_id, cluster_id, img_id)
    page.refresh()
    page._person_list.setCurrentRow(0)

    page._on_remove_requested(face_id)
    page._on_remove_requested(face_id)  # undo

    assert face_id not in page._pending
    assert not page._confirm_btn.isEnabled()
    assert not page._cancel_btn.isEnabled()


def test_move_marks_pending(tmp_path, qtbot):
    """_on_move_requested creates a MOVE entry in _pending."""
    page = _make_page(tmp_path, qtbot)
    person_id = _add_person_with_clusters(page.session_factory, "Frank")
    adult_cluster_id = _get_cluster_id(page.session_factory, person_id, "adult")
    infant_cluster_id = _get_cluster_id(page.session_factory, person_id, "infant")
    img_id = _add_image(page.session_factory)
    face_id = _add_identified_face(
        page.session_factory, person_id, adult_cluster_id, img_id
    )
    page.refresh()
    page._person_list.setCurrentRow(0)

    page._on_move_requested(face_id, infant_cluster_id)

    assert face_id in page._pending
    change = page._pending[face_id]
    assert change.kind == _PendingKind.MOVE
    assert change.new_cluster_id == infant_cluster_id


def test_move_overwrites_remove(tmp_path, qtbot):
    """Stage REMOVE then MOVE for the same face → only MOVE remains."""
    page = _make_page(tmp_path, qtbot)
    person_id = _add_person_with_clusters(page.session_factory, "Grace")
    adult_cluster_id = _get_cluster_id(page.session_factory, person_id, "adult")
    infant_cluster_id = _get_cluster_id(page.session_factory, person_id, "infant")
    img_id = _add_image(page.session_factory)
    face_id = _add_identified_face(
        page.session_factory, person_id, adult_cluster_id, img_id
    )
    page.refresh()
    page._person_list.setCurrentRow(0)

    page._on_remove_requested(face_id)
    page._on_move_requested(face_id, infant_cluster_id)

    assert page._pending[face_id].kind == _PendingKind.MOVE
    assert len(page._pending) == 1


def test_confirm_applies_removal(tmp_path, qtbot):
    """After confirm, the face is UNIDENTIFIED with no person/cluster."""
    page = _make_page(tmp_path, qtbot)
    person_id = _add_person_with_clusters(page.session_factory, "Heidi")
    cluster_id = _get_cluster_id(page.session_factory, person_id, "adult")
    img_id = _add_image(page.session_factory)
    face_id = _add_identified_face(page.session_factory, person_id, cluster_id, img_id)
    page.refresh()
    page._person_list.setCurrentRow(0)

    page._on_remove_requested(face_id)
    page._confirm()

    with page.session_factory() as session:
        face = session.get(Face, face_id)
        assert face is not None
        assert face.state == FaceState.UNIDENTIFIED
        assert face.person_id is None
        assert face.cluster_id is None
        assert face.labelled_at is None


def test_confirm_applies_move(tmp_path, qtbot):
    """After confirm, the face has the new cluster_id and stays IDENTIFIED."""
    page = _make_page(tmp_path, qtbot)
    person_id = _add_person_with_clusters(page.session_factory, "Ivan")
    adult_cluster_id = _get_cluster_id(page.session_factory, person_id, "adult")
    infant_cluster_id = _get_cluster_id(page.session_factory, person_id, "infant")
    img_id = _add_image(page.session_factory)
    face_id = _add_identified_face(
        page.session_factory, person_id, adult_cluster_id, img_id
    )
    page.refresh()
    page._person_list.setCurrentRow(0)

    page._on_move_requested(face_id, infant_cluster_id)
    page._confirm()

    with page.session_factory() as session:
        face = session.get(Face, face_id)
        assert face is not None
        assert face.cluster_id == infant_cluster_id
        assert face.state == FaceState.IDENTIFIED


def test_confirm_clears_pending(tmp_path, qtbot):
    """_pending is empty after confirm; Confirm button is disabled."""
    page = _make_page(tmp_path, qtbot)
    person_id = _add_person_with_clusters(page.session_factory, "Julia")
    cluster_id = _get_cluster_id(page.session_factory, person_id, "adult")
    img_id = _add_image(page.session_factory)
    face_id = _add_identified_face(page.session_factory, person_id, cluster_id, img_id)
    page.refresh()
    page._person_list.setCurrentRow(0)

    page._on_remove_requested(face_id)
    page._confirm()

    assert len(page._pending) == 0
    assert not page._confirm_btn.isEnabled()


def test_cancel_clears_pending(tmp_path, qtbot):
    """After cancel, _pending is empty and widgets show no overlay."""
    page = _make_page(tmp_path, qtbot)
    person_id = _add_person_with_clusters(page.session_factory, "Karl")
    cluster_id = _get_cluster_id(page.session_factory, person_id, "adult")
    img_id = _add_image(page.session_factory)
    face_id = _add_identified_face(page.session_factory, person_id, cluster_id, img_id)
    page.refresh()
    page._person_list.setCurrentRow(0)

    page._on_remove_requested(face_id)
    assert page._confirm_btn.isEnabled()

    page._cancel()

    assert len(page._pending) == 0
    assert not page._confirm_btn.isEnabled()
    assert not page._cancel_btn.isEnabled()
    widget = page._face_widgets[face_id]
    assert widget._status_label.isHidden()


def test_cancel_no_db_change(tmp_path, qtbot):
    """After cancel, the DB face is unchanged."""
    page = _make_page(tmp_path, qtbot)
    person_id = _add_person_with_clusters(page.session_factory, "Laura")
    cluster_id = _get_cluster_id(page.session_factory, person_id, "adult")
    img_id = _add_image(page.session_factory)
    face_id = _add_identified_face(page.session_factory, person_id, cluster_id, img_id)
    page.refresh()
    page._person_list.setCurrentRow(0)

    page._on_remove_requested(face_id)
    page._cancel()

    with page.session_factory() as session:
        face = session.get(Face, face_id)
        assert face is not None
        assert face.state == FaceState.IDENTIFIED
        assert face.person_id == person_id
        assert face.cluster_id == cluster_id


def test_search_filter(tmp_path, qtbot):
    """Typing 'ali' hides 'Bob' and shows 'Alice'."""
    page = _make_page(tmp_path, qtbot)
    _add_person_with_clusters(page.session_factory, "Alice")
    _add_person_with_clusters(page.session_factory, "Bob")
    page.refresh()

    page._filter_edit.setText("ali")

    alice_hidden = False
    bob_hidden = False
    for i in range(page._person_list.count()):
        item = page._person_list.item(i)
        if item.text() == "Alice":
            alice_hidden = item.isHidden()
        elif item.text() == "Bob":
            bob_hidden = item.isHidden()

    assert not alice_hidden
    assert bob_hidden


def test_buttons_disabled_initially(tmp_path, qtbot):
    """Confirm and Cancel are disabled on construction."""
    page = _make_page(tmp_path, qtbot)
    assert not page._confirm_btn.isEnabled()
    assert not page._cancel_btn.isEnabled()


def test_refresh_reloads_list(tmp_path, qtbot):
    """Adding a person via session then calling refresh() shows it."""
    page = _make_page(tmp_path, qtbot)
    page.refresh()
    assert page._person_list.count() == 0

    _add_person_with_clusters(page.session_factory, "NewPerson")
    page.refresh()

    assert page._person_list.count() == 1
    assert page._person_list.item(0).text() == "NewPerson"


def test_mainwindow_has_persons_page(tmp_path, qtbot):
    """MainWindow.stacked has 3 pages (Library, Label, Persons)."""
    window = _make_window(tmp_path, qtbot)
    assert window.stacked.count() == 3


def test_switch_to_persons_page(tmp_path, qtbot):
    """_switch_page(2) makes persons_page the current widget."""
    window = _make_window(tmp_path, qtbot)
    window._switch_page(2)
    assert window.stacked.currentWidget() is window.persons_page
