"""Tests for PersonsPage and ReferenceFaceWidget."""

from pathlib import Path
from unittest.mock import patch, MagicMock

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


def _make_window(
    tmp_app_paths: AppPaths, qtbot, collection_path: str = ""
) -> MainWindow:
    apply_migrations(f"sqlite:///{tmp_app_paths.db_path}")
    window = MainWindow(tmp_app_paths, check_gpu=False, enable_onboarding=False)
    window._settings.collection_path = collection_path
    qtbot.addWidget(window)
    return window


def _make_page(tmp_app_paths: AppPaths, qtbot) -> PersonsPage:
    apply_migrations(f"sqlite:///{tmp_app_paths.db_path}")
    from photoaident.db.database import get_engine, get_session_factory
    from photoaident.db.vector_store import VectorStore

    engine = get_engine(str(tmp_app_paths.db_path))
    session_factory = get_session_factory(engine)
    page = PersonsPage(session_factory, tmp_app_paths, VectorStore())
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
) -> int:
    with session_factory() as session:
        face = Face(
            image_id=image_id,
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


def test_empty_db(tmp_app_paths, qtbot):
    """No persons → list is empty, placeholder shown, buttons disabled."""
    page = _make_page(tmp_app_paths, qtbot)
    page.refresh()

    assert page._person_list.count() == 0
    assert not page._placeholder_label.isHidden()
    assert not page._confirm_btn.isEnabled()
    assert not page._cancel_btn.isEnabled()


def test_persons_listed_sorted(tmp_app_paths, qtbot):
    """'Zebra' and 'Alice' → Alice appears first in the list."""
    page = _make_page(tmp_app_paths, qtbot)
    _add_person_with_clusters(page.session_factory, "Zebra")
    _add_person_with_clusters(page.session_factory, "Alice")
    page.refresh()

    assert page._person_list.count() == 2
    assert page._person_list.item(0).text() == "Alice"
    assert page._person_list.item(1).text() == "Zebra"


def test_person_selection_shows_five_clusters(tmp_app_paths, qtbot):
    """Selecting a person shows 5 QGroupBoxes in AGE_CLUSTERS order."""
    page = _make_page(tmp_app_paths, qtbot)
    _add_person_with_clusters(page.session_factory, "Alice")
    page.refresh()

    page._person_list.setCurrentRow(0)

    # Count QGroupBox children of the clusters widget
    group_boxes = page._clusters_widget.findChildren(QtWidgets.QGroupBox)
    assert len(group_boxes) == 5


def test_reference_faces_shown(tmp_app_paths, qtbot):
    """2 IDENTIFIED faces in adult cluster → 2 ReferenceFaceWidgets."""
    page = _make_page(tmp_app_paths, qtbot)
    person_id = _add_person_with_clusters(page.session_factory, "Bob")
    cluster_id = _get_cluster_id(page.session_factory, person_id, "adult")
    img_id = _add_image(page.session_factory)
    _add_identified_face(page.session_factory, person_id, cluster_id, img_id)
    _add_identified_face(page.session_factory, person_id, cluster_id, img_id)
    page.refresh()

    page._person_list.setCurrentRow(0)

    widgets = page._clusters_widget.findChildren(ReferenceFaceWidget)
    assert len(widgets) == 2


def test_empty_cluster_rendered(tmp_app_paths, qtbot):
    """A cluster with 0 faces shows the '(No faces)' placeholder label."""
    page = _make_page(tmp_app_paths, qtbot)
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


def test_remove_marks_pending(tmp_app_paths, qtbot):
    """_on_remove_requested adds a REMOVE entry; Confirm becomes enabled."""
    page = _make_page(tmp_app_paths, qtbot)
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


def test_remove_twice_undoes_pending(tmp_app_paths, qtbot):
    """Calling _on_remove_requested twice removes the entry (undo)."""
    page = _make_page(tmp_app_paths, qtbot)
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


def test_move_marks_pending(tmp_app_paths, qtbot):
    """_on_move_requested creates a MOVE entry in _pending."""
    page = _make_page(tmp_app_paths, qtbot)
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


def test_move_overwrites_remove(tmp_app_paths, qtbot):
    """Stage REMOVE then MOVE for the same face → only MOVE remains."""
    page = _make_page(tmp_app_paths, qtbot)
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


def test_confirm_applies_removal(tmp_app_paths, qtbot):
    """After confirm, the face is UNIDENTIFIED with no person/cluster."""
    page = _make_page(tmp_app_paths, qtbot)
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


def test_confirm_applies_move(tmp_app_paths, qtbot):
    """After confirm, the face has the new cluster_id and stays IDENTIFIED."""
    page = _make_page(tmp_app_paths, qtbot)
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


def test_confirm_clears_pending(tmp_app_paths, qtbot):
    """_pending is empty after confirm; Confirm button is disabled."""
    page = _make_page(tmp_app_paths, qtbot)
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


def test_cancel_clears_pending(tmp_app_paths, qtbot):
    """After cancel, _pending is empty and widgets show no overlay."""
    page = _make_page(tmp_app_paths, qtbot)
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


def test_cancel_no_db_change(tmp_app_paths, qtbot):
    """After cancel, the DB face is unchanged."""
    page = _make_page(tmp_app_paths, qtbot)
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


def test_search_filter(tmp_app_paths, qtbot):
    """Typing 'ali' hides 'Bob' and shows 'Alice'."""
    page = _make_page(tmp_app_paths, qtbot)
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


def test_buttons_disabled_initially(tmp_app_paths, qtbot):
    """Confirm and Cancel are disabled on construction."""
    page = _make_page(tmp_app_paths, qtbot)
    assert not page._confirm_btn.isEnabled()
    assert not page._cancel_btn.isEnabled()


def test_refresh_reloads_list(tmp_app_paths, qtbot):
    """Adding a person via session then calling refresh() shows it."""
    page = _make_page(tmp_app_paths, qtbot)
    page.refresh()
    assert page._person_list.count() == 0

    _add_person_with_clusters(page.session_factory, "NewPerson")
    page.refresh()

    assert page._person_list.count() == 1
    assert page._person_list.item(0).text() == "NewPerson"


def test_mainwindow_has_persons_page(tmp_app_paths, qtbot):
    """MainWindow.stacked has 4 pages (Library, Label, Persons, Browse)."""
    window = _make_window(tmp_app_paths, qtbot)
    assert window._stacked_pages.count() == 4


def test_switch_to_persons_page(tmp_app_paths, qtbot):
    """_switch_page(2) makes persons_page the current widget."""
    window = _make_window(tmp_app_paths, qtbot)
    window._switch_page(2)
    assert window._stacked_pages.currentWidget() is window._persons_page


def test_new_person_button_exists(tmp_app_paths, qtbot):
    """PersonsPage has a 'New Person…' button in the left panel."""
    page = _make_page(tmp_app_paths, qtbot)
    assert hasattr(page, "_new_person_btn")
    assert "New Person" in page._new_person_btn.text()


def test_new_person_dialog_creates_person_and_selects_it(tmp_app_paths, qtbot):
    """Accepting NewPersonDialog refreshes the list and selects the new person."""
    page = _make_page(tmp_app_paths, qtbot)
    page.refresh()
    assert page._person_list.count() == 0

    new_person_id = _add_person_with_clusters(page.session_factory, "Zara")

    mock_dialog = MagicMock()
    mock_dialog.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
    mock_dialog.created_person_id.return_value = new_person_id

    with patch(
        "photoaident.ui.pages.persons.NewPersonDialog", return_value=mock_dialog
    ):
        page._on_new_person()

    assert page._person_list.count() == 1
    assert page._person_list.item(0).text() == "Zara"
    assert page._person_list.currentItem() is not None
    assert page._person_list.currentItem().text() == "Zara"


def test_new_person_clears_filter_so_person_is_visible(tmp_app_paths, qtbot):
    """Active search filter is cleared when a new person is created, so the
    newly created person is not immediately hidden by the filter."""
    page = _make_page(tmp_app_paths, qtbot)
    _add_person_with_clusters(page.session_factory, "Alice")
    page.refresh()

    # Activate a filter that would hide "Zara"
    page._filter_edit.setText("ali")
    assert page._person_list.count() == 1  # Alice visible

    new_person_id = _add_person_with_clusters(page.session_factory, "Zara")

    mock_dialog = MagicMock()
    mock_dialog.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
    mock_dialog.created_person_id.return_value = new_person_id

    with patch(
        "photoaident.ui.pages.persons.NewPersonDialog", return_value=mock_dialog
    ):
        page._on_new_person()

    # Filter must be cleared
    assert page._filter_edit.text() == ""
    # Both persons visible
    assert page._person_list.count() == 2
    # New person selected and not hidden
    current = page._person_list.currentItem()
    assert current is not None
    assert current.text() == "Zara"
    assert not current.isHidden()


def test_new_person_dialog_cancel_does_not_change_list(tmp_app_paths, qtbot):
    """Cancelling NewPersonDialog leaves the person list unchanged."""
    page = _make_page(tmp_app_paths, qtbot)
    _add_person_with_clusters(page.session_factory, "Alice")
    page.refresh()
    assert page._person_list.count() == 1

    mock_dialog = MagicMock()
    mock_dialog.exec.return_value = QtWidgets.QDialog.DialogCode.Rejected

    with patch(
        "photoaident.ui.pages.persons.NewPersonDialog", return_value=mock_dialog
    ):
        page._on_new_person()

    assert page._person_list.count() == 1


# ─── Coverage gap tests ──────────────────────────────────────────────────────


def test_reference_face_widget_missing_crop(tmp_path: Path, qtbot):
    """ReferenceFaceWidget with a non-existent crop path shows '?' label."""
    crop_path = tmp_path / "nonexistent.jpg"
    assert not crop_path.exists()

    widget = ReferenceFaceWidget(
        face_id=1, crop_path=crop_path, cluster_id=1, other_clusters=[]
    )
    qtbot.addWidget(widget)

    assert widget._image_label.text() == "?"


def test_reference_face_widget_valid_crop(tmp_path: Path, qtbot):
    """ReferenceFaceWidget with a valid image file displays the pixmap (not '?')."""
    from PIL import Image as PILImage

    crop_path = tmp_path / "face.jpg"
    PILImage.new("RGB", (120, 120), color=(128, 64, 32)).save(crop_path)

    widget = ReferenceFaceWidget(
        face_id=2, crop_path=crop_path, cluster_id=1, other_clusters=[]
    )
    qtbot.addWidget(widget)

    # A pixmap should be set — text() would be "" when a pixmap is shown
    assert widget._image_label.text() == ""
    assert not widget._image_label.pixmap().isNull()


def test_reference_face_remove_button_emits_signal(tmp_path: Path, qtbot):
    """Clicking _remove_btn emits remove_requested with the correct face_id."""
    crop_path = tmp_path / "missing.jpg"
    widget = ReferenceFaceWidget(
        face_id=42, crop_path=crop_path, cluster_id=1, other_clusters=[]
    )
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.remove_requested, timeout=1000) as blocker:
        widget._remove_btn.click()

    assert blocker.args == [42]


def test_reference_face_move_button_emits_signal(tmp_path: Path, qtbot):
    """_on_move_clicked emits move_requested when a menu action is chosen."""
    from PySide6 import QtWidgets as _QtWidgets

    crop_path = tmp_path / "missing.jpg"
    widget = ReferenceFaceWidget(
        face_id=42,
        crop_path=crop_path,
        cluster_id=1,
        other_clusters=[(2, "Infant"), (3, "Youngster")],
    )
    qtbot.addWidget(widget)

    received: list[tuple[int, int]] = []
    widget.move_requested.connect(lambda fid, cid: received.append((fid, cid)))

    mock_action = MagicMock()
    mock_action.data.return_value = 2  # target cluster_id

    with patch.object(_QtWidgets, "QMenu") as MockQMenu:
        mock_menu = MockQMenu.return_value
        mock_menu.addAction.return_value = MagicMock()
        mock_menu.exec.return_value = mock_action

        widget._on_move_clicked()

    assert received == [(42, 2)]


def test_refresh_reselects_current_person(tmp_app_paths, qtbot):
    """refresh() re-selects the person that was selected before the call."""
    page = _make_page(tmp_app_paths, qtbot)
    _add_person_with_clusters(page.session_factory, "Alice")
    _add_person_with_clusters(page.session_factory, "Bob")
    page.refresh()

    # Bob is second in sorted order
    page._person_list.setCurrentRow(1)
    bob_id = page._selected_person_id
    assert page._person_list.item(1).text() == "Bob"

    page.refresh()

    current = page._person_list.currentItem()
    assert current is not None
    assert current.text() == "Bob"
    assert page._selected_person_id == bob_id


def test_new_person_none_id_preserves_state(tmp_app_paths, qtbot):
    """Accepted dialog with created_person_id()=None leaves list unchanged."""
    page = _make_page(tmp_app_paths, qtbot)
    _add_person_with_clusters(page.session_factory, "Alice")
    page.refresh()
    assert page._person_list.count() == 1

    mock_dialog = MagicMock()
    mock_dialog.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
    mock_dialog.created_person_id.return_value = None

    with patch(
        "photoaident.ui.pages.persons.NewPersonDialog", return_value=mock_dialog
    ):
        page._on_new_person()

    assert page._person_list.count() == 1


def test_person_deselect_clears_panel(tmp_app_paths, qtbot):
    """Calling _on_person_selected(None, None) clears the right panel."""
    page = _make_page(tmp_app_paths, qtbot)
    _add_person_with_clusters(page.session_factory, "Alice")
    page.refresh()
    page._person_list.setCurrentRow(0)

    # Panel should be showing the person
    assert not page._person_name_edit.isHidden()
    assert not page._scroll.isHidden()

    page._on_person_selected(None, None)

    assert page._selected_person_id is None
    assert not page._placeholder_label.isHidden()
    assert page._scroll.isHidden()
    assert page._person_name_edit.isHidden()
    assert len(page._pending) == 0


def test_load_person_deleted_between_select_and_load(tmp_app_paths, qtbot):
    """_load_person with a non-existent person_id clears the panel without crashing."""
    page = _make_page(tmp_app_paths, qtbot)
    _add_person_with_clusters(page.session_factory, "Alice")
    page.refresh()
    page._person_list.setCurrentRow(0)

    # Panel shows the person
    assert not page._person_name_edit.isHidden()

    # Simulate person disappearing from DB by using a never-existing ID
    page._load_person(999999)

    assert not page._placeholder_label.isHidden()
    assert page._person_name_edit.isHidden()


def test_load_person_skips_missing_age_groups(tmp_app_paths, qtbot):
    """Person with clusters for only 2 of 5 age groups → only 2 QGroupBoxes rendered."""
    page = _make_page(tmp_app_paths, qtbot)

    with page.session_factory() as session:
        person = Person(name="Partial")
        session.add(person)
        session.flush()
        person_id = person.id
        session.add(
            EmbeddingCluster(person_id=person_id, label="adult", age_group="adult")
        )
        session.add(
            EmbeddingCluster(person_id=person_id, label="infant", age_group="infant")
        )
        session.commit()

    page.refresh()
    page._person_list.setCurrentRow(0)

    group_boxes = page._clusters_widget.findChildren(QtWidgets.QGroupBox)
    assert len(group_boxes) == 2


def test_confirm_with_no_pending_is_noop(tmp_app_paths, qtbot):
    """_confirm() with empty _pending is a no-op — no crash, no DB change."""
    page = _make_page(tmp_app_paths, qtbot)
    _add_person_with_clusters(page.session_factory, "Alice")
    page.refresh()
    page._person_list.setCurrentRow(0)

    assert len(page._pending) == 0
    page._confirm()  # must not crash
    assert len(page._pending) == 0
    assert not page._confirm_btn.isEnabled()


def test_confirm_skips_deleted_face(tmp_app_paths, qtbot):
    """If a face is deleted between staging and confirming, _confirm skips it."""
    page = _make_page(tmp_app_paths, qtbot)
    person_id = _add_person_with_clusters(page.session_factory, "Alice")
    cluster_id = _get_cluster_id(page.session_factory, person_id, "adult")
    img_id = _add_image(page.session_factory)
    face_id = _add_identified_face(page.session_factory, person_id, cluster_id, img_id)
    page.refresh()
    page._person_list.setCurrentRow(0)

    page._on_remove_requested(face_id)
    assert face_id in page._pending

    # Delete the face from DB before confirming
    with page.session_factory() as session:
        face = session.get(Face, face_id)
        assert face is not None
        session.delete(face)
        session.commit()

    # Confirm must not crash and should clear pending
    page._confirm()
    assert len(page._pending) == 0


def test_name_edit_marks_pending(tmp_app_paths, qtbot):
    """Editing the name field sets _pending_name and enables Confirm."""
    page = _make_page(tmp_app_paths, qtbot)
    _add_person_with_clusters(page.session_factory, "Alice")
    page.refresh()
    page._person_list.setCurrentRow(0)

    page._person_name_edit.setText("Alicia")
    page._on_name_edited("Alicia")

    assert page._pending_name == "Alicia"
    assert page._confirm_btn.isEnabled()
    assert page._cancel_btn.isEnabled()


def test_name_edit_same_value_clears_pending(tmp_app_paths, qtbot):
    """Reverting the name to its original value clears _pending_name."""
    page = _make_page(tmp_app_paths, qtbot)
    _add_person_with_clusters(page.session_factory, "Alice")
    page.refresh()
    page._person_list.setCurrentRow(0)

    page._on_name_edited("Alice Renamed")
    assert page._pending_name is not None

    page._on_name_edited("Alice")
    assert page._pending_name is None
    assert not page._confirm_btn.isEnabled()


def test_name_edit_empty_marks_dirty_disables_confirm_enables_cancel(
    tmp_app_paths, qtbot
):
    """Clearing the name marks dirty: Cancel enabled, Confirm disabled."""
    page = _make_page(tmp_app_paths, qtbot)
    _add_person_with_clusters(page.session_factory, "Alice")
    page.refresh()
    page._person_list.setCurrentRow(0)

    page._on_name_edited("")

    assert page._pending_name == ""
    assert not page._confirm_btn.isEnabled()
    assert page._cancel_btn.isEnabled()


def test_name_edit_whitespace_only_clears_pending_and_normalizes(tmp_app_paths, qtbot):
    """Whitespace-only edit: normalizes to empty, Cancel enabled, Confirm disabled."""
    page = _make_page(tmp_app_paths, qtbot)
    _add_person_with_clusters(page.session_factory, "Alice")
    page.refresh()
    page._person_list.setCurrentRow(0)

    page._person_name_edit.setText("   ")
    page._on_name_edited("   ")

    assert page._person_name_edit.text() == ""
    assert page._pending_name == ""
    assert not page._confirm_btn.isEnabled()
    assert page._cancel_btn.isEnabled()


def test_cancel_after_empty_name_edit_restores_original(tmp_app_paths, qtbot):
    """Cancel after clearing the name field restores the original name."""
    page = _make_page(tmp_app_paths, qtbot)
    _add_person_with_clusters(page.session_factory, "Alice")
    page.refresh()
    page._person_list.setCurrentRow(0)

    page._on_name_edited("")
    assert page._cancel_btn.isEnabled()

    page._cancel()

    assert page._person_name_edit.text() == "Alice"
    assert page._pending_name is None
    assert not page._cancel_btn.isEnabled()
    assert not page._confirm_btn.isEnabled()


def test_name_edit_whitespace_around_new_name_normalizes(tmp_app_paths, qtbot):
    """Whitespace around a new name is stripped; the change is still pending."""
    page = _make_page(tmp_app_paths, qtbot)
    _add_person_with_clusters(page.session_factory, "Alice")
    page.refresh()
    page._person_list.setCurrentRow(0)

    page._person_name_edit.setText("  Alicia  ")
    page._on_name_edited("  Alicia  ")

    assert page._person_name_edit.text() == "Alicia"
    assert page._pending_name == "Alicia"
    assert page._confirm_btn.isEnabled()


def test_confirm_saves_new_name(tmp_app_paths, qtbot):
    """Confirming a name change persists the new name in the DB."""
    page = _make_page(tmp_app_paths, qtbot)
    person_id = _add_person_with_clusters(page.session_factory, "Alice")
    page.refresh()
    page._person_list.setCurrentRow(0)

    page._on_name_edited("Alicia")
    page._confirm()

    with page.session_factory() as session:
        from photoaident.db.database import Person as PersonModel

        person = session.get(PersonModel, person_id)
        assert person is not None
        assert person.name == "Alicia"

    assert page._pending_name is None
    assert not page._confirm_btn.isEnabled()


def test_confirm_name_change_refreshes_list(tmp_app_paths, qtbot):
    """After confirming a name change, the person list shows the new name."""
    page = _make_page(tmp_app_paths, qtbot)
    _add_person_with_clusters(page.session_factory, "Alice")
    page.refresh()
    page._person_list.setCurrentRow(0)

    page._on_name_edited("Alicia")
    page._confirm()

    assert page._person_list.count() == 1
    assert page._person_list.item(0).text() == "Alicia"


def test_cancel_reverts_name_edit(tmp_app_paths, qtbot):
    """Cancelling a pending name change restores the original name in the edit."""
    page = _make_page(tmp_app_paths, qtbot)
    _add_person_with_clusters(page.session_factory, "Alice")
    page.refresh()
    page._person_list.setCurrentRow(0)

    page._person_name_edit.setText("Alicia")
    page._on_name_edited("Alicia")
    assert page._confirm_btn.isEnabled()

    page._cancel()

    assert page._person_name_edit.text() == "Alice"
    assert page._pending_name is None
    assert not page._confirm_btn.isEnabled()


def test_pending_count_includes_name_change(tmp_app_paths, qtbot):
    """Changes label counts the name change alongside face changes."""
    page = _make_page(tmp_app_paths, qtbot)
    person_id = _add_person_with_clusters(page.session_factory, "Alice")
    cluster_id = _get_cluster_id(page.session_factory, person_id, "adult")
    img_id = _add_image(page.session_factory)
    face_id = _add_identified_face(page.session_factory, person_id, cluster_id, img_id)
    page.refresh()
    page._person_list.setCurrentRow(0)

    page._on_name_edited("Alicia")
    page._on_remove_requested(face_id)

    assert "2" in page._changes_label.text()


def test_confirm_name_change_keeps_person_selected(tmp_app_paths, qtbot):
    """After confirming a name change, the same person remains selected in the list."""
    page = _make_page(tmp_app_paths, qtbot)
    person_id = _add_person_with_clusters(page.session_factory, "Alice")
    page.refresh()
    page._person_list.setCurrentRow(0)
    assert page._selected_person_id == person_id

    page._on_name_edited("Alicia")
    page._confirm()

    # The list item text must be updated …
    assert page._person_list.item(0).text() == "Alicia"
    # … and still selected
    current = page._person_list.currentItem()
    assert current is not None
    assert current.text() == "Alicia"
    assert page._selected_person_id == person_id


def test_confirm_when_selected_person_is_none(tmp_app_paths, qtbot):
    """_confirm() with _selected_person_id=None calls _update_action_buttons."""
    page = _make_page(tmp_app_paths, qtbot)
    person_id = _add_person_with_clusters(page.session_factory, "Alice")
    cluster_id = _get_cluster_id(page.session_factory, person_id, "adult")
    img_id = _add_image(page.session_factory)
    face_id = _add_identified_face(page.session_factory, person_id, cluster_id, img_id)
    page.refresh()
    page._person_list.setCurrentRow(0)

    page._on_remove_requested(face_id)

    # Simulate person deselection after staging
    page._selected_person_id = None

    page._confirm()

    assert len(page._pending) == 0
    assert not page._confirm_btn.isEnabled()

    with page.session_factory() as session:
        face = session.get(Face, face_id)
        assert face is not None
        assert face.state == FaceState.UNIDENTIFIED


def test_confirm_face_only_does_not_reload_persons_list(tmp_app_paths, qtbot):
    """Face-only confirm reloads the right panel but not the persons list."""
    page = _make_page(tmp_app_paths, qtbot)
    person_id = _add_person_with_clusters(page.session_factory, "Felix")
    cluster_id = _get_cluster_id(page.session_factory, person_id, "adult")
    img_id = _add_image(page.session_factory)
    face_id = _add_identified_face(page.session_factory, person_id, cluster_id, img_id)
    page.refresh()
    page._person_list.setCurrentRow(0)

    load_persons_calls: list[None] = []
    original_load = page._load_persons

    def _tracking_load() -> None:
        load_persons_calls.append(None)
        original_load()

    page._load_persons = _tracking_load  # type: ignore[method-assign]

    page._on_remove_requested(face_id)
    page._confirm()

    assert (
        load_persons_calls == []
    ), "_load_persons should not be called for face-only changes"
    assert not page._person_name_edit.isHidden()
    assert page._selected_person_id == person_id


# ===========================================================================
# _delete_person_btn / _on_delete_person / _delete_person
# ===========================================================================


def test_delete_person_btn_disabled_initially(tmp_app_paths, qtbot):
    """Delete Person button is disabled when no person is selected."""
    page = _make_page(tmp_app_paths, qtbot)
    assert not page._delete_person_btn.isEnabled()


def test_delete_person_btn_enabled_after_selection(tmp_app_paths, qtbot):
    """Delete Person button becomes enabled when a person is selected."""
    page = _make_page(tmp_app_paths, qtbot)
    _add_person_with_clusters(page.session_factory, "Alice")
    page.refresh()

    page._person_list.setCurrentRow(0)

    assert page._delete_person_btn.isEnabled()


def test_delete_person_btn_disabled_after_deselection(tmp_app_paths, qtbot):
    """Delete Person button becomes disabled again after the selection is cleared."""
    page = _make_page(tmp_app_paths, qtbot)
    _add_person_with_clusters(page.session_factory, "Alice")
    page.refresh()
    page._person_list.setCurrentRow(0)
    assert page._delete_person_btn.isEnabled()

    page._on_person_selected(None, None)

    assert not page._delete_person_btn.isEnabled()


def test_delete_person_removes_person_from_db(tmp_app_paths, qtbot):
    """_delete_person removes the Person row from the database."""
    page = _make_page(tmp_app_paths, qtbot)
    person_id = _add_person_with_clusters(page.session_factory, "Alice")
    page.refresh()
    page._person_list.setCurrentRow(0)

    page._delete_person(person_id)

    with page.session_factory() as session:
        person = session.get(Person, person_id)
        assert person is None


def test_delete_person_cascades_to_clusters(tmp_app_paths, qtbot):
    """_delete_person removes all EmbeddingCluster rows for the person."""
    from sqlalchemy import select as sa_select

    page = _make_page(tmp_app_paths, qtbot)
    person_id = _add_person_with_clusters(page.session_factory, "Bob")
    page.refresh()
    page._person_list.setCurrentRow(0)

    page._delete_person(person_id)

    with page.session_factory() as session:
        count = (
            session.execute(
                sa_select(EmbeddingCluster).where(
                    EmbeddingCluster.person_id == person_id
                )
            )
            .scalars()
            .all()
        )
        assert count == []


def test_delete_person_unlinks_identified_faces(tmp_app_paths, qtbot):
    """_delete_person sets face state to UNIDENTIFIED and clears person/cluster."""
    page = _make_page(tmp_app_paths, qtbot)
    person_id = _add_person_with_clusters(page.session_factory, "Carol")
    cluster_id = _get_cluster_id(page.session_factory, person_id, "adult")
    img_id = _add_image(page.session_factory)
    face_id = _add_identified_face(page.session_factory, person_id, cluster_id, img_id)
    page.refresh()
    page._person_list.setCurrentRow(0)

    page._delete_person(person_id)

    with page.session_factory() as session:
        face = session.get(Face, face_id)
        assert face is not None
        assert face.state == FaceState.UNIDENTIFIED
        assert face.person_id is None
        assert face.cluster_id is None
        assert face.labelled_at is None


def test_delete_person_refreshes_person_list(tmp_app_paths, qtbot):
    """After _delete_person, the deleted person no longer appears in the list."""
    page = _make_page(tmp_app_paths, qtbot)
    person_id = _add_person_with_clusters(page.session_factory, "Dave")
    _add_person_with_clusters(page.session_factory, "Eve")
    page.refresh()
    assert page._person_list.count() == 2

    page._person_list.setCurrentRow(0)

    page._delete_person(person_id)

    assert page._person_list.count() == 1
    assert page._person_list.item(0).text() == "Eve"


def test_on_delete_person_cancel_does_nothing(tmp_app_paths, qtbot):
    """_on_delete_person does not delete when the user cancels the dialog."""
    page = _make_page(tmp_app_paths, qtbot)
    person_id = _add_person_with_clusters(page.session_factory, "Frank")
    page.refresh()
    page._person_list.setCurrentRow(0)

    mock_msg = MagicMock()
    delete_btn = MagicMock()
    mock_msg.addButton.return_value = delete_btn
    # clickedButton returns a different object → user did not click Delete
    mock_msg.clickedButton.return_value = MagicMock()

    with patch(
        "photoaident.ui.pages.persons.QtWidgets.QMessageBox", return_value=mock_msg
    ):
        page._on_delete_person()

    with page.session_factory() as session:
        person = session.get(Person, person_id)
        assert person is not None


def test_on_delete_person_confirm_deletes_person(tmp_app_paths, qtbot):
    """_on_delete_person deletes the person when the user confirms."""
    page = _make_page(tmp_app_paths, qtbot)
    person_id = _add_person_with_clusters(page.session_factory, "Grace")
    page.refresh()
    page._person_list.setCurrentRow(0)

    mock_msg = MagicMock()
    delete_btn = MagicMock()
    mock_msg.addButton.return_value = delete_btn
    # clickedButton returns the same object as addButton → user clicked Delete
    mock_msg.clickedButton.return_value = delete_btn

    with patch(
        "photoaident.ui.pages.persons.QtWidgets.QMessageBox", return_value=mock_msg
    ):
        page._on_delete_person()

    with page.session_factory() as session:
        person = session.get(Person, person_id)
        assert person is None


def test_on_delete_person_uses_pending_name_in_dialog(tmp_app_paths, qtbot):
    """_on_delete_person shows the pending (edited) name in the confirmation dialog."""
    page = _make_page(tmp_app_paths, qtbot)
    _add_person_with_clusters(page.session_factory, "Harold")
    page.refresh()
    page._person_list.setCurrentRow(0)

    # Simulate the user editing the name without confirming (textEdited only fires on
    # user input, not programmatic setText, so drive the handler directly)
    page._on_name_edited("Harold Renamed")

    mock_msg = MagicMock()
    delete_btn = MagicMock()
    mock_msg.addButton.return_value = delete_btn
    mock_msg.clickedButton.return_value = MagicMock()  # cancel

    setText_calls: list[str] = []
    mock_msg.setText.side_effect = lambda t: setText_calls.append(t)

    with patch(
        "photoaident.ui.pages.persons.QtWidgets.QMessageBox", return_value=mock_msg
    ):
        page._on_delete_person()

    assert setText_calls, "setText was never called on the message box"
    assert "Harold Renamed" in setText_calls[0]
    assert "Harold" not in setText_calls[0].replace("Harold Renamed", "")
