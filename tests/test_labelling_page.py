from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from PySide6 import QtWidgets

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
from photoaident.ui.pages.labelling import LabellingPage


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
def session_factory(tmp_path):
    db_path = tmp_path / "test.db"
    engine = get_engine(str(db_path))
    apply_migrations(f"sqlite:///{db_path}")
    return get_session_factory(engine)


@pytest.fixture
def vector_store():
    return VectorStore()


def _insert_face(session_factory, file_path: str = "/path/to/img.jpg") -> int:
    """Insert an unidentified face into the DB and return its id."""
    with session_factory() as session:
        img = Image(file_path=file_path, file_size=1000, file_hash="abc123")
        session.add(img)
        session.flush()

        meta = ImageMetadata(
            image_id=img.id,
            taken_at_source=TakenAtSource.FILESYSTEM,
            width=800,
            height=600,
        )
        session.add(meta)
        session.flush()

        face = Face(
            image_id=img.id,
            faiss_id=1,
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


def test_labelling_page_no_faces(qtbot, session_factory, test_paths, vector_store):
    """Page shows done state when the DB has no unidentified faces."""
    page = LabellingPage(session_factory, test_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    assert not page.assign_btn.isEnabled()
    assert not page.anonymous_btn.isEnabled()
    assert not page.skip_btn.isEnabled()
    assert "done" in page.status_label.text().lower()


def test_labelling_page_shows_face(qtbot, session_factory, test_paths, vector_store):
    """Page loads and displays a face when one exists in the DB."""
    face_id = _insert_face(session_factory)

    page = LabellingPage(session_factory, test_paths, vector_store)
    qtbot.addWidget(page)

    with patch.object(page.face_crop, "load") as mock_load:
        page.refresh()
        assert mock_load.called

    assert page._current_face_id == face_id
    assert page.assign_btn.isEnabled()
    assert page.anonymous_btn.isEnabled()
    assert page.skip_btn.isEnabled()


def test_mark_anonymous(qtbot, session_factory, test_paths, vector_store):
    """Clicking Mark Anonymous sets face.state = ANONYMOUS and advances."""
    face_id = _insert_face(session_factory)

    page = LabellingPage(session_factory, test_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    assert page._current_face_id == face_id

    page._mark_anonymous()

    # DB should reflect the state change
    with session_factory() as session:
        face = session.get(Face, face_id)
        assert face is not None
        assert face.state == FaceState.ANONYMOUS
        assert face.labelled_at is not None

    # No more unidentified faces → done state
    assert not page.assign_btn.isEnabled()
    assert page._current_face_id is None


def test_skip_face(qtbot, session_factory, test_paths, vector_store):
    """Skip does not change face state and moves to next (or done when all skipped)."""
    face_id = _insert_face(session_factory)

    page = LabellingPage(session_factory, test_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    assert page._current_face_id == face_id

    page._skip_face()

    # Face state must not have changed
    with session_factory() as session:
        face = session.get(Face, face_id)
        assert face is not None
        assert face.state == FaceState.UNIDENTIFIED
        assert face.labelled_at is None

    # Face should be in _skipped
    assert face_id in page._skipped

    # All faces skipped → buttons disabled
    assert not page.assign_btn.isEnabled()


def test_face_with_taken_at_displays_date(
    qtbot, session_factory, test_paths, vector_store
):
    """When metadata has taken_at, it is formatted as YYYY-MM-DD."""
    with session_factory() as session:
        img = Image(file_path="/dated.jpg", file_size=1000, file_hash="datehash")
        session.add(img)
        session.flush()
        session.add(
            ImageMetadata(
                image_id=img.id,
                taken_at_source=TakenAtSource.EXIF,
                taken_at=datetime(2020, 6, 15, 10, 30),
                width=800,
                height=600,
            )
        )
        session.flush()
        session.add(
            Face(
                image_id=img.id,
                faiss_id=77,
                bbox_x=0,
                bbox_y=0,
                bbox_w=50,
                bbox_h=50,
                detection_confidence=0.9,
                state=FaceState.UNIDENTIFIED,
                model_version="test",
            )
        )
        session.commit()

    page = LabellingPage(session_factory, test_paths, vector_store)
    qtbot.addWidget(page)

    with patch.object(page.face_crop, "load") as mock_load:
        page.refresh()
        assert mock_load.called
        assert mock_load.call_args.kwargs["taken_at"] == "2020-06-15"


def test_assign_face_identifies_and_advances(
    qtbot, session_factory, test_paths, vector_store
):
    """_assign_face sets state=IDENTIFIED and advances to the next face."""
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

    page = LabellingPage(session_factory, test_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    assert page._current_face_id == face_id

    mock_person = MagicMock()
    mock_person.id = person_id
    mock_cluster = MagicMock()
    mock_cluster.id = cluster_id

    with patch("photoaident.ui.pages.labelling.AssignPersonDialog") as MockDlg:
        inst = MockDlg.return_value
        inst.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
        inst.result_person_cluster.return_value = (mock_person, mock_cluster)
        page._assign_face()

    with session_factory() as session:
        face = session.get(Face, face_id)
        assert face is not None
        assert face.state == FaceState.IDENTIFIED
        assert face.person_id == person_id
        assert face.cluster_id == cluster_id
        assert face.labelled_at is not None


def test_assign_face_cancelled_does_nothing(
    qtbot, session_factory, test_paths, vector_store
):
    """Cancelling AssignPersonDialog leaves face state unchanged."""
    face_id = _insert_face(session_factory, "/cancel.jpg")

    page = LabellingPage(session_factory, test_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    with patch("photoaident.ui.pages.labelling.AssignPersonDialog") as MockDlg:
        inst = MockDlg.return_value
        inst.exec.return_value = QtWidgets.QDialog.DialogCode.Rejected
        page._assign_face()

    with session_factory() as session:
        face = session.get(Face, face_id)
        assert face is not None
        assert face.state == FaceState.UNIDENTIFIED


def test_mark_anonymous_noop_when_no_current_face(
    qtbot, session_factory, test_paths, vector_store
):
    """_mark_anonymous does nothing when _current_face_id is None."""
    page = LabellingPage(session_factory, test_paths, vector_store)
    qtbot.addWidget(page)

    assert page._current_face_id is None
    page._mark_anonymous()  # should not raise
    assert page._current_face_id is None


def test_skip_face_noop_when_no_current_face(
    qtbot, session_factory, test_paths, vector_store
):
    """_skip_face does nothing when _current_face_id is None."""
    page = LabellingPage(session_factory, test_paths, vector_store)
    qtbot.addWidget(page)

    assert page._current_face_id is None
    page._skip_face()  # should not raise
    assert page._current_face_id is None


def test_assign_face_passes_vector_store_to_dialog(
    qtbot, session_factory, test_paths, vector_store
):
    """_assign_face passes vector_store to AssignPersonDialog."""
    _insert_face(session_factory)

    page = LabellingPage(session_factory, test_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    assert page._current_face_id is not None

    with patch("photoaident.ui.pages.labelling.AssignPersonDialog") as MockDlg:
        inst = MockDlg.return_value
        inst.exec.return_value = QtWidgets.QDialog.DialogCode.Rejected
        page._assign_face()

        # Verify vector_store was passed as keyword argument
        call_kwargs = MockDlg.call_args.kwargs
        assert call_kwargs.get("vector_store") is vector_store


def _insert_face_for_image(session_factory, image_id: int, faiss_id: int) -> int:
    """Insert an unidentified face for an existing image; return the face id."""
    with session_factory() as session:
        face = Face(
            image_id=image_id,
            faiss_id=faiss_id,
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


def _insert_image(session_factory, file_path: str, file_hash: str) -> int:
    """Insert a bare Image row and return its id."""
    with session_factory() as session:
        img = Image(file_path=file_path, file_size=1000, file_hash=file_hash)
        session.add(img)
        session.commit()
        return img.id


def test_priority_image_shows_image_faces_first(
    qtbot, session_factory, test_paths, vector_store
):
    """refresh(priority_image_id=X) causes the first face to come from image X."""
    img1_id = _insert_image(session_factory, "/img1.jpg", "hash1")
    img2_id = _insert_image(session_factory, "/img2.jpg", "hash2")
    _insert_face_for_image(session_factory, img2_id, faiss_id=10)
    face1_id = _insert_face_for_image(session_factory, img1_id, faiss_id=11)

    page = LabellingPage(session_factory, test_paths, vector_store)
    qtbot.addWidget(page)

    with patch.object(page.face_crop, "load"):
        page.refresh(priority_image_id=img1_id)

    assert page._current_face_id == face1_id
    assert page._priority_image_id == img1_id


def test_priority_clears_after_all_image_faces_done(
    qtbot, session_factory, test_paths, vector_store
):
    """After all priority-image faces are labelled, priority is cleared."""
    img1_id = _insert_image(session_factory, "/p1.jpg", "phash1")
    img2_id = _insert_image(session_factory, "/p2.jpg", "phash2")
    face1_id = _insert_face_for_image(session_factory, img1_id, faiss_id=20)
    _insert_face_for_image(session_factory, img2_id, faiss_id=21)

    page = LabellingPage(session_factory, test_paths, vector_store)
    qtbot.addWidget(page)

    with patch.object(page.face_crop, "load"):
        page.refresh(priority_image_id=img1_id)

    assert page._current_face_id == face1_id

    # Mark the priority face anonymous — priority should clear, other face appears
    page._mark_anonymous()

    assert page._priority_image_id is None
    assert page._current_face_id is not None
    assert page._current_face_id != face1_id


def test_priority_status_message_shows_image_count(
    qtbot, session_factory, test_paths, vector_store
):
    """Status label says 'remaining in this image' when priority is set."""
    img_id = _insert_image(session_factory, "/sm.jpg", "smhash")
    _insert_face_for_image(session_factory, img_id, faiss_id=30)

    page = LabellingPage(session_factory, test_paths, vector_store)
    qtbot.addWidget(page)

    with patch.object(page.face_crop, "load"):
        page.refresh(priority_image_id=img_id)

    assert "remaining in this image" in page.status_label.text()


def test_refresh_without_priority_clears_priority(
    qtbot, session_factory, test_paths, vector_store
):
    """Calling refresh() without arguments clears any previously set priority."""
    img_id = _insert_image(session_factory, "/clr.jpg", "clrhash")
    _insert_face_for_image(session_factory, img_id, faiss_id=40)

    page = LabellingPage(session_factory, test_paths, vector_store)
    qtbot.addWidget(page)

    with patch.object(page.face_crop, "load"):
        page.refresh(priority_image_id=img_id)
    assert page._priority_image_id == img_id

    with patch.object(page.face_crop, "load"):
        page.refresh()
    assert page._priority_image_id is None


def test_skip_image_skips_all_faces_of_current_image(
    qtbot, session_factory, test_paths, vector_store
):
    """_skip_image records the current image in _skipped_images so all its faces
    are excluded from subsequent face queries."""
    img1_id = _insert_image(session_factory, "/si1.jpg", "sihash1")
    img2_id = _insert_image(session_factory, "/si2.jpg", "sihash2")
    face1_id = _insert_face_for_image(session_factory, img1_id, faiss_id=50)
    face2_id = _insert_face_for_image(session_factory, img1_id, faiss_id=51)
    face3_id = _insert_face_for_image(session_factory, img2_id, faiss_id=52)

    page = LabellingPage(session_factory, test_paths, vector_store)
    qtbot.addWidget(page)

    with patch.object(page.face_crop, "load"):
        page.refresh()

    assert page._current_face_id in (face1_id, face2_id, face3_id)
    first_image_id = (
        img1_id if page._current_face_id in (face1_id, face2_id) else img2_id
    )
    other_image_id = img2_id if first_image_id == img1_id else img1_id

    with patch.object(page.face_crop, "load"):
        page._skip_image()

    # The skipped image ID must be recorded
    assert first_image_id in page._skipped_images
    assert other_image_id not in page._skipped_images

    # After one click the current face must be from the OTHER image
    assert page._current_face_id in (face1_id, face2_id, face3_id)
    assert page._current_face_id not in (
        [face1_id, face2_id] if first_image_id == img1_id else [face3_id]
    )


def test_skip_image_does_not_change_face_state(
    qtbot, session_factory, test_paths, vector_store
):
    """_skip_image must not alter face state in the DB."""
    img_id = _insert_image(session_factory, "/sino.jpg", "sinohash")
    face1_id = _insert_face_for_image(session_factory, img_id, faiss_id=60)
    face2_id = _insert_face_for_image(session_factory, img_id, faiss_id=61)

    page = LabellingPage(session_factory, test_paths, vector_store)
    qtbot.addWidget(page)

    with patch.object(page.face_crop, "load"):
        page.refresh()

    page._skip_image()

    with session_factory() as session:
        for fid in (face1_id, face2_id):
            face = session.get(Face, fid)
            assert face is not None
            assert face.state == FaceState.UNIDENTIFIED
            assert face.labelled_at is None


def test_skip_image_noop_when_no_current_face(
    qtbot, session_factory, test_paths, vector_store
):
    """_skip_image does nothing when _current_face_id is None."""
    page = LabellingPage(session_factory, test_paths, vector_store)
    qtbot.addWidget(page)

    assert page._current_face_id is None
    page._skip_image()  # should not raise
    assert page._current_face_id is None
    assert len(page._skipped_images) == 0


def test_skip_image_button_exists_and_enabled_with_face(
    qtbot, session_factory, test_paths, vector_store
):
    """The Skip Image button is present and enabled when a face is loaded."""
    _insert_face(session_factory)

    page = LabellingPage(session_factory, test_paths, vector_store)
    qtbot.addWidget(page)

    with patch.object(page.face_crop, "load"):
        page.refresh()

    assert hasattr(page, "skip_image_btn")
    assert page.skip_image_btn.isEnabled()


def test_skip_image_button_disabled_when_no_faces(
    qtbot, session_factory, test_paths, vector_store
):
    """The Skip Image button is disabled when there are no faces to label."""
    page = LabellingPage(session_factory, test_paths, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    assert hasattr(page, "skip_image_btn")
    assert not page.skip_image_btn.isEnabled()


def test_skip_image_clears_priority_when_all_image_faces_skipped(
    qtbot, session_factory, test_paths, vector_store
):
    """Skipping the priority image via _skip_image clears priority in one click."""
    img1_id = _insert_image(session_factory, "/pri1.jpg", "prihash1")
    img2_id = _insert_image(session_factory, "/pri2.jpg", "prihash2")
    _insert_face_for_image(session_factory, img1_id, faiss_id=70)
    _insert_face_for_image(session_factory, img1_id, faiss_id=72)
    _insert_face_for_image(session_factory, img2_id, faiss_id=71)

    page = LabellingPage(session_factory, test_paths, vector_store)
    qtbot.addWidget(page)

    with patch.object(page.face_crop, "load"):
        page.refresh(priority_image_id=img1_id)

    assert page._priority_image_id == img1_id

    with patch.object(page.face_crop, "load"):
        page._skip_image()

    # One click must skip the entire priority image and clear priority
    assert img1_id in page._skipped_images
    assert page._priority_image_id is None
    # The next face shown should be from img2
    assert page._current_face_id is not None
