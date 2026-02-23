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


def test_labelling_page_no_faces(qtbot, session_factory, test_paths):
    """Page shows done state when the DB has no unidentified faces."""
    page = LabellingPage(session_factory, test_paths)
    qtbot.addWidget(page)
    page.refresh()

    assert not page.assign_btn.isEnabled()
    assert not page.anonymous_btn.isEnabled()
    assert not page.skip_btn.isEnabled()
    assert "done" in page.status_label.text().lower()


def test_labelling_page_shows_face(qtbot, session_factory, test_paths):
    """Page loads and displays a face when one exists in the DB."""
    face_id = _insert_face(session_factory)

    page = LabellingPage(session_factory, test_paths)
    qtbot.addWidget(page)

    with patch.object(page.face_crop, "load") as mock_load:
        page.refresh()
        assert mock_load.called

    assert page._current_face_id == face_id
    assert page.assign_btn.isEnabled()
    assert page.anonymous_btn.isEnabled()
    assert page.skip_btn.isEnabled()


def test_mark_anonymous(qtbot, session_factory, test_paths):
    """Clicking Mark Anonymous sets face.state = ANONYMOUS and advances."""
    face_id = _insert_face(session_factory)

    page = LabellingPage(session_factory, test_paths)
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


def test_skip_face(qtbot, session_factory, test_paths):
    """Skip does not change face state and moves to next (or done when all skipped)."""
    face_id = _insert_face(session_factory)

    page = LabellingPage(session_factory, test_paths)
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


def test_face_with_taken_at_displays_date(qtbot, session_factory, test_paths):
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

    page = LabellingPage(session_factory, test_paths)
    qtbot.addWidget(page)

    with patch.object(page.face_crop, "load") as mock_load:
        page.refresh()
        assert mock_load.called
        assert mock_load.call_args.kwargs["taken_at"] == "2020-06-15"


def test_assign_face_identifies_and_advances(qtbot, session_factory, test_paths):
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

    page = LabellingPage(session_factory, test_paths)
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


def test_assign_face_cancelled_does_nothing(qtbot, session_factory, test_paths):
    """Cancelling AssignPersonDialog leaves face state unchanged."""
    face_id = _insert_face(session_factory, "/cancel.jpg")

    page = LabellingPage(session_factory, test_paths)
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


def test_mark_anonymous_noop_when_no_current_face(qtbot, session_factory, test_paths):
    """_mark_anonymous does nothing when _current_face_id is None."""
    page = LabellingPage(session_factory, test_paths)
    qtbot.addWidget(page)

    assert page._current_face_id is None
    page._mark_anonymous()  # should not raise
    assert page._current_face_id is None


def test_skip_face_noop_when_no_current_face(qtbot, session_factory, test_paths):
    """_skip_face does nothing when _current_face_id is None."""
    page = LabellingPage(session_factory, test_paths)
    qtbot.addWidget(page)

    assert page._current_face_id is None
    page._skip_face()  # should not raise
    assert page._current_face_id is None
