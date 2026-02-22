from unittest.mock import patch

import pytest

from photoaident.db.database import (
    Face,
    FaceState,
    Image,
    ImageMetadata,
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
