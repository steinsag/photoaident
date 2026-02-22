import os

import pytest

from photoaident.core.indexer import InventoryTask
from photoaident.db.database import Image


@pytest.fixture
def session_factory(db_engine):
    """Factory for transactional sessions."""

    def factory():
        # InventoryTask uses this to create its own sessions
        from sqlalchemy.orm import Session

        return Session(bind=db_engine)

    return factory


@pytest.fixture(autouse=True)
def clean_db(db_engine):
    """Ensure database is empty before each test."""
    from sqlalchemy import delete
    from sqlalchemy.orm import Session
    from photoaident.db.database import Image

    with Session(db_engine) as session:
        session.execute(delete(Image))
        session.commit()


def test_inventory_task_success(tmp_path, session_factory, db_engine):
    # Setup: Create some dummy image files
    (tmp_path / "img1.jpg").write_bytes(b"data1")
    (tmp_path / "img2.JPEG").write_bytes(b"data2")
    (tmp_path / "not_image.txt").write_text("text")

    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "img3.jpg").write_bytes(b"data3")

    task = InventoryTask(str(tmp_path), session_factory)

    # Use a QSignalSpy or similar to verify signals if needed,
    # but here we can just run it synchronously for testing logic
    task.run()

    # Verify database
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    with Session(db_engine) as session:
        images = session.scalars(select(Image)).all()
        assert len(images) == 3
        paths = {os.path.basename(img.file_path) for img in images}
        assert paths == {"img1.jpg", "img2.JPEG", "img3.jpg"}
        for img in images:
            assert img.file_size > 0
            assert img.file_hash is None


def test_inventory_task_empty(tmp_path, session_factory, db_engine):
    task = InventoryTask(str(tmp_path), session_factory)
    task.run()

    from sqlalchemy import select
    from sqlalchemy.orm import Session

    with Session(db_engine) as session:
        images = session.scalars(select(Image)).all()
        assert len(images) == 0


def test_inventory_task_invalid_dir(session_factory, db_engine):
    task = InventoryTask("/non/existent/path", session_factory)
    task.run()

    from sqlalchemy import select
    from sqlalchemy.orm import Session

    with Session(db_engine) as session:
        images = session.scalars(select(Image)).all()
        assert len(images) == 0


def test_inventory_task_cancel(tmp_path, session_factory, db_engine):
    # Create many images to ensure we can cancel during scan or batch
    for i in range(200):
        (tmp_path / f"img{i}.jpg").write_bytes(b"data")

    task = InventoryTask(str(tmp_path), session_factory)

    # We want to cancel it. Since it runs synchronously in this test,
    # we'd need to mock something or run in thread.
    # Let's mock the session.add to cancel when called.

    # Start and immediately cancel
    # This is tricky because run() is a block.
    # Let's just test that the cancel flag is respected.
    task.cancel()
    task.run()

    from sqlalchemy import select
    from sqlalchemy.orm import Session

    with Session(db_engine) as session:
        images = session.scalars(select(Image)).all()
        # Should be 0 because we cancelled before it started doing anything useful
        assert len(images) == 0
