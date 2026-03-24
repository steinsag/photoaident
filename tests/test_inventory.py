import os
import pathlib
from unittest.mock import patch

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from photoaident.core.inventory import InventoryTask
from photoaident.db.database import Image


@pytest.fixture
def require_symlinks(tmp_path: pathlib.Path) -> None:
    """Skip the test if the platform does not support directory symlinks."""
    probe_target = tmp_path / "_symlink_probe_target"
    probe_target.mkdir()
    probe_link = tmp_path / "_symlink_probe_link"
    try:
        probe_link.symlink_to(probe_target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks not supported on this platform: {exc}")


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


def test_inventory_task_skips_duplicate_image(tmp_path, session_factory, db_engine):
    """An image already in the DB is not inserted a second time."""
    img_file = tmp_path / "img.jpg"
    img_file.write_bytes(b"data")

    # Pre-populate the DB with the same path
    with session_factory() as session:
        session.add(Image(file_path=str(img_file), file_size=4))
        session.commit()

    task = InventoryTask(str(tmp_path), session_factory)
    task.run()

    with Session(db_engine) as session:
        images = session.scalars(select(Image)).all()
    assert len(images) == 1  # still only one record


def test_inventory_task_cancel_during_batch_loop(tmp_path, db_engine):
    """Cancel inside the DB batch loop triggers rollback before any row is written."""
    for i in range(3):
        (tmp_path / f"img{i}.jpg").write_bytes(b"x")

    task_box: list[InventoryTask] = []

    class _SessionThatCancels:
        def __init__(self) -> None:
            self._s = Session(bind=db_engine)

        def __enter__(self):
            # Cancel the owning task the moment the session opens so the
            # first iteration of the batch loop sees _is_cancelled=True.
            if task_box:
                task_box[0].cancel()
            return self._s.__enter__()

        def __exit__(self, *args):
            return self._s.__exit__(*args)

    def sf():
        return _SessionThatCancels()

    task = InventoryTask(str(tmp_path), sf)  # type: ignore[arg-type]
    task_box.append(task)
    task.run()

    with Session(db_engine) as session:
        images = session.scalars(select(Image)).all()
    assert len(images) == 0


def test_inventory_follows_symlinked_directory(
    tmp_path, session_factory, db_engine, require_symlinks
):
    """Images inside a symlinked subdirectory are discovered."""
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "img.jpg").write_bytes(b"x")

    collection = tmp_path / "collection"
    collection.mkdir()
    (collection / "link").symlink_to(real_dir, target_is_directory=True)

    task = InventoryTask(str(collection), session_factory)
    task.run()

    with Session(db_engine) as session:
        images = session.scalars(select(Image)).all()
    assert len(images) == 1
    assert images[0].file_path.endswith("img.jpg")


def test_inventory_handles_symlink_cycle(
    tmp_path, session_factory, db_engine, require_symlinks
):
    """A symlink that points back to an ancestor does not cause infinite recursion."""
    collection = tmp_path / "collection"
    collection.mkdir()
    (collection / "img.jpg").write_bytes(b"x")

    # Create a cycle: collection/loop -> collection
    (collection / "loop").symlink_to(collection, target_is_directory=True)

    task = InventoryTask(str(collection), session_factory)
    task.run()  # must terminate

    with Session(db_engine) as session:
        images = session.scalars(select(Image)).all()
    assert len(images) == 1
    assert images[0].file_path.endswith("img.jpg")


def test_inventory_skips_files_via_duplicate_symlink(
    tmp_path, session_factory, db_engine, require_symlinks
):
    """Two symlinks pointing to the same real directory don't double-count images."""
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "img.jpg").write_bytes(b"x")

    collection = tmp_path / "collection"
    collection.mkdir()
    (collection / "link_a").symlink_to(real_dir, target_is_directory=True)
    (collection / "link_b").symlink_to(real_dir, target_is_directory=True)

    task = InventoryTask(str(collection), session_factory)
    task.run()

    with Session(db_engine) as session:
        images = session.scalars(select(Image)).all()
    assert len(images) == 1


def test_inventory_task_skips_file_with_stat_error(
    tmp_path, session_factory, db_engine
):
    """Files whose stat() raises are silently skipped."""
    good = tmp_path / "good.jpg"
    bad = tmp_path / "bad.jpg"
    good.write_bytes(b"x")
    bad.write_bytes(b"x")

    original_stat = pathlib.Path.stat

    def selective_stat(self, *, follow_symlinks: bool = True):
        if self.name == "bad.jpg":
            raise OSError("permission denied")
        return original_stat(self, follow_symlinks=follow_symlinks)

    with patch.object(pathlib.Path, "stat", selective_stat):
        task = InventoryTask(str(tmp_path), session_factory)
        task.run()

    with Session(db_engine) as session:
        images = session.scalars(select(Image)).all()
    assert len(images) == 1
    assert images[0].file_path.endswith("good.jpg")
