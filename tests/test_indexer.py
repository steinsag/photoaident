import os
import pathlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image as PILImage
from sqlalchemy import select
from sqlalchemy.orm import Session

from photoaident.core.indexer import InventoryTask, IndexingTask
from photoaident.db.database import Image

_rng = np.random.default_rng(seed=42)


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


def test_indexing_task_success(
    tmp_path, session_factory, db_engine, vector_store, tmp_paths
):
    # Setup: Create an image that needs indexing
    img_file = tmp_path / "img1.jpg"
    PILImage.new("RGB", (100, 100), "red").save(img_file, "JPEG")

    with session_factory() as session:
        img = Image(file_path=str(img_file), file_size=len(b"dummy image data"))
        session.add(img)
        session.commit()
        img_id = img.id

    task = IndexingTask(session_factory, vector_store, tmp_paths, ctx_id=-1)

    # Mock FaceEmbedder to avoid running the real model
    mock_embedder = MagicMock()
    mock_embedder.process_image.return_value = [
        {
            "bbox": [10, 10, 60, 60],
            "embedding": _rng.random(512).astype(np.float32),
            "det_score": 0.95,
            "gender": 1,
            "age": 30,
        }
    ]
    mock_embedder.extract_face_crop.return_value = PILImage.new(
        "RGB", (224, 224), "blue"
    )

    task._embedder = mock_embedder

    task.run()

    # Verify DB
    from sqlalchemy.orm import Session

    with Session(db_engine) as session:
        img = session.get(Image, img_id)
        assert img is not None
        assert img.file_hash is not None
        assert len(img.faces) == 1
        face = img.faces[0]
        assert face.bbox_x == 10
        assert face.bbox_y == 10
        assert face.bbox_w == 50
        assert face.bbox_h == 50
        assert face.faiss_id == 0

    # Verify FAISS
    assert vector_store.index.ntotal == 1

    # Verify crop was saved
    crop_path = tmp_paths.face_crops_dir / f"{face.id}.jpg"
    assert crop_path.exists()

    # Verify full-photo thumbnail was saved
    thumb_path = tmp_paths.thumbs_dir / f"{img.file_hash}.jpg"
    assert thumb_path.exists()


# ---------------------------------------------------------------------------
# InventoryTask — additional branch coverage
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# IndexingTask — additional branch coverage
# ---------------------------------------------------------------------------


def test_indexing_task_cancel_method(session_factory, vector_store, tmp_paths):
    """cancel() sets the _is_cancelled flag."""
    task = IndexingTask(session_factory, vector_store, tmp_paths)
    assert not task._is_cancelled
    task.cancel()
    assert task._is_cancelled


def test_indexing_task_get_embedder_cached(session_factory, vector_store, tmp_paths):
    """_get_embedder() creates the embedder once and returns the same instance."""
    task = IndexingTask(session_factory, vector_store, tmp_paths, ctx_id=-1)

    with patch("photoaident.core.indexer.FaceEmbedder") as MockFE:
        mock_instance = MagicMock()
        MockFE.return_value = mock_instance

        e1 = task._get_embedder()
        e2 = task._get_embedder()

    assert e1 is mock_instance
    assert e2 is mock_instance
    assert MockFE.call_count == 1  # constructor called exactly once


def test_indexing_task_no_images_exits_early(session_factory, vector_store, tmp_paths):
    """IndexingTask finishes immediately when there are no un-indexed images."""
    # clean_db cleared the DB; no images → total_images == 0
    task = IndexingTask(session_factory, vector_store, tmp_paths, ctx_id=-1)
    task.run()  # must not raise


def test_indexing_task_cancel_during_inner_loop(
    tmp_path, session_factory, db_engine, vector_store, tmp_paths
):
    """Cancel triggered mid-batch stops processing after the current image."""
    img1 = tmp_path / "img1_inner.jpg"
    img2 = tmp_path / "img2_inner.jpg"
    PILImage.new("RGB", (10, 10), "red").save(img1)
    PILImage.new("RGB", (10, 10), "blue").save(img2)

    with session_factory() as session:
        session.add(Image(file_path=str(img1), file_size=img1.stat().st_size))
        session.add(Image(file_path=str(img2), file_size=img2.stat().st_size))
        session.commit()

    task = IndexingTask(session_factory, vector_store, tmp_paths, ctx_id=-1)

    mock_embedder = MagicMock()

    def process_and_cancel(_path):
        # Cancel after processing the first image so the second iteration
        # hits the `if self._is_cancelled: break` guard (line 182).
        task.cancel()
        return []

    mock_embedder.process_image.side_effect = process_and_cancel
    task._embedder = mock_embedder

    task.run()

    with Session(db_engine) as session:
        all_imgs = session.scalars(select(Image)).all()
        indexed = [i for i in all_imgs if i.file_hash not in (None, "MISSING", "ERROR")]
        not_indexed = [i for i in all_imgs if i.file_hash is None]

    assert len(indexed) == 1  # first image was fully processed
    assert len(not_indexed) == 1  # second image was skipped


def test_indexing_task_marks_missing_file(
    tmp_path, session_factory, db_engine, vector_store, tmp_paths
):
    """An image whose file no longer exists on disk is marked MISSING."""
    nonexistent = tmp_path / "gone.jpg"
    # Deliberately do NOT create the file.

    with session_factory() as session:
        img = Image(file_path=str(nonexistent), file_size=100)
        session.add(img)
        session.commit()
        img_id = img.id

    task = IndexingTask(session_factory, vector_store, tmp_paths, ctx_id=-1)
    task.run()

    with Session(db_engine) as session:
        img = session.get(Image, img_id)
    assert img is not None
    assert img.file_hash == "MISSING"


def test_indexing_task_marks_error_on_exception(
    tmp_path, session_factory, db_engine, vector_store, tmp_paths
):
    """An exception during indexing marks the image as ERROR."""
    img_file = tmp_path / "err.jpg"
    PILImage.new("RGB", (10, 10)).save(img_file)

    with session_factory() as session:
        img = Image(file_path=str(img_file), file_size=img_file.stat().st_size)
        session.add(img)
        session.commit()
        img_id = img.id

    task = IndexingTask(session_factory, vector_store, tmp_paths, ctx_id=-1)

    with patch.object(task, "_calculate_hash", side_effect=RuntimeError("boom")):
        task.run()

    with Session(db_engine) as session:
        img = session.get(Image, img_id)
    assert img is not None
    assert img.file_hash == "ERROR"
