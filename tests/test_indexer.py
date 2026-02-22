import os
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image as PILImage

from photoaident.core.indexer import InventoryTask, IndexingTask
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
            "embedding": np.random.rand(512).astype(np.float32),
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
