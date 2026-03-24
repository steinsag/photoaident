from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image as PILImage
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from photoaident.core.indexing import IndexingTask, _dms_to_decimal
from photoaident.db.database import (
    Face as FaceModel,
    FaceState,
    Image,
    ImageMetadata,
    TakenAtSource,
)
from photoaident.db.vector_store import VectorStore


class _Ratio:
    """Minimal Ratio stub matching exifread's Ratio type."""

    def __init__(self, num: int, den: int) -> None:
        self.num = num
        self.den = den


class _MockTag:
    """Minimal exifread IfdTag stub."""

    def __init__(self, values) -> None:
        self.values = values

    def __str__(self) -> str:
        if isinstance(self.values, str):
            return self.values
        return str(self.values)


_rng = np.random.default_rng(seed=42)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_image_file(tmp_path, filename="img.jpg", size=(100, 100), color="red"):
    """Create a JPEG on disk and return its path."""
    img_file = tmp_path / filename
    PILImage.new("RGB", size, color).save(img_file, "JPEG")
    return img_file


def _insert_image(session_factory, img_file, *, file_size=None):
    """Insert an Image row and return its id."""
    with session_factory() as session:
        img = Image(
            file_path=str(img_file),
            file_size=file_size if file_size is not None else img_file.stat().st_size,
        )
        session.add(img)
        session.commit()
        return img.id


def _make_indexed_image(tmp_path, session_factory, filename="img.jpg"):
    """Create a real JPEG on disk and a matching unindexed Image row."""
    img_file = _make_image_file(tmp_path, filename)
    return img_file, _insert_image(session_factory, img_file)


def _make_task(session_factory, vector_store, tmp_app_paths, **kwargs):
    """Create an IndexingTask with a no-op mock embedder (no detections)."""
    task = IndexingTask(
        session_factory, vector_store, tmp_app_paths, ctx_id=-1, **kwargs
    )
    mock_embedder = MagicMock()
    mock_embedder.process_image.return_value = []
    task._embedder = mock_embedder
    return task


def _fetch_metadata(db_engine, img_id):
    """Return the ImageMetadata row for img_id, or None."""
    with Session(db_engine) as session:
        return session.execute(
            select(ImageMetadata).where(ImageMetadata.image_id == img_id)
        ).scalar_one_or_none()


def _run_task_with_tags(session_factory, vector_store, tmp_paths, fake_tags):
    """Run IndexingTask with mocked exifread returning fake_tags."""
    task = _make_task(session_factory, vector_store, tmp_paths)
    with patch(
        "photoaident.core.indexing.exifread.process_file", return_value=fake_tags
    ):
        task.run()


def _setup_image_with_stale_face(tmp_path, session_factory, vector_store, filename):
    """Create an image with a pre-existing face in DB and FAISS, then reset its hash
    so it will be picked up for re-indexing.

    Returns (img_id, old_face_id, old_embedding).
    """
    img_file = _make_image_file(tmp_path, filename)
    old_embedding = _rng.random(VectorStore.DEFAULT_DIMENSION).astype(
        VectorStore.EMBEDDING_DTYPE
    )
    with session_factory() as session:
        img = Image(file_path=str(img_file), file_size=img_file.stat().st_size)
        session.add(img)
        session.commit()
        old_face = FaceModel(
            image_id=img.id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=10,
            bbox_h=10,
            detection_confidence=0.9,
            state=FaceState.UNIDENTIFIED,
            model_version="buffalo_l",
        )
        session.add(old_face)
        session.flush()
        old_face_id = old_face.id
        vector_store.add(old_face_id, old_embedding)
        session.commit()
        img_id = img.id

    with session_factory() as session:
        img = session.get(Image, img_id)
        img.file_hash = None
        session.commit()

    return img_id, old_face_id, old_embedding


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session_factory(db_engine):
    """Factory for transactional sessions."""

    def factory():
        from sqlalchemy.orm import Session

        return Session(bind=db_engine)

    return factory


@pytest.fixture(autouse=True)
def clean_db(db_engine):
    """Ensure database is empty before each test."""
    from sqlalchemy import delete
    from sqlalchemy.orm import Session
    from photoaident.db.database import Image, ImageMetadata

    with Session(db_engine) as session:
        # Delete child tables first (FK constraints, even without enforcement)
        session.execute(delete(ImageMetadata))
        session.execute(delete(Image))
        session.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_indexing_task_success(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    img_file = _make_image_file(tmp_path, "img1.jpg")
    img_id = _insert_image(session_factory, img_file)

    task = IndexingTask(session_factory, vector_store, tmp_app_paths, ctx_id=-1)
    mock_embedder = MagicMock()
    mock_embedder.process_image.return_value = [
        {
            "bbox": [10, 10, 60, 60],
            "embedding": _rng.random(VectorStore.DEFAULT_DIMENSION).astype(
                VectorStore.EMBEDDING_DTYPE
            ),
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

    # Verify FAISS
    assert vector_store.index.ntotal == 1

    # Verify crop was saved
    crop_path = tmp_app_paths.face_crops_dir / f"{face.id}.jpg"
    assert crop_path.exists()

    # Verify no full-photo thumbnail was created during indexing (lazy generation only)
    assert not any(tmp_app_paths.thumbs_dir.iterdir())

    # Verify image_metadata was populated
    meta = _fetch_metadata(db_engine, img_id)
    assert meta is not None
    assert meta.width == 100
    assert meta.height == 100
    # PIL-created fixture images have no EXIF and no filepath pattern → no date
    assert meta.taken_at is None
    assert meta.taken_at_source is None


def test_indexing_task_cancel_method(session_factory, vector_store, tmp_app_paths):
    """cancel() sets the _is_cancelled flag."""
    task = IndexingTask(session_factory, vector_store, tmp_app_paths)
    assert not task._is_cancelled
    task.cancel()
    assert task._is_cancelled


def test_indexing_task_get_embedder_cached(
    session_factory, vector_store, tmp_app_paths
):
    """_get_embedder() creates the embedder once and returns the same instance."""
    task = IndexingTask(session_factory, vector_store, tmp_app_paths, ctx_id=-1)

    with patch("photoaident.core.indexing.FaceEmbedder") as MockFE:
        mock_instance = MagicMock()
        MockFE.return_value = mock_instance

        e1 = task._get_embedder()
        e2 = task._get_embedder()

    assert e1 is mock_instance
    assert e2 is mock_instance
    assert MockFE.call_count == 1  # constructor called exactly once


def test_indexing_task_no_images_exits_early(
    session_factory, vector_store, tmp_app_paths
):
    """IndexingTask finishes immediately when there are no un-indexed images."""
    # clean_db cleared the DB; no images → total_images == 0
    task = IndexingTask(session_factory, vector_store, tmp_app_paths, ctx_id=-1)
    task.run()  # must not raise


def test_indexing_task_cancel_during_inner_loop(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """Cancel triggered mid-batch stops processing after the current image."""
    img1 = _make_image_file(tmp_path, "img1_inner.jpg", size=(10, 10), color="red")
    img2 = _make_image_file(tmp_path, "img2_inner.jpg", size=(10, 10), color="blue")
    _insert_image(session_factory, img1)
    _insert_image(session_factory, img2)

    task = IndexingTask(session_factory, vector_store, tmp_app_paths, ctx_id=-1)
    mock_embedder = MagicMock()

    def process_and_cancel(_):
        # Cancel after processing the first image so the second iteration
        # hits the `if self._is_cancelled: break` guard.
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
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """An image whose file no longer exists on disk is marked MISSING."""
    nonexistent = tmp_path / "gone.jpg"
    # Deliberately do NOT create the file.
    img_id = _insert_image(session_factory, nonexistent, file_size=100)

    task = IndexingTask(session_factory, vector_store, tmp_app_paths, ctx_id=-1)
    task._embedder = MagicMock()  # file is missing; don't load the real model
    task.run()

    with Session(db_engine) as session:
        img = session.get(Image, img_id)
    assert img is not None
    assert img.file_hash == "MISSING"


def test_indexing_task_marks_error_on_exception(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """An exception during indexing marks the image as ERROR."""
    img_file = _make_image_file(tmp_path, "err.jpg", size=(10, 10))
    img_id = _insert_image(session_factory, img_file)

    task = IndexingTask(session_factory, vector_store, tmp_app_paths, ctx_id=-1)
    task._embedder = MagicMock()  # error is in _calculate_hash; don't load real model

    with patch.object(task, "_calculate_hash", side_effect=RuntimeError("boom")):
        task.run()

    with Session(db_engine) as session:
        img = session.get(Image, img_id)
    assert img is not None
    assert img.file_hash == "ERROR"


def test_indexing_task_metadata_populated(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """ImageMetadata is created for each successfully indexed image."""
    img_file = _make_image_file(
        tmp_path, "meta_test.jpg", size=(320, 240), color="green"
    )
    img_id = _insert_image(session_factory, img_file)

    _make_task(session_factory, vector_store, tmp_app_paths).run()

    meta = _fetch_metadata(db_engine, img_id)
    assert meta is not None
    assert meta.width == 320
    assert meta.height == 240
    # No EXIF and no filepath pattern → no date
    assert meta.taken_at is None
    assert meta.taken_at_source is None


def test_indexing_task_no_thumbnail_during_indexing(
    tmp_path, session_factory, vector_store, tmp_app_paths
):
    """No thumbnail files are written to thumbs_dir during indexing."""
    img_file = _make_image_file(tmp_path, "no_thumb.jpg", size=(50, 50), color="blue")
    _insert_image(session_factory, img_file)

    _make_task(session_factory, vector_store, tmp_app_paths).run()

    assert not any(tmp_app_paths.thumbs_dir.iterdir())


def test_indexing_task_exif_failure_does_not_abort(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """EXIF extraction failure must not abort indexing — image is still marked done."""
    img_file = _make_image_file(tmp_path, "exif_fail.jpg", size=(10, 10))
    img_id = _insert_image(session_factory, img_file)

    task = _make_task(session_factory, vector_store, tmp_app_paths)
    with patch(
        "photoaident.core.indexing.exifread.process_file",
        side_effect=OSError("corrupt"),
    ):
        task.run()

    with Session(db_engine) as session:
        img = session.get(Image, img_id)
    assert img is not None
    # Image should be fully indexed (hash set, not ERROR) despite EXIF failure
    assert img.file_hash not in (None, "ERROR", "MISSING")


def test_dms_to_decimal_north():
    """N reference → positive decimal degrees."""
    values = [_Ratio(10, 1), _Ratio(30, 1), _Ratio(0, 1)]  # 10°30'0"
    assert _dms_to_decimal(values, "N") == pytest.approx(10.5)


def test_dms_to_decimal_east():
    """E reference → positive decimal degrees."""
    values = [_Ratio(2, 1), _Ratio(21, 1), _Ratio(0, 1)]  # 2°21'0"
    assert _dms_to_decimal(values, "E") == pytest.approx(2.35)


def test_dms_to_decimal_south_negated():
    """S reference → negated decimal degrees."""
    values = [_Ratio(10, 1), _Ratio(30, 1), _Ratio(0, 1)]
    assert _dms_to_decimal(values, "S") == pytest.approx(-10.5)


def test_dms_to_decimal_west_negated():
    """W reference → negated decimal degrees."""
    values = [_Ratio(10, 1), _Ratio(30, 1), _Ratio(0, 1)]
    assert _dms_to_decimal(values, "W") == pytest.approx(-10.5)


def test_dms_to_decimal_returns_none_on_error():
    """Empty list causes IndexError → returns None."""
    assert _dms_to_decimal([], "N") is None


def test_exif_datetime_from_exif_tag(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """taken_at and taken_at_source == EXIF when DateTimeOriginal is present."""
    _, img_id = _make_indexed_image(tmp_path, session_factory, "dt_exif.jpg")
    fake_tags = {"EXIF DateTimeOriginal": _MockTag("2022:08:20 10:15:30")}
    _run_task_with_tags(session_factory, vector_store, tmp_app_paths, fake_tags)

    meta = _fetch_metadata(db_engine, img_id)
    assert meta is not None
    assert meta.taken_at_source == TakenAtSource.EXIF
    assert meta.taken_at == datetime(2022, 8, 20, 10, 15, 30)


def test_exif_invalid_datetime_falls_through_to_next_tag(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """Invalid DateTimeOriginal (ValueError) → falls through to DateTimeDigitized."""
    _, img_id = _make_indexed_image(tmp_path, session_factory, "dt_fallback.jpg")
    fake_tags = {
        "EXIF DateTimeOriginal": _MockTag("not-a-date"),
        "EXIF DateTimeDigitized": _MockTag("2021:03:01 09:00:00"),
    }
    _run_task_with_tags(session_factory, vector_store, tmp_app_paths, fake_tags)

    meta = _fetch_metadata(db_engine, img_id)
    assert meta is not None
    assert meta.taken_at_source == TakenAtSource.EXIF
    assert meta.taken_at is not None
    assert meta.taken_at.year == 2021


def test_exif_gps_coordinates(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """GPS lat/lon are extracted and stored when GPS tags are present."""
    _, img_id = _make_indexed_image(tmp_path, session_factory, "gps.jpg")
    fake_tags = {
        "GPS GPSLatitude": _MockTag([_Ratio(48, 1), _Ratio(51, 1), _Ratio(0, 1)]),
        "GPS GPSLatitudeRef": _MockTag("N"),
        "GPS GPSLongitude": _MockTag([_Ratio(2, 1), _Ratio(21, 1), _Ratio(0, 1)]),
        "GPS GPSLongitudeRef": _MockTag("E"),
    }
    _run_task_with_tags(session_factory, vector_store, tmp_app_paths, fake_tags)

    meta = _fetch_metadata(db_engine, img_id)
    assert meta is not None
    assert meta.gps_lat is not None
    assert meta.gps_lon is not None
    assert float(meta.gps_lat) == pytest.approx(48.85)
    assert float(meta.gps_lon) == pytest.approx(2.35)


def test_exif_gps_altitude_above_sea_level(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """Altitude ref 0 (above sea level) → positive gps_altitude."""
    _, img_id = _make_indexed_image(tmp_path, session_factory, "alt_above.jpg")
    fake_tags = {
        "GPS GPSAltitude": _MockTag([_Ratio(1000, 1)]),
        "GPS GPSAltitudeRef": _MockTag("0"),
    }
    _run_task_with_tags(session_factory, vector_store, tmp_app_paths, fake_tags)

    meta = _fetch_metadata(db_engine, img_id)
    assert meta is not None
    assert meta.gps_altitude == pytest.approx(1000.0)


def test_exif_gps_altitude_below_sea_level(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """Altitude ref 1 (below sea level) → negated gps_altitude."""
    _, img_id = _make_indexed_image(tmp_path, session_factory, "alt_below.jpg")
    fake_tags = {
        "GPS GPSAltitude": _MockTag([_Ratio(50, 1)]),
        "GPS GPSAltitudeRef": _MockTag("1"),
    }
    _run_task_with_tags(session_factory, vector_store, tmp_app_paths, fake_tags)

    meta = _fetch_metadata(db_engine, img_id)
    assert meta is not None
    assert meta.gps_altitude == pytest.approx(-50.0)


def test_exif_gps_altitude_exception_silenced(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """Malformed altitude tag is silently ignored; gps_altitude stays None."""

    class _BadAltTag:
        values = [
            object()
        ]  # no .num/.den → AttributeError inside _extract_exif_metadata

        def __str__(self) -> str:
            return "bad"

    _, img_id = _make_indexed_image(tmp_path, session_factory, "alt_err.jpg")
    fake_tags = {"GPS GPSAltitude": _BadAltTag()}
    _run_task_with_tags(session_factory, vector_store, tmp_app_paths, fake_tags)

    meta = _fetch_metadata(db_engine, img_id)
    assert meta is not None
    assert meta.gps_altitude is None


def test_exif_orientation_stored(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """Orientation from EXIF is stored correctly."""
    _, img_id = _make_indexed_image(tmp_path, session_factory, "orient.jpg")
    fake_tags = {"Image Orientation": _MockTag([6])}
    _run_task_with_tags(session_factory, vector_store, tmp_app_paths, fake_tags)

    meta = _fetch_metadata(db_engine, img_id)
    assert meta is not None
    assert meta.orientation == 6


def test_exif_orientation_invalid_falls_back_to_1(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """Non-integer orientation value falls back to default (1)."""

    class _BadOrientTag:
        values = ["not-an-int"]

        def __str__(self) -> str:
            return "bad"

    _, img_id = _make_indexed_image(tmp_path, session_factory, "orient_inv.jpg")
    fake_tags = {"Image Orientation": _BadOrientTag()}
    _run_task_with_tags(session_factory, vector_store, tmp_app_paths, fake_tags)

    meta = _fetch_metadata(db_engine, img_id)
    assert meta is not None
    assert meta.orientation == 1


def test_filepath_date_extracted_when_no_exif(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """taken_at_source == FILEPATH when EXIF is absent but path matches the pattern."""
    # Place image in a dated subdirectory that the pattern will match
    dated_dir = tmp_path / "2023-07-14"
    dated_dir.mkdir()
    img_file = dated_dir / "photo.jpg"
    PILImage.new("RGB", (50, 50)).save(img_file, "JPEG")
    img_id = _insert_image(session_factory, img_file)

    task = _make_task(
        session_factory,
        vector_store,
        tmp_app_paths,
        filepath_date_pattern="{YYYY}-{MM}-{DD}",
    )
    # No EXIF tags → filepath pattern should be used
    with patch("photoaident.core.indexing.exifread.process_file", return_value={}):
        task.run()

    meta = _fetch_metadata(db_engine, img_id)
    assert meta is not None
    assert meta.taken_at_source == TakenAtSource.FILEPATH
    assert meta.taken_at == datetime(2023, 7, 14)


def test_indexing_task_invalid_filepath_date_pattern_does_not_raise(
    session_factory, vector_store, tmp_app_paths
):
    """IndexingTask.__init__ must not raise when given an invalid pattern."""
    task = IndexingTask(
        session_factory,
        vector_store,
        tmp_app_paths,
        ctx_id=-1,
        filepath_date_pattern="no-placeholders-here",
    )
    assert task._compiled_pattern is None


def test_faiss_add_failure_leaves_no_orphaned_faces(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """If vector_store.add() raises, no Face rows are committed and FAISS stays empty.

    Regression test for the DB↔FAISS divergence bug where session.flush() before
    vector_store.add() could leave flushed Face rows committed without a corresponding
    FAISS vector when the exception was caught in run() and session.commit() executed.
    """
    img_file = _make_image_file(tmp_path, "faiss_fail.jpg", size=(50, 50))
    img_id = _insert_image(session_factory, img_file)

    task = IndexingTask(session_factory, vector_store, tmp_app_paths, ctx_id=-1)
    mock_embedder = MagicMock()
    mock_embedder.process_image.return_value = [
        {
            "bbox": [0, 0, 10, 10],
            "embedding": _rng.random(VectorStore.DEFAULT_DIMENSION).astype(
                VectorStore.EMBEDDING_DTYPE
            ),
            "det_score": 0.9,
        }
    ]
    task._embedder = mock_embedder

    with patch.object(vector_store, "add", side_effect=RuntimeError("faiss boom")):
        task.run()

    # Image should be marked ERROR (exception handled inside run())
    with Session(db_engine) as session:
        img = session.get(Image, img_id)
        assert img is not None
        assert img.file_hash == "ERROR"
        face_count = session.execute(
            select(func.count(FaceModel.id)).where(FaceModel.image_id == img_id)
        ).scalar_one()

    assert face_count == 0, "No Face rows should be committed when FAISS add fails"
    assert vector_store.index.ntotal == 0, "FAISS index must remain empty"


def test_faiss_save_failure_rolls_back_faces_and_vectors(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """If vector_store.save() fails after faces were added, both DB rows and
    in-memory FAISS vectors are cleaned up so no divergence remains.

    Regression test: previously the exception propagated to run() which did
    session.commit() with file_hash=ERROR, accidentally committing the Face
    rows while FAISS on disk stayed stale.
    """
    img_file = _make_image_file(tmp_path, "save_fail.jpg", size=(50, 50))
    img_id = _insert_image(session_factory, img_file)

    task = IndexingTask(session_factory, vector_store, tmp_app_paths, ctx_id=-1)
    mock_embedder = MagicMock()
    mock_embedder.process_image.return_value = [
        {
            "bbox": [0, 0, 10, 10],
            "embedding": _rng.random(VectorStore.DEFAULT_DIMENSION).astype(
                VectorStore.EMBEDDING_DTYPE
            ),
            "det_score": 0.9,
        }
    ]
    task._embedder = mock_embedder

    # First save() call (inside _index_single_image) fails; second call (end of
    # run()) succeeds — use side_effect list to model this.
    with patch.object(vector_store, "save", side_effect=[OSError("disk full"), None]):
        task.run()

    with Session(db_engine) as session:
        img = session.get(Image, img_id)
        assert img is not None
        assert img.file_hash == "ERROR"
        face_count = session.execute(
            select(func.count(FaceModel.id)).where(FaceModel.image_id == img_id)
        ).scalar_one()

    assert face_count == 0, "Face rows must be rolled back when FAISS save fails"
    assert (
        vector_store.index.ntotal == 0
    ), "FAISS vectors must be removed on save failure"


# ---------------------------------------------------------------------------
# Stale face cleanup
# ---------------------------------------------------------------------------


def test_stale_faces_soft_deleted_and_removed_from_faiss(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """Faces from a previous index pass that are not re-detected are soft-deleted and
    their FAISS vectors are removed."""
    _img_id, old_face_id, _embedding = _setup_image_with_stale_face(
        tmp_path, session_factory, vector_store, "reindex.jpg"
    )

    _make_task(session_factory, vector_store, tmp_app_paths).run()

    with Session(db_engine) as session:
        old_face = session.get(FaceModel, old_face_id)
    assert old_face is not None
    assert old_face.deleted_at is not None, "Stale face must be soft-deleted"
    assert vector_store.index.ntotal == 0, "Stale FAISS vector must be removed"


def test_stale_face_rollback_restores_faiss_on_downstream_failure(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """If FAISS save fails after stale-face removal, the stale embedding is restored
    in FAISS and the soft-delete is rolled back in the DB."""
    _img_id, old_face_id, _embedding = _setup_image_with_stale_face(
        tmp_path, session_factory, vector_store, "rollback_stale.jpg"
    )

    task = _make_task(session_factory, vector_store, tmp_app_paths)

    # First save() raises to trigger the downstream failure path; second call (end of
    # run()) must succeed so run() can finish cleanly.
    with patch.object(vector_store, "save", side_effect=[OSError("disk full"), None]):
        task.run()

    # DB: soft-delete must be rolled back.
    with Session(db_engine) as session:
        old_face = session.get(FaceModel, old_face_id)
    assert old_face is not None
    assert old_face.deleted_at is None, "Rolled-back face must not be soft-deleted"

    # FAISS: stale embedding must be restored.
    assert (
        vector_store.index.ntotal == 1
    ), "Stale FAISS vector must be restored on rollback"
