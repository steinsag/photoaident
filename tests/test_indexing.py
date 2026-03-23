from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image as PILImage
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from photoaident.core.indexing import IndexingTask, _bbox_iou, _dms_to_decimal
from photoaident.db.database import (
    EmbeddingCluster,
    Face,
    FaceState,
    Image,
    ImageMetadata,
    ImageTag,
    Person,
    TagSource,
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


def _make_indexed_image(tmp_path, session_factory, filename="img.jpg"):
    """Create a real JPEG on disk and a matching unindexed Image row."""
    img_file = tmp_path / filename
    PILImage.new("RGB", (100, 100)).save(img_file, "JPEG")
    with session_factory() as session:
        img = Image(file_path=str(img_file), file_size=img_file.stat().st_size)
        session.add(img)
        session.commit()
        img_id = img.id
    return img_file, img_id


def _run_task_with_tags(session_factory, vector_store, tmp_paths, fake_tags):
    """Run IndexingTask with mocked exifread returning fake_tags."""
    task = IndexingTask(session_factory, vector_store, tmp_paths, ctx_id=-1)
    mock_embedder = MagicMock()
    mock_embedder.process_image.return_value = []
    task._embedder = mock_embedder
    with patch(
        "photoaident.core.indexing.exifread.process_file", return_value=fake_tags
    ):
        task.run()


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
    from photoaident.db.database import Image, ImageMetadata

    with Session(db_engine) as session:
        # Delete child tables first (FK constraints, even without enforcement)
        session.execute(delete(ImageMetadata))
        session.execute(delete(Image))
        session.commit()


def test_indexing_task_success(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    # Setup: Create an image that needs indexing
    img_file = tmp_path / "img1.jpg"
    PILImage.new("RGB", (100, 100), "red").save(img_file, "JPEG")

    with session_factory() as session:
        img = Image(file_path=str(img_file), file_size=len(b"dummy image data"))
        session.add(img)
        session.commit()
        img_id = img.id

    task = IndexingTask(session_factory, vector_store, tmp_app_paths, ctx_id=-1)

    # Mock FaceEmbedder to avoid running the real model
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
    # Verify FAISS
    assert vector_store.index.ntotal == 1

    # Verify crop was saved
    crop_path = tmp_app_paths.face_crops_dir / f"{face.id}.jpg"
    assert crop_path.exists()

    # Verify no full-photo thumbnail was created during indexing (lazy generation only)
    assert not any(tmp_app_paths.thumbs_dir.iterdir())

    # Verify image_metadata was populated
    with Session(db_engine) as session:
        meta = session.execute(
            select(ImageMetadata).where(ImageMetadata.image_id == img_id)
        ).scalar_one_or_none()
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
    img1 = tmp_path / "img1_inner.jpg"
    img2 = tmp_path / "img2_inner.jpg"
    PILImage.new("RGB", (10, 10), "red").save(img1)
    PILImage.new("RGB", (10, 10), "blue").save(img2)

    with session_factory() as session:
        session.add(Image(file_path=str(img1), file_size=img1.stat().st_size))
        session.add(Image(file_path=str(img2), file_size=img2.stat().st_size))
        session.commit()

    task = IndexingTask(session_factory, vector_store, tmp_app_paths, ctx_id=-1)

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
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """An image whose file no longer exists on disk is marked MISSING."""
    nonexistent = tmp_path / "gone.jpg"
    # Deliberately do NOT create the file.

    with session_factory() as session:
        img = Image(file_path=str(nonexistent), file_size=100)
        session.add(img)
        session.commit()
        img_id = img.id

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
    img_file = tmp_path / "err.jpg"
    PILImage.new("RGB", (10, 10)).save(img_file)

    with session_factory() as session:
        img = Image(file_path=str(img_file), file_size=img_file.stat().st_size)
        session.add(img)
        session.commit()
        img_id = img.id

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
    img_file = tmp_path / "meta_test.jpg"
    PILImage.new("RGB", (320, 240), "green").save(img_file, "JPEG")

    with session_factory() as session:
        img = Image(file_path=str(img_file), file_size=img_file.stat().st_size)
        session.add(img)
        session.commit()
        img_id = img.id

    task = IndexingTask(session_factory, vector_store, tmp_app_paths, ctx_id=-1)
    mock_embedder = MagicMock()
    mock_embedder.process_image.return_value = []
    task._embedder = mock_embedder

    task.run()

    with Session(db_engine) as session:
        meta = session.execute(
            select(ImageMetadata).where(ImageMetadata.image_id == img_id)
        ).scalar_one_or_none()
    assert meta is not None
    assert meta.width == 320
    assert meta.height == 240
    # No EXIF and no filepath pattern → no date
    assert meta.taken_at is None
    assert meta.taken_at_source is None


def test_indexing_task_no_thumbnail_during_indexing(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """No thumbnail files are written to thumbs_dir during indexing."""
    img_file = tmp_path / "no_thumb.jpg"
    PILImage.new("RGB", (50, 50), "blue").save(img_file, "JPEG")

    with session_factory() as session:
        img = Image(file_path=str(img_file), file_size=img_file.stat().st_size)
        session.add(img)
        session.commit()

    task = IndexingTask(session_factory, vector_store, tmp_app_paths, ctx_id=-1)
    mock_embedder = MagicMock()
    mock_embedder.process_image.return_value = []
    task._embedder = mock_embedder

    task.run()

    assert not any(tmp_app_paths.thumbs_dir.iterdir())


def test_indexing_task_exif_failure_does_not_abort(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """EXIF extraction failure must not abort indexing — image is still marked done."""
    img_file = tmp_path / "exif_fail.jpg"
    PILImage.new("RGB", (10, 10)).save(img_file, "JPEG")

    with session_factory() as session:
        img = Image(file_path=str(img_file), file_size=img_file.stat().st_size)
        session.add(img)
        session.commit()
        img_id = img.id

    task = IndexingTask(session_factory, vector_store, tmp_app_paths, ctx_id=-1)
    mock_embedder = MagicMock()
    mock_embedder.process_image.return_value = []
    task._embedder = mock_embedder

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

    with Session(db_engine) as session:
        meta = session.execute(
            select(ImageMetadata).where(ImageMetadata.image_id == img_id)
        ).scalar_one_or_none()
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

    with Session(db_engine) as session:
        meta = session.execute(
            select(ImageMetadata).where(ImageMetadata.image_id == img_id)
        ).scalar_one_or_none()
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

    with Session(db_engine) as session:
        meta = session.execute(
            select(ImageMetadata).where(ImageMetadata.image_id == img_id)
        ).scalar_one_or_none()
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

    with Session(db_engine) as session:
        meta = session.execute(
            select(ImageMetadata).where(ImageMetadata.image_id == img_id)
        ).scalar_one_or_none()
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

    with Session(db_engine) as session:
        meta = session.execute(
            select(ImageMetadata).where(ImageMetadata.image_id == img_id)
        ).scalar_one_or_none()
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

    with Session(db_engine) as session:
        meta = session.execute(
            select(ImageMetadata).where(ImageMetadata.image_id == img_id)
        ).scalar_one_or_none()
    assert meta is not None
    assert meta.gps_altitude is None


def test_exif_orientation_stored(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """Orientation from EXIF is stored correctly."""
    _, img_id = _make_indexed_image(tmp_path, session_factory, "orient.jpg")
    fake_tags = {"Image Orientation": _MockTag([6])}
    _run_task_with_tags(session_factory, vector_store, tmp_app_paths, fake_tags)

    with Session(db_engine) as session:
        meta = session.execute(
            select(ImageMetadata).where(ImageMetadata.image_id == img_id)
        ).scalar_one_or_none()
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

    with Session(db_engine) as session:
        meta = session.execute(
            select(ImageMetadata).where(ImageMetadata.image_id == img_id)
        ).scalar_one_or_none()
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

    with session_factory() as session:
        img = Image(file_path=str(img_file), file_size=img_file.stat().st_size)
        session.add(img)
        session.commit()
        img_id = img.id

    task = IndexingTask(
        session_factory,
        vector_store,
        tmp_app_paths,
        ctx_id=-1,
        filepath_date_pattern="{YYYY}-{MM}-{DD}",
    )
    mock_embedder = MagicMock()
    mock_embedder.process_image.return_value = []
    task._embedder = mock_embedder
    # No EXIF tags → filepath pattern should be used
    with patch("photoaident.core.indexing.exifread.process_file", return_value={}):
        task.run()

    with Session(db_engine) as session:
        meta = session.execute(
            select(ImageMetadata).where(ImageMetadata.image_id == img_id)
        ).scalar_one_or_none()
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
    img_file = tmp_path / "faiss_fail.jpg"
    PILImage.new("RGB", (50, 50)).save(img_file, "JPEG")

    with session_factory() as session:
        img = Image(file_path=str(img_file), file_size=img_file.stat().st_size)
        session.add(img)
        session.commit()
        img_id = img.id

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

    from photoaident.db.database import Face as FaceModel

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
    img_file = tmp_path / "save_fail.jpg"
    PILImage.new("RGB", (50, 50)).save(img_file, "JPEG")

    with session_factory() as session:
        img = Image(file_path=str(img_file), file_size=img_file.stat().st_size)
        session.add(img)
        session.commit()
        img_id = img.id

    task = IndexingTask(session_factory, vector_store, tmp_app_paths, ctx_id=-1)
    mock_embedder = MagicMock()
    embedding = _rng.random(VectorStore.DEFAULT_DIMENSION).astype(
        VectorStore.EMBEDDING_DTYPE
    )
    mock_embedder.process_image.return_value = [
        {
            "bbox": [0, 0, 10, 10],
            "embedding": embedding,
            "det_score": 0.9,
        }
    ]
    task._embedder = mock_embedder

    # First save() call (inside _index_single_image) fails; second call (end of
    # run()) succeeds — use side_effect list to model this.
    with patch.object(vector_store, "save", side_effect=[OSError("disk full"), None]):
        task.run()

    from photoaident.db.database import Face as FaceModel

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


# ===========================================================================
# _bbox_iou
# ===========================================================================


def test_bbox_iou_perfect_overlap():
    """Identical boxes produce IoU of 1.0."""
    assert _bbox_iou((10, 20, 50, 60), (10, 20, 50, 60)) == pytest.approx(1.0)


def test_bbox_iou_no_overlap():
    """Disjoint boxes produce IoU of 0.0."""
    assert _bbox_iou((0, 0, 10, 10), (100, 100, 10, 10)) == pytest.approx(0.0)


def test_bbox_iou_partial_overlap():
    """Partially overlapping boxes produce correct IoU."""
    # Box A: (0,0) to (10,10), area 100
    # Box B: (5,5) to (15,15), area 100
    # Intersection: (5,5) to (10,10) = 25
    # Union: 100 + 100 - 25 = 175
    assert _bbox_iou((0, 0, 10, 10), (5, 5, 10, 10)) == pytest.approx(25 / 175)


def test_bbox_iou_zero_area_box():
    """A zero-area box produces IoU of 0.0."""
    assert _bbox_iou((0, 0, 0, 10), (0, 0, 10, 10)) == pytest.approx(0.0)


def test_bbox_iou_one_inside_other():
    """A box fully contained in another produces correct IoU."""
    # Box A: (0,0) to (20,20), area 400
    # Box B: (5,5) to (15,15), area 100
    # Intersection: 100
    # Union: 400 + 100 - 100 = 400
    assert _bbox_iou((0, 0, 20, 20), (5, 5, 10, 10)) == pytest.approx(100 / 400)


def test_bbox_iou_adjacent_boxes():
    """Touching but non-overlapping boxes produce IoU of 0.0."""
    assert _bbox_iou((0, 0, 10, 10), (10, 0, 10, 10)) == pytest.approx(0.0)


# ===========================================================================
# _find_matching_reference
# ===========================================================================


def _make_face(
    face_id: int,
    bbox: tuple[int, int, int, int],
    state: FaceState = FaceState.IDENTIFIED,
    cluster_id: int | None = 1,
    person_id: int | None = 1,
) -> Face:
    """Build a detached Face object for unit tests (no DB session needed)."""
    f = Face(
        bbox_x=bbox[0],
        bbox_y=bbox[1],
        bbox_w=bbox[2],
        bbox_h=bbox[3],
        detection_confidence=0.9,
        model_version="test-v1",
        state=state,
        cluster_id=cluster_id,
        person_id=person_id,
        image_id=1,
    )
    # Set the id directly (normally assigned by DB)
    f.id = face_id
    return f


def test_find_matching_reference_found():
    """Matching reference face with sufficient IoU is returned."""
    ref = _make_face(10, (0, 0, 100, 100))
    result = IndexingTask._find_matching_reference((5, 5, 100, 100), [ref], set())
    assert result is ref


def test_find_matching_reference_low_iou():
    """No match when IoU is below threshold."""
    ref = _make_face(10, (0, 0, 10, 10))
    result = IndexingTask._find_matching_reference((500, 500, 10, 10), [ref], set())
    assert result is None


def test_find_matching_reference_already_matched_skipped():
    """Already-matched face is skipped even if IoU is high."""
    ref = _make_face(10, (0, 0, 100, 100))
    result = IndexingTask._find_matching_reference((0, 0, 100, 100), [ref], {10})
    assert result is None


def test_find_matching_reference_returns_first_match():
    """First matching reference is returned when multiple could match."""
    ref1 = _make_face(10, (0, 0, 100, 100))
    ref2 = _make_face(20, (5, 5, 100, 100))
    result = IndexingTask._find_matching_reference(
        (0, 0, 100, 100), [ref1, ref2], set()
    )
    assert result is ref1


def test_find_matching_reference_skips_first_returns_second():
    """When first match is already used, second matching reference is returned."""
    ref1 = _make_face(10, (0, 0, 100, 100))
    ref2 = _make_face(20, (0, 0, 100, 100))
    result = IndexingTask._find_matching_reference((0, 0, 100, 100), [ref1, ref2], {10})
    assert result is ref2


def test_find_matching_reference_empty_list():
    """Empty reference list returns None."""
    result = IndexingTask._find_matching_reference((0, 0, 100, 100), [], set())
    assert result is None


# ===========================================================================
# _cleanup_stale_faces
# ===========================================================================


def _make_cleanup_task(session_factory, vector_store, tmp_app_paths):
    """Build an IndexingTask for cleanup tests."""
    task = IndexingTask(session_factory, vector_store, tmp_app_paths, ctx_id=-1)
    task._embedder = MagicMock()
    return task


def _insert_person_with_cluster(session) -> tuple[int, int]:
    """Insert a Person with one EmbeddingCluster, return (person_id, cluster_id)."""
    person = Person(name="Test Person")
    session.add(person)
    session.flush()
    cluster = EmbeddingCluster(person_id=person.id, label="adult", age_group="adult")
    session.add(cluster)
    session.flush()
    return person.id, cluster.id


def test_cleanup_no_existing_faces(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """No existing faces returns empty list without errors."""
    task = _make_cleanup_task(session_factory, vector_store, tmp_app_paths)
    img_file = tmp_path / "clean.jpg"
    PILImage.new("RGB", (10, 10)).save(img_file)

    with session_factory() as session:
        img = Image(file_path=str(img_file), file_size=100)
        session.add(img)
        session.flush()

        result = task._cleanup_stale_faces(img, session)
        assert result == []
        session.commit()


def test_cleanup_only_unidentified_faces_deleted(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """All unidentified faces are deleted from DB and FAISS."""
    task = _make_cleanup_task(session_factory, vector_store, tmp_app_paths)
    img_file = tmp_path / "unid.jpg"
    PILImage.new("RGB", (10, 10)).save(img_file)

    emb = _rng.random(VectorStore.DEFAULT_DIMENSION).astype(VectorStore.EMBEDDING_DTYPE)

    with session_factory() as session:
        img = Image(file_path=str(img_file), file_size=100, file_hash="oldhash")
        session.add(img)
        session.flush()

        face = Face(
            image_id=img.id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=10,
            bbox_h=10,
            detection_confidence=0.9,
            model_version="v1",
            state=FaceState.UNIDENTIFIED,
        )
        session.add(face)
        session.flush()
        face_id = face.id

        # Add to FAISS
        vector_store.add(face_id, emb)
        assert vector_store.index.ntotal == 1

        result = task._cleanup_stale_faces(img, session)
        session.commit()

    assert result == []
    assert vector_store.index.ntotal == 0

    # Verify face is gone from DB
    with Session(db_engine) as session:
        count = session.execute(
            select(func.count(Face.id)).where(Face.id == face_id)
        ).scalar_one()
    assert count == 0


def test_cleanup_preserves_identified_faces_with_cluster(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """IDENTIFIED faces with cluster_id are preserved."""
    task = _make_cleanup_task(session_factory, vector_store, tmp_app_paths)
    img_file = tmp_path / "ident.jpg"
    PILImage.new("RGB", (10, 10)).save(img_file)

    with session_factory() as session:
        img = Image(file_path=str(img_file), file_size=100, file_hash="oldhash")
        session.add(img)
        session.flush()

        person_id, cluster_id = _insert_person_with_cluster(session)

        identified_face = Face(
            image_id=img.id,
            bbox_x=10,
            bbox_y=10,
            bbox_w=50,
            bbox_h=50,
            detection_confidence=0.95,
            model_version="v1",
            state=FaceState.IDENTIFIED,
            person_id=person_id,
            cluster_id=cluster_id,
        )
        unidentified_face = Face(
            image_id=img.id,
            bbox_x=200,
            bbox_y=200,
            bbox_w=30,
            bbox_h=30,
            detection_confidence=0.8,
            model_version="v1",
            state=FaceState.UNIDENTIFIED,
        )
        session.add_all([identified_face, unidentified_face])
        session.flush()
        ident_id = identified_face.id
        unident_id = unidentified_face.id

        result = task._cleanup_stale_faces(img, session)
        result_ids = [f.id for f in result]
        session.commit()

    assert len(result_ids) == 1
    assert result_ids[0] == ident_id

    with Session(db_engine) as session:
        # Identified face still exists
        assert session.get(Face, ident_id) is not None
        # Unidentified face was deleted
        assert session.get(Face, unident_id) is None


def test_cleanup_identified_without_cluster_is_deleted(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """IDENTIFIED face without cluster_id is treated as disposable."""
    task = _make_cleanup_task(session_factory, vector_store, tmp_app_paths)
    img_file = tmp_path / "no_cluster.jpg"
    PILImage.new("RGB", (10, 10)).save(img_file)

    with session_factory() as session:
        img = Image(file_path=str(img_file), file_size=100, file_hash="oldhash")
        session.add(img)
        session.flush()

        person = Person(name="Orphan Person")
        session.add(person)
        session.flush()

        face = Face(
            image_id=img.id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=10,
            bbox_h=10,
            detection_confidence=0.9,
            model_version="v1",
            state=FaceState.IDENTIFIED,
            person_id=person.id,
            cluster_id=None,  # no cluster!
        )
        session.add(face)
        session.flush()
        face_id = face.id

        result = task._cleanup_stale_faces(img, session)
        session.commit()

    assert result == []

    with Session(db_engine) as session:
        assert session.get(Face, face_id) is None


def test_cleanup_deletes_face_crop_files(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """Face crop JPEG files are deleted for disposable faces."""
    task = _make_cleanup_task(session_factory, vector_store, tmp_app_paths)
    img_file = tmp_path / "crop.jpg"
    PILImage.new("RGB", (10, 10)).save(img_file)

    with session_factory() as session:
        img = Image(file_path=str(img_file), file_size=100, file_hash="oldhash")
        session.add(img)
        session.flush()

        face = Face(
            image_id=img.id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=10,
            bbox_h=10,
            detection_confidence=0.9,
            model_version="v1",
            state=FaceState.UNIDENTIFIED,
        )
        session.add(face)
        session.flush()

        # Create a fake crop file
        crop_path = tmp_app_paths.face_crops_dir / f"{face.id}.jpg"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop_path.write_bytes(b"fake crop data")
        assert crop_path.exists()

        task._cleanup_stale_faces(img, session)
        session.commit()

    assert not crop_path.exists()


def test_cleanup_deletes_metadata_and_tags(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """ImageMetadata and ImageTag rows are deleted during cleanup."""
    task = _make_cleanup_task(session_factory, vector_store, tmp_app_paths)
    img_file = tmp_path / "meta.jpg"
    PILImage.new("RGB", (10, 10)).save(img_file)

    with session_factory() as session:
        img = Image(file_path=str(img_file), file_size=100, file_hash="oldhash")
        session.add(img)
        session.flush()

        # A face must exist for cleanup to proceed past the early return
        face = Face(
            image_id=img.id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=10,
            bbox_h=10,
            detection_confidence=0.9,
            model_version="v1",
            state=FaceState.UNIDENTIFIED,
        )
        meta = ImageMetadata(image_id=img.id, width=10, height=10, orientation=1)
        tag = ImageTag(
            image_id=img.id,
            tag_key="scene",
            tag_value="outdoor",
            tag_source=TagSource.MODEL,
        )
        session.add_all([face, meta, tag])
        session.flush()

        task._cleanup_stale_faces(img, session)
        img_id = img.id
        session.commit()

    with Session(db_engine) as session:
        meta_count = session.execute(
            select(func.count(ImageMetadata.id)).where(ImageMetadata.image_id == img_id)
        ).scalar_one()
        tag_count = session.execute(
            select(func.count(ImageTag.id)).where(ImageTag.image_id == img_id)
        ).scalar_one()

    assert meta_count == 0
    assert tag_count == 0


# ===========================================================================
# _index_single_image — IoU-based reference face matching
# ===========================================================================


def test_reindex_updates_matching_reference_face(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """Reindexing updates a reference face matched by IoU."""
    img_file = tmp_path / "reindex.jpg"
    PILImage.new("RGB", (100, 100), "red").save(img_file, "JPEG")

    with session_factory() as session:
        img = Image(file_path=str(img_file), file_size=img_file.stat().st_size)
        session.add(img)
        session.flush()

        person_id, cluster_id = _insert_person_with_cluster(session)

        old_emb = _rng.random(VectorStore.DEFAULT_DIMENSION).astype(
            VectorStore.EMBEDDING_DTYPE
        )
        ref_face = Face(
            image_id=img.id,
            bbox_x=10,
            bbox_y=10,
            bbox_w=50,
            bbox_h=50,
            detection_confidence=0.8,
            model_version="old-v1",
            state=FaceState.IDENTIFIED,
            person_id=person_id,
            cluster_id=cluster_id,
        )
        session.add(ref_face)
        session.flush()
        ref_face_id = ref_face.id
        vector_store.add(ref_face_id, old_emb)
        session.commit()

    new_emb = _rng.random(VectorStore.DEFAULT_DIMENSION).astype(
        VectorStore.EMBEDDING_DTYPE
    )

    task = IndexingTask(session_factory, vector_store, tmp_app_paths, ctx_id=-1)
    mock_embedder = MagicMock()
    # Detection bbox overlaps the reference face (IoU > 0.5)
    mock_embedder.process_image.return_value = [
        {
            "bbox": [
                12,
                12,
                58,
                58,
            ],  # (12,12,46,46) in xywh — high IoU with (10,10,50,50)
            "embedding": new_emb,
            "det_score": 0.95,
        }
    ]
    mock_embedder.extract_face_crop.return_value = PILImage.new(
        "RGB", (112, 112), "blue"
    )
    task._embedder = mock_embedder

    task.run()

    with Session(db_engine) as session:
        face = session.get(Face, ref_face_id)
        assert face is not None
        # State and assignment preserved
        assert face.state == FaceState.IDENTIFIED
        assert face.person_id == person_id
        assert face.cluster_id == cluster_id
        # Bbox and confidence updated
        assert face.bbox_x == 12
        assert face.bbox_y == 12
        assert face.detection_confidence == pytest.approx(0.95)
        assert face.model_version == "buffalo_l"

        # No new face was created
        all_faces = (
            session.execute(select(Face).where(Face.image_id == face.image_id))
            .scalars()
            .all()
        )
        assert len(all_faces) == 1


def test_reindex_creates_new_face_when_no_iou_match(
    tmp_path, session_factory, db_engine, vector_store, tmp_app_paths
):
    """Detection far from reference face creates a new UNIDENTIFIED face."""
    img_file = tmp_path / "reindex_new.jpg"
    PILImage.new("RGB", (500, 500), "red").save(img_file, "JPEG")

    with session_factory() as session:
        img = Image(file_path=str(img_file), file_size=img_file.stat().st_size)
        session.add(img)
        session.flush()

        person_id, cluster_id = _insert_person_with_cluster(session)

        ref_face = Face(
            image_id=img.id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=50,
            bbox_h=50,
            detection_confidence=0.8,
            model_version="old-v1",
            state=FaceState.IDENTIFIED,
            person_id=person_id,
            cluster_id=cluster_id,
        )
        session.add(ref_face)
        session.flush()
        ref_face_id = ref_face.id
        img_id = img.id

        old_emb = _rng.random(VectorStore.DEFAULT_DIMENSION).astype(
            VectorStore.EMBEDDING_DTYPE
        )
        vector_store.add(ref_face_id, old_emb)
        session.commit()

    new_emb = _rng.random(VectorStore.DEFAULT_DIMENSION).astype(
        VectorStore.EMBEDDING_DTYPE
    )

    task = IndexingTask(session_factory, vector_store, tmp_app_paths, ctx_id=-1)
    mock_embedder = MagicMock()
    # Detection bbox far from reference face — no IoU match
    mock_embedder.process_image.return_value = [
        {
            "bbox": [400, 400, 450, 450],  # far from (0,0,50,50)
            "embedding": new_emb,
            "det_score": 0.9,
        }
    ]
    mock_embedder.extract_face_crop.return_value = PILImage.new(
        "RGB", (112, 112), "green"
    )
    task._embedder = mock_embedder

    task.run()

    with Session(db_engine) as session:
        all_faces = (
            session.execute(select(Face).where(Face.image_id == img_id)).scalars().all()
        )
        # Reference face preserved + one new face created
        assert len(all_faces) == 2

        new_face = [f for f in all_faces if f.id != ref_face_id]
        assert len(new_face) == 1
        assert new_face[0].state == FaceState.UNIDENTIFIED
        assert new_face[0].person_id is None
