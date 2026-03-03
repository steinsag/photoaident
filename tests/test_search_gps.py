import pytest

from photoaident.core.geo import GpsBoundingBox
from photoaident.core.search import find_images_by_gps_bbox
from photoaident.db.database import (
    Image,
    ImageMetadata,
    TakenAtSource,
    get_engine,
    get_session_factory,
)
from photoaident.db.migrate import apply_migrations


@pytest.fixture
def session_factory(tmp_path):
    db_path = tmp_path / "search_gps_test.db"
    engine = get_engine(str(db_path))
    apply_migrations(f"sqlite:///{db_path}")
    return get_session_factory(engine)


def test_find_images_by_gps_bbox(session_factory):
    with session_factory() as session:
        # Image 1: Inside box (Berlin-ish)
        img1 = Image(file_path="/img1.jpg", file_size=100, file_hash="h1")
        session.add(img1)
        session.flush()
        meta1 = ImageMetadata(
            image_id=img1.id,
            gps_lat=52.52,
            gps_lon=13.40,
            taken_at_source=TakenAtSource.FILESYSTEM,
            width=100,
            height=100,
        )
        session.add(meta1)

        # Image 2: Outside box (London-ish)
        img2 = Image(file_path="/img2.jpg", file_size=100, file_hash="h2")
        session.add(img2)
        session.flush()
        meta2 = ImageMetadata(
            image_id=img2.id,
            gps_lat=51.50,
            gps_lon=-0.12,
            taken_at_source=TakenAtSource.FILESYSTEM,
            width=100,
            height=100,
        )
        session.add(meta2)

        # Image 3: No GPS
        img3 = Image(file_path="/img3.jpg", file_size=100, file_hash="h3")
        session.add(img3)
        session.flush()
        meta3 = ImageMetadata(
            image_id=img3.id,
            taken_at_source=TakenAtSource.FILESYSTEM,
            width=100,
            height=100,
        )
        session.add(meta3)

        session.commit()

        # Bbox covering Germany
        bbox = GpsBoundingBox(south=47.0, west=5.0, north=55.0, east=15.0)
        results = find_images_by_gps_bbox(session_factory, bbox)

        assert len(results) == 1
        assert results[0] == img1.id


def test_find_images_by_gps_bbox_antimeridian(session_factory):
    with session_factory() as session:
        # Image 1: Inside box (Fiji area, crossing 180) - East of 180
        img1 = Image(file_path="/img1.jpg", file_size=100, file_hash="h1")
        session.add(img1)
        session.flush()
        meta1 = ImageMetadata(
            image_id=img1.id,
            gps_lat=-18.0,
            gps_lon=179.0,
            taken_at_source=TakenAtSource.FILESYSTEM,
            width=100,
            height=100,
        )
        session.add(meta1)

        # Image 2: Inside box (Fiji area, crossing 180) - West of -180
        img2 = Image(file_path="/img2.jpg", file_size=100, file_hash="h2")
        session.add(img2)
        session.flush()
        meta2 = ImageMetadata(
            image_id=img2.id,
            gps_lat=-18.0,
            gps_lon=-179.0,
            taken_at_source=TakenAtSource.FILESYSTEM,
            width=100,
            height=100,
        )
        session.add(meta2)

        # Image 3: Outside box (London)
        img3 = Image(file_path="/img3.jpg", file_size=100, file_hash="h3")
        session.add(img3)
        session.flush()
        meta3 = ImageMetadata(
            image_id=img3.id,
            gps_lat=51.50,
            gps_lon=-0.12,
            taken_at_source=TakenAtSource.FILESYSTEM,
            width=100,
            height=100,
        )
        session.add(meta3)

        session.commit()

        # Bbox crossing antimeridian
        bbox = GpsBoundingBox(south=-20.0, west=170.0, north=-10.0, east=-170.0)
        results = find_images_by_gps_bbox(session_factory, bbox)

        assert len(results) == 2
        assert set(results) == {img1.id, img2.id}
