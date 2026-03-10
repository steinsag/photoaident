from photoaident.core.geo import GpsBoundingBox


def test_gps_bounding_box_contains():
    # Central Europe box
    bbox = GpsBoundingBox(south=45.0, west=5.0, north=55.0, east=20.0)

    assert bbox.contains(50.0, 10.0) is True
    assert bbox.contains(44.9, 10.0) is False
    assert bbox.contains(55.1, 10.0) is False
    assert bbox.contains(50.0, 4.9) is False
    assert bbox.contains(50.0, 20.1) is False

    # Boundary cases
    assert bbox.contains(45.0, 5.0) is True
    assert bbox.contains(55.0, 20.0) is True


def test_gps_bounding_box_antimeridian():
    # Box crossing the 180/-180 meridian (e.g. around Fiji/Tonga)
    bbox = GpsBoundingBox(south=-20.0, west=170.0, north=-10.0, east=-170.0)

    assert bbox.contains(-15.0, 175.0) is True  # East of 170
    assert bbox.contains(-15.0, -175.0) is True  # West of -170
    assert bbox.contains(-15.0, 0.0) is False
    assert bbox.contains(-15.0, 169.0) is False
    assert bbox.contains(-15.0, -169.0) is False
