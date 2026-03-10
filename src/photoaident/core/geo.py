from dataclasses import dataclass


@dataclass(frozen=True)
class GpsBoundingBox:
    """A GPS bounding box defined by its south, west, north, and east boundaries.

    All values are in degrees.
    """

    south: float  # min latitude
    west: float  # min longitude
    north: float  # max latitude
    east: float  # max longitude

    def contains(self, lat: float, lon: float) -> bool:
        """Check if a coordinate is within the bounding box."""
        # Simple case: doesn't cross the antimeridian
        if self.west <= self.east:
            return self.south <= lat <= self.north and self.west <= lon <= self.east

        # Crosses the antimeridian
        return self.south <= lat <= self.north and (
            lon >= self.west or lon <= self.east
        )
