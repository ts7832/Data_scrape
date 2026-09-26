"""Local flat-earth projection: accurate to well under 0.1% at the few-km scale Kuulo fuses at."""

from __future__ import annotations

from math import cos, hypot, pi, radians, sin
from typing import Protocol

from .models import GeoPoint

M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON_EQUATOR = 111_320.0


class HasLatLon(Protocol):
    lat: float
    lon: float


def to_local(origin: HasLatLon, point: HasLatLon) -> tuple[float, float]:
    """Metres (east, north) of `point` relative to `origin`."""
    x = (point.lon - origin.lon) * M_PER_DEG_LON_EQUATOR * cos(radians(origin.lat))
    y = (point.lat - origin.lat) * M_PER_DEG_LAT
    return x, y


def from_local(origin: HasLatLon, x: float, y: float) -> GeoPoint:
    return GeoPoint(
        lat=origin.lat + y / M_PER_DEG_LAT,
        lon=origin.lon + x / (M_PER_DEG_LON_EQUATOR * cos(radians(origin.lat))),
    )


def distance_m(a: HasLatLon, b: HasLatLon) -> float:
    return hypot(*to_local(a, b))


def ellipse_ring(
    center: HasLatLon, semi_major_m: float, semi_minor_m: float, bearing_deg: float,
    steps: int = 64,
) -> list[GeoPoint]:
    """Closed ring of points on an ellipse; bearing is the major axis, clockwise from north."""
    th = radians(bearing_deg)
    ux, uy = sin(th), cos(th)  # major axis (east, north)
    vx, vy = cos(th), -sin(th)  # minor axis, perpendicular
    ring = []
    for k in range(steps):
        a = 2 * pi * k / steps
        x = semi_major_m * cos(a) * ux + semi_minor_m * sin(a) * vx
        y = semi_major_m * cos(a) * uy + semi_minor_m * sin(a) * vy
        ring.append(from_local(center, x, y))
    ring.append(ring[0])
    return ring
