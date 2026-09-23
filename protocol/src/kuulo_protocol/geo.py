"""Local flat-earth projection: accurate to well under 0.1% at the few-km scale Kuulo fuses at."""

from __future__ import annotations

from math import cos, hypot, radians
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
