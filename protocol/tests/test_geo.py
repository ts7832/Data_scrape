import pytest

from kuulo_protocol.geo import distance_m, from_local, to_local
from kuulo_protocol.models import GeoPoint

HKI = GeoPoint(lat=60.1699, lon=24.9384)


def test_zero_distance():
    assert distance_m(HKI, HKI) == 0


def test_600m_north_and_east():
    north = GeoPoint(lat=HKI.lat + 600 / 110_540, lon=HKI.lon)
    assert distance_m(HKI, north) == pytest.approx(600, rel=1e-3)
    x, y = to_local(HKI, from_local(HKI, 600, 0))
    assert x == pytest.approx(600, rel=1e-6) and y == pytest.approx(0, abs=1e-6)


def test_round_trip():
    p = from_local(HKI, -1234.5, 987.0)
    assert to_local(HKI, p) == pytest.approx((-1234.5, 987.0), rel=1e-9)
