import random

import pytest
from pydantic import ValidationError

from kuulo_protocol.geo import distance_m
from kuulo_protocol.models import GeoPoint, TimeQuality
from kuulo_sim.physics import (
    clock_offset_s,
    confidence_from_snr,
    detection_probability,
    propagation_delay_s,
    received_level_db,
)
from kuulo_sim.scenario import DroneSpec, Scenario, drone_position, node_offline


def test_farther_is_quieter():
    levels = [received_level_db(100, r) for r in (1, 10, 100, 500, 1000, 3000)]
    assert levels == sorted(levels, reverse=True)
    assert received_level_db(100, 0) == received_level_db(100, 1)


def test_detection_probability_shape():
    assert detection_probability(6.0) == pytest.approx(0.5)
    assert detection_probability(20) > 0.99
    assert detection_probability(-10) < 0.01


def test_confidence_bounds():
    rng = random.Random(0)
    values = [confidence_from_snr(snr, rng) for snr in range(-20, 60, 5) for _ in range(20)]
    assert all(0 <= v <= 1 for v in values)


def test_clock_offsets_scale_with_quality():
    rng = random.Random(1)
    gps = [abs(clock_offset_s(TimeQuality.GPS, rng)) for _ in range(200)]
    manual = [abs(clock_offset_s(TimeQuality.MANUAL, rng)) for _ in range(200)]
    assert max(gps) < 1e-5 < sum(manual) / len(manual)


def test_propagation_delay():
    assert propagation_delay_s(343) == pytest.approx(1.0)


def test_drone_position_along_route():
    d = DroneSpec(id="d", speed_mps=10, start_s=5, waypoints=[(60.17, 24.90), (60.17, 24.95)])
    start = GeoPoint(lat=60.17, lon=24.90)
    assert drone_position(d, 4.9) is None
    assert distance_m(drone_position(d, 5), start) == pytest.approx(0, abs=1e-6)
    assert distance_m(drone_position(d, 105), start) == pytest.approx(1000, rel=1e-3)
    assert drone_position(d, 10_000) is None  # route finished


def _scenario(**extra):
    base = {"name": "t", "duration_s": 10, "nodes": [{"id": "n1", "lat": 60.17, "lon": 24.94}]}
    base.update(extra)
    return Scenario.model_validate(base)


def test_duplicate_node_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate node id"):
        _scenario(nodes=[{"id": "n1", "lat": 60, "lon": 24}, {"id": "n1", "lat": 60, "lon": 25}])


def test_failure_must_reference_known_node():
    with pytest.raises(ValidationError, match="unknown node"):
        _scenario(node_failures=[{"node_id": "nope"}])


def test_node_offline_window():
    sc = _scenario(node_failures=[{"node_id": "n1", "offline_from_s": 5, "offline_to_s": 8}])
    assert not node_offline(sc, "n1", 4.9)
    assert node_offline(sc, "n1", 5) and node_offline(sc, "n1", 7.9)
    assert not node_offline(sc, "n1", 8)
    forever = _scenario(node_failures=[{"node_id": "n1"}])
    assert node_offline(forever, "n1", 9999)
