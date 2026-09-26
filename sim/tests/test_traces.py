from datetime import UTC, datetime

import numpy as np

from kuulo_protocol.features import BAND_EDGES_HZ
from kuulo_protocol.signing import verify
from kuulo_protocol.traces import SEGMENT_FRAMES, FeatureTraceHeader, decode_body
from kuulo_sim.engine import SimulationRun
from kuulo_sim.scenario import DroneSpec, Scenario, drone_position, load_scenario, resolve_scenario
from kuulo_sim.traces import synth_frames

T0 = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


def run_of(name: str) -> SimulationRun:
    return SimulationRun(load_scenario(resolve_scenario(name)), T0)


def band_of(freq: float) -> int:
    return int(np.searchsorted(BAND_EDGES_HZ, freq, side="right") - 1)


def test_doppler_raises_pitch_approaching_and_lowers_it_receding():
    approach = synth_frames(level_db=70, noise_floor_db=35, f0_hz=180, doppler=343 / (343 - 20),
                            frames=10, rng=np.random.default_rng(0))
    recede = synth_frames(level_db=70, noise_floor_db=35, f0_hz=180, doppler=343 / (343 + 20),
                          frames=10, rng=np.random.default_rng(0))
    assert np.all(approach.peak_freq_hz > 180) and np.all(recede.peak_freq_hz < 180)
    assert np.all(approach.peak_freq_hz > recede.peak_freq_hz)


def test_louder_source_raises_band_energy_over_the_floor():
    rng = np.random.default_rng(0)
    quiet = synth_frames(level_db=40, noise_floor_db=35, f0_hz=180, doppler=1.0, frames=5, rng=rng)
    loud = synth_frames(level_db=80, noise_floor_db=35, f0_hz=180, doppler=1.0, frames=5, rng=rng)
    b = band_of(180)
    assert loud.band_db[:, b].mean() > quiet.band_db[:, b].mean() + 30
    assert loud.rms_db.mean() > quiet.rms_db.mean()


def test_every_detection_has_signed_valid_segments():
    run = run_of("helsinki_pass")
    detections = run.detections()
    assert detections
    for node_id, detection_id in detections:
        segments = run.trace_segments(node_id, detection_id)
        assert segments and segments[-1].header.final
        _, public = run.node_keys[node_id]
        for seg in segments:
            assert verify(seg.header, public)
            decode_body(seg.body, seg.header.frame_count)


def test_long_event_is_one_long_detection_per_node_with_many_segments():
    run = run_of("long_event")
    per_node: dict[str, list] = {}
    for node_id, detection_id in run.detections():
        per_node.setdefault(node_id, []).append(detection_id)
    longest = max(len(run.trace_segments(n, d)) for n, ds in per_node.items() for d in ds)
    assert longest >= 10
    seg = run.trace_segments(*run.detections()[0])[0]
    assert isinstance(seg.header, FeatureTraceHeader)
    assert seg.header.frame_count <= SEGMENT_FRAMES


def test_segment_end_times_are_increasing():
    run = run_of("long_event")
    node_id, detection_id = run.detections()[0]
    ends = [s.available_at for s in run.trace_segments(node_id, detection_id)]
    assert ends == sorted(ends) and len(set(ends)) == len(ends)


def test_traces_are_deterministic():
    a = run_of("long_event")
    b = run_of("long_event")
    key = a.detections()[0]
    assert [s.body for s in a.trace_segments(*key)] == [s.body for s in b.trace_segments(*key)]


def test_repeat_loops_the_route():
    # ~99.5 m north and back: one loop is ~199 m, so ~19.9 s at 10 m/s, three loops ~59.7 s.
    drone = DroneSpec(id="d", speed_mps=10, waypoints=[(60.0, 25.0), (60.0009, 25.0)], repeat=3)
    assert drone_position(drone, 5) is not None
    assert drone_position(drone, 55) is not None  # third loop
    assert drone_position(drone, 61) is None
    start, looped = drone_position(drone, 0), drone_position(drone, 20)
    assert abs(start.lat - looped.lat) < 2e-5  # back at the start, within ~2 m


def test_scenario_rejects_non_positive_repeat():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Scenario.model_validate({
            "name": "x", "duration_s": 10, "nodes": [{"id": "a", "lat": 60, "lon": 25}],
            "drones": [{"id": "d", "speed_mps": 1, "waypoints": [[60, 25], [60.1, 25]],
                        "repeat": 0}],
        })
