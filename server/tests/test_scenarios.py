from kuulo_protocol.geo import distance_m
from kuulo_protocol.models import TrackStatus

LOCATION_ERROR_BOUND_M = 750.0


def statuses(result):
    return {u.status for u in result.updates}


def mean_location_error(result) -> float:
    errors = []
    for track in result.updates:
        if track.status is not TrackStatus.CONFIRMED:
            continue
        near = [p for p in result.truth if abs((p.at - track.last_seen).total_seconds()) <= 0.5]
        if near:
            errors.append(min(distance_m(track.position, p.position) for p in near))
    assert errors, "no confirmed track could be matched to ground truth"
    return sum(errors) / len(errors)


def test_single_node_only_tentative(run_scenario):
    result = run_scenario("single_node")
    assert result.updates
    assert TrackStatus.CONFIRMED not in statuses(result)
    assert TrackStatus.TENTATIVE in statuses(result)


def test_helsinki_pass_confirmed_and_located(run_scenario):
    result = run_scenario("helsinki_pass")
    assert TrackStatus.CONFIRMED in statuses(result)
    error = mean_location_error(result)
    print(f"helsinki_pass mean location error: {error:.0f} m")
    assert error < LOCATION_ERROR_BOUND_M


def test_false_alarm_never_confirmed(run_scenario):
    result = run_scenario("false_alarm")
    assert TrackStatus.CONFIRMED not in statuses(result)
    assert TrackStatus.DOWNGRADED in statuses(result)


def test_two_drones_two_tracks(run_scenario):
    result = run_scenario("two_drones")
    confirmed_ids = {u.track_id for u in result.updates if u.status is TrackStatus.CONFIRMED}
    assert len(confirmed_ids) >= 2
    node_of = {
        m.payload.observation_id: m.payload.source.id
        for m in result.run.messages() if m.kind == "observation"
    }
    for track in result.updates:
        clusters = {node_of[i][0] for i in track.observation_ids}
        assert len(clusters) == 1, f"track {track.track_id} mixes clusters {clusters}"


def test_offline_neighbour_silence_not_counted(run_scenario):
    result = run_scenario("node_failure")
    assert TrackStatus.DOWNGRADED not in statuses(result)
    assert TrackStatus.TENTATIVE in statuses(result)


def test_long_event_confirms_and_uploads_a_multi_segment_trace(run_scenario):
    result = run_scenario("long_event")
    assert TrackStatus.CONFIRMED in statuses(result)
    per_detection: dict[str, list[int]] = {}
    for _, detection_id, index, _ in result.traces:
        per_detection.setdefault(detection_id, []).append(index)
    assert max(len(v) for v in per_detection.values()) >= 10
    assert all(reason == "fulfilled" for _, _, reason in result.requests)


def test_helsinki_pass_uploads_traces_only_for_requested_detections(run_scenario):
    result = run_scenario("helsinki_pass")
    requested = {(n, d) for n, d, _ in result.requests}
    uploaded = {(n, d) for n, d, _, _ in result.traces}
    assert requested and uploaded == requested
    assert all(reason == "fulfilled" for _, _, reason in result.requests)


def test_false_alarm_uploads_no_traces(run_scenario):
    result = run_scenario("false_alarm")
    assert result.traces == [] and result.requests == []
