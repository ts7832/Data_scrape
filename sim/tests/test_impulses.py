from datetime import UTC, datetime

from kuulo_protocol.geo import distance_m
from kuulo_protocol.impulses import speed_of_sound
from kuulo_sim.engine import SimulationRun
from kuulo_sim.scenario import Scenario, load_scenario, resolve_scenario

T0 = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


def reports(run):
    return [m.payload for m in run.messages() if m.kind == "impulse"]


def scenario(name, **update):
    return load_scenario(resolve_scenario(name)).model_copy(update=update)


def test_impact_strike_reports_the_detonation_from_every_node_in_range():
    run = SimulationRun(scenario("impact_strike"), T0)
    [truth] = run.impulse_truth()
    rs = reports(run)
    assert truth.label.startswith("impact:") and len(rs) == len(run.scenario.nodes)


def test_arrival_differences_match_geometry_for_gps_nodes():
    run = SimulationRun(scenario("impact_strike", nlos_delay_s={}), T0)
    [truth] = run.impulse_truth()
    c = speed_of_sound(run.scenario.air_temperature_c)
    gps = [r for r in reports(run) if r.time_quality.value == "gps"]
    a, b = gps[0], gps[1]
    expected = (distance_m(a.sensor_location, truth.position)
                - distance_m(b.sensor_location, truth.position)) / c
    assert abs((a.onset_at - b.onset_at).total_seconds() - expected) < 0.003


def test_speed_factor_does_not_compress_arrival_differences():
    slow = reports(SimulationRun(scenario("impact_strike"), T0))
    fast = reports(SimulationRun(scenario("impact_strike"), T0, time_scale=4.0))
    diff = lambda rs: [(r.onset_at - rs[0].onset_at).total_seconds() for r in rs]  # noqa: E731
    assert [round(d, 6) for d in diff(slow)] == [round(d, 6) for d in diff(fast)]


def test_nlos_node_hears_it_late():
    open_path = {r.source.id: r.onset_at for r in reports(SimulationRun(
        scenario("impact_strike", nlos_delay_s={}), T0))}
    blocked = {r.source.id: r.onset_at for r in reports(SimulationRun(
        scenario("impact_strike"), T0))}
    late = sorted(n for n in open_path if (blocked[n] - open_path[n]).total_seconds() > 0.1)
    assert late == sorted(scenario("impact_strike").nlos_delay_s) == ["s06"]


def test_weak_impulse_is_not_heard():
    sc = scenario("firework")
    weak = sc.model_copy(update={"impulses": [sc.impulses[0].model_copy(update={"source_db": 60})]})
    assert reports(SimulationRun(weak, T0)) == []  # 60 dB at 1 m is below every node's floor


def test_impulses_do_not_change_existing_scenario_messages():
    plain = SimulationRun(scenario("helsinki_pass"), T0).messages()
    base = scenario("helsinki_pass").model_dump()
    extra = Scenario.model_validate({**base, "impulses": [
        {"lat": 60.17, "lon": 24.94, "at_s": 50, "source_db": 150, "label": "bang"}]})
    messages = SimulationRun(extra, T0).messages()
    assert any(m.kind == "impulse" for m in messages)  # the bang really is heard
    other = [m for m in messages if m.kind != "impulse"]
    assert [m.payload.model_dump_json() for m in plain] == \
           [m.payload.model_dump_json() for m in other]


def test_deterministic():
    a = [r.model_dump_json() for r in reports(SimulationRun(scenario("impact_strike"), T0))]
    b = [r.model_dump_json() for r in reports(SimulationRun(scenario("impact_strike"), T0))]
    assert a == b
