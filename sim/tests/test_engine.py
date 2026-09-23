from datetime import UTC, datetime

from kuulo_protocol.models import Observation
from kuulo_protocol.signing import verify
from kuulo_sim.engine import SimulationRun
from kuulo_sim.scenario import bundled_scenarios, load_scenario, resolve_scenario

T0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def run_of(name: str, **kw) -> SimulationRun:
    return SimulationRun(load_scenario(resolve_scenario(name)), T0, **kw)


def observations(run):
    return [m.payload for m in run.messages() if m.kind == "observation"]


def test_all_bundled_scenarios_load_and_run():
    assert set(bundled_scenarios()) >= {
        "single_node", "helsinki_pass", "false_alarm", "two_drones", "node_failure"
    }
    for name in bundled_scenarios():
        assert run_of(name).messages()


def test_node_keys_stable_across_scenarios_sharing_a_node_id():
    # Review finding: running the README's own two commands back-to-back
    # ("make sim" then "make sim SCENARIO=false_alarm") re-registers "n01"
    # with a different key each time and gets 409 Conflict, because keys used
    # to be derived from (seed, node_id) instead of node_id alone.
    a = run_of("single_node")
    b = run_of("false_alarm")
    assert a.node_keys["n01"] == b.node_keys["n01"]


def test_deterministic_for_a_seed():
    a = [m.payload.model_dump_json() for m in run_of("helsinki_pass").messages()]
    b = [m.payload.model_dump_json() for m in run_of("helsinki_pass").messages()]
    assert a == b


def test_messages_sorted_and_signed():
    run = run_of("helsinki_pass")
    msgs = run.messages()
    assert [m.at for m in msgs] == sorted(m.at for m in msgs)
    keys = {r.node_id: r.public_key for r in run.registrations()}
    for m in msgs:
        node_id = m.payload.source.id if isinstance(m.payload, Observation) else m.payload.node_id
        assert verify(m.payload, keys[node_id])


def test_helsinki_pass_heard_by_several_nodes():
    starts = [o for o in observations(run_of("helsinki_pass")) if o.event.phase == "start"]
    assert len({o.source.id for o in starts}) >= 4


def test_false_alarm_only_at_n01():
    obs = observations(run_of("false_alarm"))
    assert obs and {o.source.id for o in obs} == {"n01"}


def test_failed_node_is_silent():
    run = run_of("node_failure")
    senders = {
        m.payload.source.id if m.kind == "observation" else m.payload.node_id
        for m in run.messages()
    }
    assert "n02" not in senders


def test_observations_arrive_after_emission_and_end_phase_exists():
    obs = observations(run_of("single_node"))
    phases = [o.event.phase for o in obs]
    assert phases[0] == "start" and "end" in phases


def test_time_scale_compresses_timestamps():
    fast = run_of("helsinki_pass", time_scale=10)
    assert (fast.end_at - T0).total_seconds() == 20


def test_truth_follows_drone():
    truth = run_of("helsinki_pass").truth()
    assert truth[0].drone_id == "d1"
    assert truth[0].at >= T0
