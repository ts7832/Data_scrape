import json
from datetime import UTC, datetime, timedelta

import httpx

from kuulo_sim.cli import main
from kuulo_sim.engine import SimulationRun
from kuulo_sim.runner import run_realtime
from kuulo_sim.scenario import load_scenario, resolve_scenario

T0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def test_runner_registers_then_sends_everything_in_time(tmp_path):
    run = SimulationRun(load_scenario(resolve_scenario("false_alarm")), T0)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={"status": "accepted", "late": False})

    client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    fake_now = [T0]
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        fake_now[0] = fake_now[0] + timedelta(seconds=seconds)

    truth = tmp_path / "truth.json"
    stats = run_realtime(run, client, sleep=sleep, now=lambda: fake_now[0], truth_path=truth)
    registrations = seen.count("/v1/nodes/register")
    assert registrations == 3 and seen[:3] == ["/v1/nodes/register"] * 3
    assert stats.sent == len(run.messages()) and stats.errors == 0
    assert sum(slept) > 60
    assert json.loads(truth.read_text()) == []  # no drones in this scenario


def test_runner_counts_errors():
    run = SimulationRun(load_scenario(resolve_scenario("single_node")), T0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/nodes/register":
            return httpx.Response(200, json={})
        return httpx.Response(401, json={"detail": "bad signature"})

    client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    stats = run_realtime(run, client, sleep=lambda s: None, now=lambda: T0)
    assert stats.errors == len(run.messages()) and stats.sent == 0


def test_cli_list(capsys):
    assert main(["list"]) == 0
    assert "helsinki_pass" in capsys.readouterr().out


def test_cli_invalid_scenario_names_the_field(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nduration_s: 10\nnodes:\n  - {id: n1, lat: 95, lon: 24}\n")
    assert main(["run", str(bad)]) == 2
    err = capsys.readouterr().err
    assert "Invalid scenario" in err and "nodes.0.lat" in err
