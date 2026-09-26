from dataclasses import dataclass
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from kuulo_protocol.impulses import ImpulseEvent
from kuulo_protocol.models import Track
from kuulo_server.app import create_app
from kuulo_server.config import Settings
from kuulo_server.db import TraceRequestRow, TraceRow
from kuulo_server.testing import T0, FakeClock, NodeKeys, register
from kuulo_sim.engine import ImpulseTruth, SimulationRun, TruthPoint
from kuulo_sim.runner import PATHS, TraceResponder
from kuulo_sim.scenario import load_scenario, resolve_scenario


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(T0)


@pytest.fixture
def app(tmp_path, clock):
    return create_app(Settings(db_path=tmp_path / "kuulo.db", clock=clock, tick_interval_s=None))


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def node(client) -> NodeKeys:
    return register(client, "n1")


@dataclass
class ScenarioResult:
    run: SimulationRun
    updates: list[Track]
    truth: list[TruthPoint]
    traces: list  # TraceRow copies: (node_id, detection_id, segment_index, final)
    requests: list  # TraceRequestRow copies: (node_id, detection_id, close_reason)
    impulse_events: list[ImpulseEvent]
    impulse_truth: list[ImpulseTruth]


@pytest.fixture
def run_scenario(tmp_path):
    """Play a bundled scenario through the real server in-process with a fake clock."""

    def _run(name: str) -> ScenarioResult:
        run = SimulationRun(load_scenario(resolve_scenario(name)), T0)
        clock = FakeClock(T0)
        updates: list[Track] = []
        app = create_app(Settings(db_path=tmp_path / f"{name}.db", clock=clock,
                                  tick_interval_s=None, on_track_update=updates.append))
        with TestClient(app) as c:
            responder = TraceResponder(run, c)
            for reg in run.registrations():
                r = c.post("/v1/nodes/register", json=reg.model_dump(mode="json"))
                assert r.status_code == 200
            next_tick = T0
            for msg in run.messages():
                while next_tick <= msg.at:
                    clock.set(next_tick)
                    app.state.run_tick()
                    if (next_tick - T0).total_seconds() % 10 == 0:
                        responder.service(next_tick)
                    next_tick += timedelta(seconds=1)
                clock.set(msg.at)
                r = c.post(PATHS[msg.kind], content=msg.payload.model_dump_json(),
                           headers={"content-type": "application/json"})
                assert r.status_code == 200, r.text
            while next_tick <= run.end_at + timedelta(seconds=60):
                clock.set(next_tick)
                app.state.run_tick()
                if (next_tick - T0).total_seconds() % 10 == 0:
                    responder.service(next_tick)
                next_tick += timedelta(seconds=1)
            with app.state.sessions() as session:
                traces = [(t.node_id, t.detection_id, t.segment_index, t.final)
                          for t in session.scalars(select(TraceRow))]
                requests = [(q.node_id, q.detection_id, q.close_reason)
                            for q in session.scalars(select(TraceRequestRow))]
            impulse_events = [ImpulseEvent.model_validate(e)
                              for e in c.get("/v1/impulse-events").json()]
        return ScenarioResult(run, updates, run.truth(), traces, requests, impulse_events,
                              run.impulse_truth())

    return _run
