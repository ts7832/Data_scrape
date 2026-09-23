import pytest
from fastapi.testclient import TestClient

from kuulo_server.app import create_app
from kuulo_server.config import Settings
from kuulo_server.testing import T0, FakeClock, NodeKeys, register


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
