"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from kuulo_protocol.api import IngestResult
from kuulo_protocol.models import NodeRegistration, Observation

from .config import Settings
from .db import make_session_factory
from .ingest import IngestError, ingest_observation, register_node


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    sessions = make_session_factory(settings.db_path)

    app = FastAPI(title="Kuulo")
    app.state.settings = settings
    app.state.sessions = sessions

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=400, content={"detail": jsonable_encoder(exc.errors())})

    @app.exception_handler(IngestError)
    async def _ingest_error(_request: Request, exc: IngestError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.reason})

    @app.post("/v1/nodes/register")
    async def post_register(reg: NodeRegistration) -> dict:
        with sessions() as session:
            register_node(session, reg, settings.clock())
        return {"status": "registered"}

    @app.post("/v1/observations")
    async def post_observation(obs: Observation) -> IngestResult:
        with sessions() as session:
            return ingest_observation(session, obs, settings.clock(), settings)

    return app
