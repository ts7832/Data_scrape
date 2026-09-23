"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import select

from kuulo_protocol.api import IngestResult, LiveEvent, NodeStatus, NodeView, TrackDetail
from kuulo_protocol.models import Heartbeat, NodeRegistration, Observation, Track

from .config import Settings
from .db import NodeRow, make_session_factory
from .fusion.context import DbFusionContext
from .fusion.loader import load_engine
from .ingest import IngestError, ingest_heartbeat, ingest_observation, register_node
from .live import LiveHub
from .status import node_view
from .tracks import close_open_tracks, open_tracks, persist_track, track_detail, uncorroborated_rate

log = logging.getLogger("kuulo.server")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    sessions = make_session_factory(settings.db_path)
    hub = LiveHub()
    last_status: dict[str, NodeStatus] = {}

    engine = load_engine(settings.fusion_engine)
    with sessions() as session:
        closed = close_open_tracks(session)
    if closed:
        log.info("closed %d tracks left open by a previous run", closed)

    def apply_updates(updates) -> None:
        with sessions() as session:
            for update in updates:
                persist_track(session, update.track)
        for update in updates:
            hub.publish(LiveEvent(type="track", data=update.track))
            if settings.on_track_update:
                settings.on_track_update(update.track)

    def run_fusion(call) -> None:
        try:
            apply_updates(call(DbFusionContext(sessions, settings.clock())))
        except Exception:
            log.exception("fusion engine failed; ingest continues")

    def view_of(session, row, now) -> NodeView:
        return node_view(row, now, uncorroborated_rate(session, row.node_id, now))

    def run_tick() -> None:
        now = settings.clock()
        with sessions() as session:
            for row in session.scalars(select(NodeRow)):
                view = view_of(session, row, now)
                if last_status.get(row.node_id) not in (None, view.status):
                    hub.publish(LiveEvent(type="node_status", data=view))
                last_status[row.node_id] = view.status
        run_fusion(lambda ctx: engine.on_tick(now, ctx))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        task = None
        if settings.tick_interval_s:
            async def loop():
                while True:
                    await asyncio.sleep(settings.tick_interval_s)
                    try:
                        run_tick()
                    except Exception:
                        log.exception("tick failed")

            task = asyncio.create_task(loop())
        yield
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="Kuulo", lifespan=lifespan)
    app.state.settings = settings
    app.state.sessions = sessions
    app.state.hub = hub
    app.state.run_tick = run_tick
    app.state.engine = engine

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
            result = ingest_observation(session, obs, settings.clock(), settings)
        if result.status == "accepted" and not result.late:
            hub.publish(LiveEvent(type="observation", data=obs))
            run_fusion(lambda ctx: engine.on_observation(obs, ctx))
        return result

    @app.post("/v1/heartbeats")
    async def post_heartbeat(hb: Heartbeat) -> dict:
        now = settings.clock()
        with sessions() as session:
            row = ingest_heartbeat(session, hb, now, settings)
            view = view_of(session, row, now)
        hub.publish(LiveEvent(type="node_status", data=view))
        return {"status": "ok"}

    @app.get("/v1/nodes")
    async def get_nodes() -> list[NodeView]:
        now = settings.clock()
        with sessions() as session:
            return [view_of(session, row, now) for row in session.scalars(select(NodeRow))]

    @app.get("/v1/tracks")
    async def get_tracks() -> list[Track]:
        with sessions() as session:
            return open_tracks(session, settings.clock())

    @app.get("/v1/tracks/{track_id}")
    async def get_track(track_id: str) -> TrackDetail:
        with sessions() as session:
            detail = track_detail(session, track_id, settings.clock())
        if detail is None:
            raise HTTPException(status_code=404, detail="unknown track")
        return detail

    @app.websocket("/v1/live")
    async def live(ws: WebSocket) -> None:
        await ws.accept()
        queue = hub.subscribe()
        try:
            while True:
                await ws.send_text(await queue.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.unsubscribe(queue)

    return app
