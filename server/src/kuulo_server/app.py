"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import select

from kuulo_protocol.api import IngestResult, LiveEvent, NodeStatus, NodeView, TrackDetail
from kuulo_protocol.models import Heartbeat, NodeRegistration, Observation, Track, TrackStatus
from kuulo_protocol.traces import FeatureTraceHeader, TraceRequest, TraceUnavailable

from .config import Settings
from .db import NodeRow, as_utc, make_session_factory
from .fusion.context import DbFusionContext
from .fusion.loader import load_engine
from .ingest import IngestError, ingest_heartbeat, ingest_observation, register_node
from .live import LiveHub
from .status import node_status, node_view
from .traces import (
    create_requests_for_track,
    mark_unavailable,
    open_requests,
    segment_count,
    store_trace,
)
from .tracks import close_open_tracks, open_tracks, persist_track, track_detail, uncorroborated_rate

log = logging.getLogger("kuulo.server")
OBSERVATION = TypeAdapter(Observation)


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
                if update.track.status is TrackStatus.CONFIRMED:
                    create_requests_for_track(session, update.track, settings.clock())
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
                status = node_status(as_utc(row.last_heartbeat_at), now)
                if last_status.get(row.node_id) not in (None, status):
                    # The noisy-node rate is a 24 h query: only pay for it when publishing.
                    hub.publish(LiveEvent(type="node_status", data=view_of(session, row, now)))
                last_status[row.node_id] = status
        run_fusion(lambda ctx: engine.on_tick(now, ctx))

    def ingest_one(obs: Observation) -> IngestResult:
        with sessions() as session:
            result = ingest_observation(session, obs, settings.clock(), settings)
        if result.status == "accepted" and not result.late:
            hub.publish(LiveEvent(type="observation", data=obs))
            run_fusion(lambda ctx: engine.on_observation(obs, ctx))
        return result

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
        # Drop "input": a rejected field can be a non-finite float (inf/nan), and
        # Starlette's JSONResponse refuses to encode those, which turned a clean
        # 400 into an unhandled 500 for exactly the malformed input this exists
        # to reject. Callers get the location and reason; not the raw value back.
        errors = [{k: v for k, v in error.items() if k != "input"} for error in exc.errors()]
        return JSONResponse(status_code=400, content={"detail": jsonable_encoder(errors)})

    @app.exception_handler(IngestError)
    async def _ingest_error(_request: Request, exc: IngestError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.reason})

    @app.post("/v1/nodes/register")
    async def post_register(reg: NodeRegistration) -> dict:
        with sessions() as session:
            register_node(session, reg, settings.clock())
        return {"status": "registered"}

    @app.post("/v1/observations")
    async def post_observation(request: Request):
        """One Observation, or a JSON array of 1..max_batch with one result per item."""
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "body is not JSON"})
        if not isinstance(body, list):
            try:
                obs = OBSERVATION.validate_python(body)
            except ValidationError as exc:
                raise RequestValidationError(exc.errors()) from exc
            return ingest_one(obs)
        if not 1 <= len(body) <= settings.max_batch:
            return JSONResponse(status_code=400, content={
                "detail": f"a batch must hold 1 to {settings.max_batch} observations"})
        results: list[dict] = []
        for item in body:
            try:
                results.append(ingest_one(OBSERVATION.validate_python(item)).model_dump())
            except ValidationError as exc:
                reason = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}"
                                   for e in exc.errors())
                results.append({"status": "rejected", "reason": reason})
            except IngestError as exc:
                results.append({"status": "rejected", "reason": exc.reason})
        return results

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
            segments = segment_count(session, detail.track)
            return detail.model_copy(update={"trace_segments": segments})

    @app.post("/v1/traces")
    async def post_trace(
        header: Annotated[str, Form()], body: Annotated[UploadFile, File()]
    ) -> dict:
        """One signed FeatureTrace segment: JSON header field plus the .npz body file."""
        data = await body.read(settings.max_trace_bytes + 1)
        if len(data) > settings.max_trace_bytes:
            raise IngestError(400, f"trace body exceeds {settings.max_trace_bytes} bytes")
        try:
            parsed = FeatureTraceHeader.model_validate_json(header)
        except ValidationError as exc:
            raise IngestError(400, f"invalid trace header: {exc.error_count()} error(s)") from exc
        with sessions() as session:
            status = store_trace(session, parsed, data, settings.traces_dir, settings.clock())
        return {"status": status}

    @app.get("/v1/traces/requests")
    async def get_trace_requests(node_id: str) -> list[TraceRequest]:
        with sessions() as session:
            return open_requests(session, node_id)

    @app.post("/v1/traces/unavailable")
    async def post_trace_unavailable(msg: TraceUnavailable) -> dict:
        with sessions() as session:
            mark_unavailable(session, msg, settings.clock())
        return {"status": "closed"}

    @app.websocket("/v1/live")
    async def live(ws: WebSocket) -> None:
        origin = ws.headers.get("origin")
        if origin is not None and origin not in settings.allowed_origins:
            await ws.close(code=1008)
            return
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
