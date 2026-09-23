"""SQLite storage (WAL mode) via SQLAlchemy 2.0."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Boolean, DateTime, Integer, String, Text, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def as_utc(value: datetime | None) -> datetime | None:
    """SQLite returns naive datetimes; everything we store is UTC."""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


class Base(DeclarativeBase):
    pass


class NodeRow(Base):
    __tablename__ = "nodes"
    node_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    public_key: Mapped[str] = mapped_column(String(64))
    lat: Mapped[float]
    lon: Mapped[float]
    accuracy_m: Mapped[float]
    time_quality: Mapped[str] = mapped_column(String(16))
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    software_version: Mapped[str | None] = mapped_column(String(32))
    mic_ok: Mapped[bool | None] = mapped_column(Boolean)
    queue_depth: Mapped[int | None] = mapped_column(Integer)


class ObservationRow(Base):
    __tablename__ = "observations"
    observation_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    node_id: Mapped[str] = mapped_column(String(64), index=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    late: Mapped[bool] = mapped_column(Boolean)
    label: Mapped[str] = mapped_column(String(32))
    detection_id: Mapped[str] = mapped_column(String(36), index=True)
    raw_json: Mapped[str] = mapped_column(Text)


class TrackRow(Base):
    __tablename__ = "tracks"
    track_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    ever_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    raw_json: Mapped[str] = mapped_column(Text)


class TrackObservationRow(Base):
    __tablename__ = "track_observations"
    track_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    observation_id: Mapped[str] = mapped_column(String(36), primary_key=True, index=True)


def make_session_factory(db_path: Path) -> sessionmaker:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
