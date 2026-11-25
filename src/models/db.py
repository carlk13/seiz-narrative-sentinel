"""Asynchronous SQLAlchemy ORM models, engine setup, and SimulationRun repository."""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import structlog
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    delete,
    select,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.pool import NullPool

from src.models.schemas import TaskStatus
from src.simulation.engine import configure_structlog

configure_structlog()
logger = structlog.get_logger(__name__)

DEFAULT_POSTGRES_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/seiz_tracker"
DEFAULT_SQLITE_FALLBACK_URL = "sqlite+aiosqlite:///./seiz_tracker.db"

PERCENTILE_SERIES_KEYS: tuple[str, ...] = (
    "S_p5",
    "S_p50",
    "S_p95",
    "E_p5",
    "E_p50",
    "E_p95",
    "I_p5",
    "I_p50",
    "I_p95",
    "Z_p5",
    "Z_p50",
    "Z_p95",
)


def _normalize_postgres_async_url(url: str) -> str:
    """Ensure postgresql:// URLs use the asyncpg driver for async SQLAlchemy engines."""
    if url.startswith("postgresql+psycopg2://"):
        return url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


def resolve_database_url(database_url: str | None = None) -> str:
    """Resolve async PostgreSQL URL or fall back to local async SQLite when unreachable."""
    candidate = (
        database_url
        if database_url is not None
        else (
            os.getenv("DATABASE_URL")
            or os.getenv("SEIZ_DATABASE_URL")
            or DEFAULT_POSTGRES_URL
        )
    )
    normalized = _normalize_postgres_async_url(candidate.strip())
    if normalized.startswith("postgresql+asyncpg://"):
        parsed = urlparse(normalized)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5432
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return normalized
        except (OSError, ValueError) as exc:
            logger.warning(
                "postgres_unreachable_fallback_sqlite",
                database_url=normalized,
                error=str(exc),
            )
            return DEFAULT_SQLITE_FALLBACK_URL
    return normalized


class Base(DeclarativeBase):
    """Declarative base for SEIZ database ORM models."""


class SimulationRun(Base):
    """Persistent task metadata, initial states, calibrated parameters, and status."""

    __tablename__ = "simulation_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_type: Mapped[str] = mapped_column(String(32), nullable=False, default="simulation")
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
        default=TaskStatus.PENDING.value,
    )
    initial_state: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    parameters: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    sse: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    trajectory: Mapped[TrajectoryData | None] = relationship(
        "TrajectoryData",
        back_populates="simulation_run",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def to_result_dict(self) -> dict[str, Any] | None:
        """Reconstruct JSON-serializable result payload matching TaskStatusResponse."""
        if self.status != TaskStatus.COMPLETED.value or self.trajectory is None:
            return None
        series_dict = self.trajectory.to_series_dict()
        if self.task_type == "calibration":
            return {
                "parameters": self.parameters,
                "sse": self.sse,
                "fitted_series": series_dict,
            }
        return series_dict

    def to_task_record(self) -> dict[str, Any]:
        """Convert ORM row into dictionary representation compatible with API routes."""
        return {
            "task_id": self.id,
            "status": TaskStatus(self.status),
            "task_type": self.task_type,
            "result": self.to_result_dict(),
            "error": self.error,
            "error_type": self.error_type,
            "initial_state": self.initial_state,
            "parameters": self.parameters,
            "sse": self.sse,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class TrajectoryData(Base):
    """Persistent compartment time-series arrays and stochastic percentile bands."""

    __tablename__ = "trajectory_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    simulation_run_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("simulation_runs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    t: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    S: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    E: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    I: Mapped[list[float]] = mapped_column(JSON, nullable=False)  # noqa: E741
    Z: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    N: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    percentiles: Mapped[dict[str, list[float]] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    simulation_run: Mapped[SimulationRun] = relationship(
        "SimulationRun",
        back_populates="trajectory",
    )

    def to_series_dict(self) -> dict[str, list[float]]:
        """Reconstruct compartment series and any stored percentile bands."""
        payload: dict[str, list[float]] = {
            "t": list(self.t),
            "S": list(self.S),
            "E": list(self.E),
            "I": list(self.I),
            "Z": list(self.Z),
            "N": list(self.N),
        }
        if self.percentiles:
            for key, values in self.percentiles.items():
                payload[key] = list(values)
        return payload


DATABASE_URL = resolve_database_url()
engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    poolclass=NullPool,
    future=True,
)
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

_INITIALIZED_ENGINE_IDS: set[int] = set()


async def init_db(target_engine: AsyncEngine | None = None) -> None:
    """Create database tables idempotently for the configured async engine."""
    active_engine = target_engine if target_engine is not None else engine
    async with active_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    _INITIALIZED_ENGINE_IDS.add(id(active_engine))


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an asynchronous SQLAlchemy session."""
    async with AsyncSessionLocal() as session:
        yield session


class SimulationRepository:
    """Async repository encapsulating SimulationRun and TrajectoryData persistence."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        target_engine: AsyncEngine | None = None,
    ) -> None:
        """Initialize repository with an async session factory and target engine."""
        self.engine: AsyncEngine = target_engine if target_engine is not None else engine
        self.session_factory: async_sessionmaker[AsyncSession] = (
            session_factory
            if session_factory is not None
            else (
                AsyncSessionLocal
                if target_engine is None
                else async_sessionmaker(
                    bind=target_engine,
                    class_=AsyncSession,
                    expire_on_commit=False,
                )
            )
        )

    async def _ensure_initialized(self) -> None:
        if id(self.engine) not in _INITIALIZED_ENGINE_IDS:
            await init_db(self.engine)

    async def create_run(
        self,
        task_id: str,
        task_type: str,
        status: TaskStatus = TaskStatus.PENDING,
        initial_state: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> SimulationRun:
        """Insert a new SimulationRun record in PostgreSQL/SQLite."""
        await self._ensure_initialized()
        now = datetime.now(UTC)
        async with self.session_factory() as session:
            run = SimulationRun(
                id=task_id,
                task_type=task_type,
                status=status.value,
                initial_state=initial_state,
                parameters=parameters,
                created_at=now,
                updated_at=now,
            )
            session.add(run)
            await session.commit()
            await session.refresh(run)
            return run

    async def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        series: dict[str, list[float]] | None = None,
        parameters: dict[str, Any] | None = None,
        sse: float | None = None,
        error: str | None = None,
        error_type: str | None = None,
    ) -> SimulationRun | None:
        """Update SimulationRun status and persist TrajectoryData on completion."""
        await self._ensure_initialized()
        async with self.session_factory() as session:
            stmt = select(SimulationRun).where(SimulationRun.id == task_id)
            run = (await session.execute(stmt)).scalar_one_or_none()
            if run is None:
                return None

            run.status = status.value
            run.error = error
            run.error_type = error_type
            run.updated_at = datetime.now(UTC)
            if parameters is not None:
                run.parameters = parameters
            if sse is not None:
                run.sse = sse

            if series is not None:
                percentile_payload = {
                    key: series[key] for key in PERCENTILE_SERIES_KEYS if key in series
                }
                if run.trajectory is None:
                    run.trajectory = TrajectoryData(
                        simulation_run_id=run.id,
                        t=series["t"],
                        S=series["S"],
                        E=series["E"],
                        I=series["I"],
                        Z=series["Z"],
                        N=series["N"],
                        percentiles=percentile_payload or None,
                        created_at=datetime.now(UTC),
                    )
                else:
                    run.trajectory.t = series["t"]
                    run.trajectory.S = series["S"]
                    run.trajectory.E = series["E"]
                    run.trajectory.I = series["I"]
                    run.trajectory.Z = series["Z"]
                    run.trajectory.N = series["N"]
                    run.trajectory.percentiles = percentile_payload or None

            await session.commit()
            await session.refresh(run)
            return run

    async def get_run(self, task_id: str) -> SimulationRun | None:
        """Retrieve a SimulationRun and its TrajectoryData by task ID."""
        await self._ensure_initialized()
        async with self.session_factory() as session:
            stmt = select(SimulationRun).where(SimulationRun.id == task_id)
            return (await session.execute(stmt)).scalar_one_or_none()

    async def get_run_dict(self, task_id: str) -> dict[str, Any] | None:
        """Retrieve a task record dictionary by task ID, or None if missing."""
        run = await self.get_run(task_id)
        return run.to_task_record() if run is not None else None

    async def list_runs(
        self,
        limit: int = 25,
        task_type: str | None = None,
    ) -> list[SimulationRun]:
        """List recent SimulationRun records ordered newest-first."""
        await self._ensure_initialized()
        async with self.session_factory() as session:
            stmt = select(SimulationRun).order_by(SimulationRun.created_at.desc())
            if task_type is not None:
                stmt = stmt.where(SimulationRun.task_type == task_type)
            stmt = stmt.limit(max(1, min(limit, 200)))
            return list((await session.execute(stmt)).scalars().all())

    async def clear_all(self) -> None:
        """Delete all SimulationRun and TrajectoryData rows (used for test isolation)."""
        await self._ensure_initialized()
        async with self.session_factory() as session:
            await session.execute(delete(TrajectoryData))
            await session.execute(delete(SimulationRun))
            await session.commit()


class TrackedNarrative(Base):
    """Persistent global narrative monitor with latest country stats and alert state."""

    __tablename__ = "tracked_narratives"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    query: Mapped[str] = mapped_column(String(256), nullable=False)
    sources: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True, default=True)
    alert_velocity_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=25.0)
    alert_rsi_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_infected_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_skeptic_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_max_velocity: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_inflection_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_rsi: Mapped[float | None] = mapped_column(Float, nullable=True)
    alert_triggered: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    alert_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    country_counts: Mapped[dict[str, int] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class NarrativeRepository:
    """Async repository managing TrackedNarrative watchlist rows and alert snapshots."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        target_engine: AsyncEngine | None = None,
    ) -> None:
        self.engine: AsyncEngine = target_engine if target_engine is not None else engine
        self.session_factory: async_sessionmaker[AsyncSession] = (
            session_factory
            if session_factory is not None
            else (
                AsyncSessionLocal
                if target_engine is None
                else async_sessionmaker(
                    bind=target_engine,
                    class_=AsyncSession,
                    expire_on_commit=False,
                )
            )
        )

    async def _ensure_initialized(self) -> None:
        if id(self.engine) not in _INITIALIZED_ENGINE_IDS:
            await init_db(self.engine)

    async def create_narrative(
        self,
        narrative_id: str,
        name: str,
        query: str,
        sources: list[str],
        alert_velocity_threshold: float = 25.0,
        alert_rsi_threshold: float = 1.0,
    ) -> TrackedNarrative:
        """Insert a new tracked global narrative monitor."""
        await self._ensure_initialized()
        now = datetime.now(UTC)
        async with self.session_factory() as session:
            row = TrackedNarrative(
                id=narrative_id,
                name=name,
                query=query,
                sources=list(sources),
                active=True,
                alert_velocity_threshold=alert_velocity_threshold,
                alert_rsi_threshold=alert_rsi_threshold,
                last_infected_count=0,
                last_skeptic_count=0,
                alert_triggered=False,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return row

    async def list_narratives(self, active_only: bool = False) -> list[TrackedNarrative]:
        """List tracked narratives newest-first."""
        await self._ensure_initialized()
        async with self.session_factory() as session:
            stmt = select(TrackedNarrative).order_by(TrackedNarrative.created_at.desc())
            if active_only:
                stmt = stmt.where(TrackedNarrative.active.is_(True))
            return list((await session.execute(stmt)).scalars().all())

    async def get_narrative(self, narrative_id: str) -> TrackedNarrative | None:
        """Fetch a single TrackedNarrative by ID."""
        await self._ensure_initialized()
        async with self.session_factory() as session:
            stmt = select(TrackedNarrative).where(TrackedNarrative.id == narrative_id)
            return (await session.execute(stmt)).scalar_one_or_none()

    async def update_snapshot(
        self,
        narrative_id: str,
        infected_count: int,
        skeptic_count: int,
        country_counts: dict[str, int],
        max_velocity: float | None = None,
        inflection_time: float | None = None,
        rsi: float | None = None,
        alert_triggered: bool = False,
        alert_message: str | None = None,
    ) -> TrackedNarrative | None:
        """Persist latest global scan metrics and alert evaluation on a narrative."""
        await self._ensure_initialized()
        async with self.session_factory() as session:
            stmt = select(TrackedNarrative).where(TrackedNarrative.id == narrative_id)
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                return None
            now = datetime.now(UTC)
            row.last_checked_at = now
            row.updated_at = now
            row.last_infected_count = int(infected_count)
            row.last_skeptic_count = int(skeptic_count)
            row.country_counts = dict(country_counts)
            row.last_max_velocity = max_velocity
            row.last_inflection_time = inflection_time
            row.last_rsi = rsi
            row.alert_triggered = bool(alert_triggered)
            row.alert_message = alert_message
            await session.commit()
            await session.refresh(row)
            return row

    async def delete_narrative(self, narrative_id: str) -> bool:
        """Delete a tracked narrative by ID, returning True if a row was removed."""
        await self._ensure_initialized()
        async with self.session_factory() as session:
            stmt = select(TrackedNarrative).where(TrackedNarrative.id == narrative_id)
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                return False
            await session.delete(row)
            await session.commit()
            return True

    async def clear_all(self) -> None:
        """Delete all TrackedNarrative rows (used for test isolation)."""
        await self._ensure_initialized()
        async with self.session_factory() as session:
            await session.execute(delete(TrackedNarrative))
            await session.commit()
