"""Celery application and distributed task definitions for SEIZ simulation and calibration."""

from __future__ import annotations

import asyncio
import os
import socket
import threading
import time
import uuid
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Protocol, cast
from urllib.parse import urlparse

import numpy as np
import structlog
from celery import Celery

from src.models.db import PERCENTILE_SERIES_KEYS, SimulationRepository
from src.models.schemas import CalibrationRequest, SimulationRequest, TaskStatus
from src.simulation.calibration import ParameterEstimator
from src.simulation.engine import SEIZSimulator, SimulationResult, configure_structlog

configure_structlog()
logger = structlog.get_logger(__name__)

DEFAULT_REDIS_URL = "redis://localhost:6379/0"

TASK_STORE: dict[str, dict[str, Any]] = {}
_STORE_LOCK = threading.Lock()
SIMULATION_REPO = SimulationRepository()


class AsyncResultProtocol(Protocol):
    """Typed protocol matching Celery AsyncResult / EagerResult attributes."""

    id: str
    state: str
    result: Any

    def successful(self) -> bool: ...
    def failed(self) -> bool: ...
    def get(self, timeout: float | None = None, propagate: bool = True) -> Any: ...


class CeleryTaskProtocol[**P, R](Protocol):
    """Typed protocol exposing Celery task execution and dispatch methods."""

    name: str

    def delay(self, *args: P.args, **kwargs: P.kwargs) -> AsyncResultProtocol: ...

    def apply_async(
        self,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
        task_id: str | None = None,
        **options: Any,
    ) -> AsyncResultProtocol: ...

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


def get_broker_url() -> str:
    """Resolve Celery Redis broker URL from environment variables."""
    return (
        os.getenv("CELERY_BROKER_URL")
        or os.getenv("REDIS_URL")
        or DEFAULT_REDIS_URL
    )


def get_result_backend() -> str:
    """Resolve Celery Redis result backend URL from environment variables."""
    return (
        os.getenv("CELERY_RESULT_BACKEND")
        or os.getenv("REDIS_URL")
        or DEFAULT_REDIS_URL
    )


def is_redis_broker_reachable(broker_url: str | None = None) -> bool:
    """Return True when the configured Redis broker host/port accepts TCP connections."""
    candidate = broker_url if broker_url is not None else get_broker_url()
    if not candidate.startswith(("redis://", "rediss://")):
        return False
    parsed = urlparse(candidate)
    host = parsed.hostname or "localhost"
    port = parsed.port or 6379
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except (OSError, ValueError):
        return False


def _create_celery_app() -> Celery:
    """Initialize Celery with Redis broker/backend or eager local fallback."""
    broker_url = get_broker_url()
    result_backend = get_result_backend()
    app = Celery(
        "seiz_workers",
        broker=broker_url,
        backend=result_backend,
    )
    eager_env = os.getenv("CELERY_TASK_ALWAYS_EAGER", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    use_eager = eager_env or not is_redis_broker_reachable(broker_url)

    config_updates: dict[str, Any] = {
        "task_serializer": "json",
        "accept_content": ["json"],
        "result_serializer": "json",
        "timezone": "UTC",
        "enable_utc": True,
        "task_track_started": True,
        "broker_connection_retry_on_startup": True,
    }
    if use_eager:
        config_updates.update(
            {
                "broker_url": "memory://",
                "result_backend": "cache+memory://",
                "task_always_eager": True,
                "task_eager_propagates": False,
                "task_store_eager_result": True,
            }
        )
        logger.warning(
            "celery_redis_unreachable_using_eager_mode",
            configured_broker=broker_url,
        )

    app.conf.update(**config_updates)
    return app


celery_app: Celery = _create_celery_app()
celery: Celery = celery_app


def _run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    """Execute an async coroutine safely regardless of active thread event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _create_task_memory(
    task_id: str,
    task_type: Literal["simulation", "calibration"],
    initial_status: TaskStatus = TaskStatus.PENDING,
) -> None:
    """Create a mirror task record in TASK_STORE for in-flight spy diagnostics."""
    with _STORE_LOCK:
        TASK_STORE[task_id] = {
            "task_id": task_id,
            "status": initial_status,
            "task_type": task_type,
            "result": None,
            "error": None,
            "error_type": None,
        }


def _update_task_memory(task_id: str, **fields: Any) -> None:
    """Update fields on an existing in-memory mirror record in TASK_STORE."""
    with _STORE_LOCK:
        if task_id in TASK_STORE:
            TASK_STORE[task_id].update(fields)


async def reset_database_state() -> None:
    """Clear persisted SimulationRun/TrajectoryData rows and in-memory TASK_STORE."""
    await SIMULATION_REPO.clear_all()
    with _STORE_LOCK:
        TASK_STORE.clear()


def series_to_dict(sim_result: SimulationResult) -> dict[str, list[float]]:
    """Convert a SimulationResult into JSON-serializable Python float lists."""
    payload: dict[str, list[float]] = {
        "t": sim_result.t.tolist(),
        "S": sim_result.S.tolist(),
        "E": sim_result.E.tolist(),
        "I": sim_result.I.tolist(),
        "Z": sim_result.Z.tolist(),
        "N": sim_result.N.tolist(),
    }
    for key in PERCENTILE_SERIES_KEYS:
        if key in sim_result:
            payload[key] = sim_result[key].tolist()
    return payload


def execute_calibration(request: CalibrationRequest) -> dict[str, Any]:
    """Run SEIZ parameter estimation and return calibrated parameters, SSE, and trajectory."""
    estimator = ParameterEstimator()
    best_params = estimator.fit_to_data(
        t_observed=request.t_observed,
        I_observed=request.I_observed,
        Z_observed=request.Z_observed,
        initial_state=request.initial_state,
        initial_guess=request.initial_guess,
        maxiter=request.maxiter,
    )
    fitted_sim = estimator.simulator.solve(
        state=request.initial_state,
        params=best_params,
        t_span=(float(request.t_observed[0]), float(request.t_observed[-1])),
        t_eval=request.t_observed,
    )
    i_obs = np.asarray(request.I_observed, dtype=np.float64)
    z_obs = np.asarray(request.Z_observed, dtype=np.float64)
    sse = float(np.sum((fitted_sim.I - i_obs) ** 2) + np.sum((fitted_sim.Z - z_obs) ** 2))
    return {
        "parameters": best_params.model_dump(),
        "sse": sse,
        "fitted_series": series_to_dict(fitted_sim),
    }


async def _ensure_and_mark_processing(
    task_id: str,
    task_type: Literal["simulation", "calibration"],
    initial_state: dict[str, Any] | None,
    parameters: dict[str, Any] | None = None,
) -> None:
    """Ensure SimulationRun exists in DB and transition status to PROCESSING."""
    existing = await SIMULATION_REPO.get_run(task_id)
    if existing is None:
        await SIMULATION_REPO.create_run(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.PENDING,
            initial_state=initial_state,
            parameters=parameters,
        )
    await SIMULATION_REPO.update_status(task_id, status=TaskStatus.PROCESSING)


def _run_simulation_task_impl(
    self: Any,
    payload: dict[str, Any],
) -> dict[str, list[float]]:
    """Celery task executing SEIZSimulator.solve() and persisting DB state."""
    task_id = str(getattr(self.request, "id", None) or uuid.uuid4())
    req = SimulationRequest.model_validate(payload)
    _create_task_memory(task_id, task_type="simulation", initial_status=TaskStatus.PENDING)
    _update_task_memory(task_id, status=TaskStatus.PROCESSING)
    _run_async(
        _ensure_and_mark_processing(
            task_id=task_id,
            task_type="simulation",
            initial_state=req.initial_state.model_dump(),
            parameters=req.parameters.model_dump(),
        )
    )

    t_start = time.perf_counter()
    with structlog.contextvars.bound_contextvars(task_id=task_id):
        try:
            sim_result = SEIZSimulator().solve(
                state=req.initial_state,
                params=req.parameters,
                t_span=req.t_span,
                num_points=req.num_points,
                stochastic=req.stochastic,
                num_simulations=req.num_simulations,
            )
            execution_time_ms = round((time.perf_counter() - t_start) * 1000.0, 4)
            series = series_to_dict(sim_result)
            _run_async(
                SIMULATION_REPO.update_status(
                    task_id,
                    status=TaskStatus.COMPLETED,
                    series=series,
                )
            )
            _update_task_memory(
                task_id,
                status=TaskStatus.COMPLETED,
                result=series,
                error=None,
                error_type=None,
            )
            logger.info(
                "simulation_task_completed",
                task_id=task_id,
                execution_time_ms=execution_time_ms,
                solver_status=TaskStatus.COMPLETED.value,
            )
            return series
        except Exception as exc:
            execution_time_ms = round((time.perf_counter() - t_start) * 1000.0, 4)
            _run_async(
                SIMULATION_REPO.update_status(
                    task_id,
                    status=TaskStatus.FAILED,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            )
            _update_task_memory(
                task_id,
                status=TaskStatus.FAILED,
                result=None,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            logger.error(
                "simulation_task_failed",
                task_id=task_id,
                execution_time_ms=execution_time_ms,
                solver_status=TaskStatus.FAILED.value,
                error=str(exc),
            )
            raise


def _run_calibration_task_impl(
    self: Any,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Celery task executing ParameterEstimator.fit_to_data() and persisting DB state."""
    task_id = str(getattr(self.request, "id", None) or uuid.uuid4())
    req = CalibrationRequest.model_validate(payload)
    _create_task_memory(task_id, task_type="calibration", initial_status=TaskStatus.PENDING)
    _update_task_memory(task_id, status=TaskStatus.PROCESSING)
    _run_async(
        _ensure_and_mark_processing(
            task_id=task_id,
            task_type="calibration",
            initial_state=req.initial_state.model_dump(),
        )
    )

    t_start = time.perf_counter()
    with structlog.contextvars.bound_contextvars(task_id=task_id):
        try:
            cal_result = execute_calibration(req)
            execution_time_ms = round((time.perf_counter() - t_start) * 1000.0, 4)
            _run_async(
                SIMULATION_REPO.update_status(
                    task_id,
                    status=TaskStatus.COMPLETED,
                    series=cal_result["fitted_series"],
                    parameters=cal_result["parameters"],
                    sse=float(cal_result["sse"]),
                )
            )
            _update_task_memory(
                task_id,
                status=TaskStatus.COMPLETED,
                result=cal_result,
                error=None,
                error_type=None,
            )
            logger.info(
                "calibration_task_completed",
                task_id=task_id,
                execution_time_ms=execution_time_ms,
                solver_status=TaskStatus.COMPLETED.value,
            )
            return cal_result
        except Exception as exc:
            execution_time_ms = round((time.perf_counter() - t_start) * 1000.0, 4)
            _run_async(
                SIMULATION_REPO.update_status(
                    task_id,
                    status=TaskStatus.FAILED,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            )
            _update_task_memory(
                task_id,
                status=TaskStatus.FAILED,
                result=None,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            logger.error(
                "calibration_task_failed",
                task_id=task_id,
                execution_time_ms=execution_time_ms,
                solver_status=TaskStatus.FAILED.value,
                error=str(exc),
            )
            raise


run_simulation_task: CeleryTaskProtocol[[dict[str, Any]], dict[str, list[float]]] = cast(
    CeleryTaskProtocol[[dict[str, Any]], dict[str, list[float]]],
    celery_app.task(
        bind=True,
        name="src.workers.celery_app.run_simulation_task",
    )(_run_simulation_task_impl),
)

run_calibration_task: CeleryTaskProtocol[[dict[str, Any]], dict[str, Any]] = cast(
    CeleryTaskProtocol[[dict[str, Any]], dict[str, Any]],
    celery_app.task(
        bind=True,
        name="src.workers.celery_app.run_calibration_task",
    )(_run_calibration_task_impl),
)
