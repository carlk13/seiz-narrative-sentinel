"""FastAPI endpoints dispatching SEIZ simulation and calibration jobs via Celery."""

from __future__ import annotations

import time
import uuid
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool

from src.api.security import (
    CALIBRATION_RATE_LIMIT,
    SIMULATION_RATE_LIMIT,
    limiter,
    verify_api_key,
)
from src.ingestion.poller import InfluxTimeSeriesStore, aggregate_post_counts, classify_post
from src.models.schemas import (
    CalibrationRequest,
    CurrentDataResponse,
    ParameterSweepRequest,
    ParameterSweepResponse,
    PostClassificationItem,
    PostClassificationRequest,
    PostClassificationResponse,
    RunHistoryResponse,
    RunSummaryItem,
    ScenarioPreset,
    SimulationRequest,
    TaskStatus,
    TaskStatusResponse,
    TaskSubmissionResponse,
)
from src.simulation.calibration import OptimizationConvergenceError, ParameterEstimator
from src.simulation.engine import SEIZSimulator
from src.simulation.scenarios import get_scenario_presets
from src.workers.celery_app import (
    _STORE_LOCK,
    SIMULATION_REPO,
    TASK_STORE,
    _create_task_memory,
    _update_task_memory,
    celery_app,
    execute_calibration,
    reset_database_state,
    run_calibration_task,
    run_simulation_task,
)

__all__ = [
    "INFLUX_STORE",
    "ParameterEstimator",
    "SEIZSimulator",
    "SIMULATION_REPO",
    "TASK_STORE",
    "_STORE_LOCK",
    "reset_database_state",
    "router",
]

router = APIRouter(
    prefix="/api/v1",
    tags=["SEIZ Simulation & Calibration"],
    dependencies=[Depends(verify_api_key)],
)

INFLUX_STORE = InfluxTimeSeriesStore()


@router.post(
    "/simulate",
    response_model=TaskSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
@limiter.limit(SIMULATION_RATE_LIMIT)
async def simulate_seiz(
    request: Request,
    payload: SimulationRequest,
    _api_key: Annotated[str, Depends(verify_api_key)],
) -> TaskSubmissionResponse:
    """Push an asynchronous SEIZ simulation task to Celery/Redis and return task_id."""
    del request, _api_key
    payload_dict = payload.model_dump()
    async_result = run_simulation_task.delay(payload_dict)
    task_id = str(async_result.id)
    if await SIMULATION_REPO.get_run(task_id) is None:
        await SIMULATION_REPO.create_run(
            task_id=task_id,
            task_type="simulation",
            status=TaskStatus.PENDING,
            initial_state=payload_dict["initial_state"],
            parameters=payload_dict["parameters"],
        )
        _create_task_memory(
            task_id,
            task_type="simulation",
            initial_status=TaskStatus.PENDING,
        )
    return TaskSubmissionResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        task_type="simulation",
    )


@router.post(
    "/calibrate",
    response_model=TaskSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
@limiter.limit(CALIBRATION_RATE_LIMIT)
async def calibrate_seiz(
    request: Request,
    payload: CalibrationRequest,
    _api_key: Annotated[str, Depends(verify_api_key)],
) -> TaskSubmissionResponse:
    """Queue via Celery or synchronously execute SEIZ parameter calibration."""
    del request, _api_key
    if payload.sync:
        task_id = str(uuid.uuid4())
        await SIMULATION_REPO.create_run(
            task_id=task_id,
            task_type="calibration",
            status=TaskStatus.PROCESSING,
            initial_state=payload.initial_state.model_dump(),
        )
        _create_task_memory(
            task_id,
            task_type="calibration",
            initial_status=TaskStatus.PROCESSING,
        )
        try:
            cal_result = await run_in_threadpool(execute_calibration, payload)
        except OptimizationConvergenceError as exc:
            await SIMULATION_REPO.update_status(
                task_id,
                status=TaskStatus.FAILED,
                error=str(exc),
                error_type="OptimizationConvergenceError",
            )
            _update_task_memory(
                task_id,
                status=TaskStatus.FAILED,
                result=None,
                error=str(exc),
                error_type="OptimizationConvergenceError",
            )
            raise
        except Exception as exc:
            await SIMULATION_REPO.update_status(
                task_id,
                status=TaskStatus.FAILED,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            _update_task_memory(
                task_id,
                status=TaskStatus.FAILED,
                result=None,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise
        await SIMULATION_REPO.update_status(
            task_id,
            status=TaskStatus.COMPLETED,
            series=cal_result["fitted_series"],
            parameters=cal_result["parameters"],
            sse=float(cal_result["sse"]),
        )
        _update_task_memory(
            task_id,
            status=TaskStatus.COMPLETED,
            result=cal_result,
            error=None,
            error_type=None,
        )
        return TaskSubmissionResponse(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            task_type="calibration",
        )

    payload_dict = payload.model_dump()
    async_result = run_calibration_task.delay(payload_dict)
    task_id = str(async_result.id)
    if await SIMULATION_REPO.get_run(task_id) is None:
        await SIMULATION_REPO.create_run(
            task_id=task_id,
            task_type="calibration",
            status=TaskStatus.PENDING,
            initial_state=payload_dict["initial_state"],
        )
        _create_task_memory(
            task_id,
            task_type="calibration",
            initial_status=TaskStatus.PENDING,
        )
    return TaskSubmissionResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        task_type="calibration",
    )


@router.get(
    "/results/{task_id}",
    response_model=TaskStatusResponse,
)
async def get_task_results(
    task_id: str,
    raise_on_error: bool = Query(
        default=False,
        description="When true, re-raise OptimizationConvergenceError for HTTP 422 formatting",
    ),
) -> TaskStatusResponse:
    """Retrieve task status and results for a submitted simulation or calibration job."""
    record: dict[str, Any] | None = await SIMULATION_REPO.get_run_dict(task_id)
    if record is None:
        celery_res = celery_app.AsyncResult(task_id)
        if celery_res.state in ("STARTED", "SUCCESS", "FAILURE"):
            mapped_status = (
                TaskStatus.PROCESSING
                if celery_res.state == "STARTED"
                else (
                    TaskStatus.COMPLETED
                    if celery_res.state == "SUCCESS"
                    else TaskStatus.FAILED
                )
            )
            return TaskStatusResponse(
                task_id=task_id,
                status=mapped_status,
                task_type="simulation",
                result=celery_res.result if celery_res.state == "SUCCESS" else None,
                error=str(celery_res.result) if celery_res.state == "FAILURE" else None,
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )

    if (
        raise_on_error
        and record["status"] == TaskStatus.FAILED
        and record.get("error_type") == "OptimizationConvergenceError"
    ):
        raise OptimizationConvergenceError(record["error"] or "Optimization failed to converge.")

    return TaskStatusResponse(
        task_id=record["task_id"],
        status=record["status"],
        task_type=record["task_type"],
        result=record["result"],
        error=record["error"],
    )


@router.get(
    "/data/current",
    response_model=CurrentDataResponse,
)
async def get_current_social_data(
    keyword: str = Query(
        default="disinfo",
        description="Tracked keyword tag in InfluxDB",
    ),
    limit: int = Query(
        default=60,
        ge=2,
        le=10_000,
        description="Maximum number of minute-level points to return",
    ),
) -> CurrentDataResponse:
    """Query InfluxDB for social metrics and return formatted I(t) and Z(t) arrays."""
    payload = await run_in_threadpool(INFLUX_STORE.query_series, keyword, limit)
    return CurrentDataResponse(**payload)


_PARAM_INDEX: dict[str, int] = {
    "beta": 0,
    "p": 1,
    "l": 2,
    "b": 3,
    "rho": 4,
    "epsilon": 5,
}


def _execute_parameter_sweep(payload: ParameterSweepRequest) -> ParameterSweepResponse:
    """Evaluate a 2D SEIZ parameter grid using the vectorized Eigen3 batch solver."""
    t_start = time.perf_counter()
    x_vals = np.linspace(payload.x_range[0], payload.x_range[1], payload.x_steps, dtype=np.float64)
    y_vals = np.linspace(payload.y_range[0], payload.y_range[1], payload.y_steps, dtype=np.float64)
    xx, yy = np.meshgrid(x_vals, y_vals)

    total = int(payload.x_steps * payload.y_steps)
    bp = payload.base_parameters
    base_vec = np.array(
        [bp.beta, bp.p, bp.l, bp.b, bp.rho, bp.epsilon],
        dtype=np.float64,
    )
    params_batch = np.tile(base_vec, (total, 1))
    params_batch[:, _PARAM_INDEX[payload.x_param]] = xx.ravel()
    params_batch[:, _PARAM_INDEX[payload.y_param]] = yy.ravel()

    trajectories = SEIZSimulator().solve_batch(
        state=payload.initial_state,
        params_batch=params_batch,
        t_span=payload.t_span,
        num_points=payload.num_points,
    )
    t_grid = np.linspace(payload.t_span[0], payload.t_span[1], payload.num_points, dtype=np.float64)
    i_curves = trajectories[:, :, 2]
    total_n = float(payload.initial_state.N)

    final_infected_pct = (100.0 * i_curves[:, -1] / total_n).reshape(
        payload.y_steps, payload.x_steps
    )
    di_dt = np.gradient(i_curves, t_grid, axis=1)
    peak_indices = np.argmax(di_dt, axis=1)
    max_vel = di_dt[np.arange(total), peak_indices].reshape(payload.y_steps, payload.x_steps)
    inflection_t = t_grid[peak_indices].reshape(payload.y_steps, payload.x_steps)

    elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 3)
    return ParameterSweepResponse(
        x_param=payload.x_param,
        y_param=payload.y_param,
        x_values=[float(v) for v in x_vals],
        y_values=[float(v) for v in y_vals],
        final_infected_pct=final_infected_pct.tolist(),
        max_viral_velocity=max_vel.tolist(),
        inflection_time=inflection_t.tolist(),
        total_combinations=total,
        execution_time_ms=elapsed_ms,
    )


@router.get(
    "/presets",
    response_model=list[ScenarioPreset],
)
async def list_scenario_presets() -> list[ScenarioPreset]:
    """Return curated real-world disinformation scenario presets."""
    return get_scenario_presets()


@router.get(
    "/runs",
    response_model=RunHistoryResponse,
)
async def list_simulation_runs(
    limit: int = Query(default=25, ge=1, le=200, description="Max runs to return"),
    task_type: Literal["simulation", "calibration"] | None = Query(
        default=None,
        description="Optional filter by task_type",
    ),
) -> RunHistoryResponse:
    """List recently persisted SimulationRun records from the database."""
    runs = await SIMULATION_REPO.list_runs(limit=limit, task_type=task_type)
    items = [
        RunSummaryItem(
            task_id=run.id,
            task_type=run.task_type,
            status=TaskStatus(run.status),
            initial_state=run.initial_state,
            parameters=run.parameters,
            sse=run.sse,
            error=run.error,
            created_at=run.created_at.isoformat(),
            updated_at=run.updated_at.isoformat(),
        )
        for run in runs
    ]
    return RunHistoryResponse(total=len(items), runs=items)


@router.post(
    "/sweep",
    response_model=ParameterSweepResponse,
)
@limiter.limit("15/minute")
async def run_parameter_sweep(
    request: Request,
    payload: ParameterSweepRequest,
    _api_key: Annotated[str, Depends(verify_api_key)],
) -> ParameterSweepResponse:
    """Run a vectorized 2D parameter sensitivity sweep via the Eigen3 batch RK4 core."""
    del request, _api_key
    return await run_in_threadpool(_execute_parameter_sweep, payload)


@router.post(
    "/ingest/analyze",
    response_model=PostClassificationResponse,
)
async def analyze_social_posts(
    payload: PostClassificationRequest,
) -> PostClassificationResponse:
    """Classify a batch of raw posts into Infected, Skeptic, or Neutral narratives."""
    items = [
        PostClassificationItem(text=post_text, label=classify_post(post_text))
        for post_text in payload.posts
    ]
    counts = aggregate_post_counts(payload.posts)
    if payload.persist_to_influx:
        await run_in_threadpool(
            INFLUX_STORE.write_minute_counts,
            payload.keyword,
            counts["Infected"],
            counts["Skeptic"],
        )
    return PostClassificationResponse(
        keyword=payload.keyword,
        counts=counts,
        items=items,
        persisted=payload.persist_to_influx,
    )
