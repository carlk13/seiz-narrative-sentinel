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
from src.ingestion.global_tracker import scan_global_narrative
from src.ingestion.poller import InfluxTimeSeriesStore, aggregate_post_counts, classify_post
from src.models.db import TrackedNarrative
from src.models.schemas import (
    CalibrationRequest,
    CountrySpreadStat,
    CurrentDataResponse,
    GlobalMentionItem,
    GlobalScanRequest,
    GlobalScanResponse,
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
    TrackedNarrativeCreateRequest,
    TrackedNarrativeResponse,
)
from src.simulation.calibration import OptimizationConvergenceError, ParameterEstimator
from src.simulation.engine import SEIZSimulator
from src.simulation.scenarios import get_scenario_presets
from src.workers.celery_app import (
    _STORE_LOCK,
    NARRATIVE_REPO,
    SIMULATION_REPO,
    TASK_STORE,
    _create_task_memory,
    _update_task_memory,
    celery_app,
    evaluate_global_scan_with_seiz,
    execute_calibration,
    poll_and_evaluate_narratives_task,
    reset_database_state,
    run_calibration_task,
    run_simulation_task,
)

__all__ = [
    "INFLUX_STORE",
    "NARRATIVE_REPO",
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


def _narrative_to_response(row: TrackedNarrative) -> TrackedNarrativeResponse:
    """Serialize a TrackedNarrative ORM row to TrackedNarrativeResponse."""
    return TrackedNarrativeResponse(
        id=row.id,
        name=row.name,
        query=row.query,
        sources=list(row.sources),
        active=row.active,
        alert_velocity_threshold=row.alert_velocity_threshold,
        alert_rsi_threshold=row.alert_rsi_threshold,
        last_checked_at=row.last_checked_at.isoformat() if row.last_checked_at else None,
        last_infected_count=row.last_infected_count,
        last_skeptic_count=row.last_skeptic_count,
        last_max_velocity=row.last_max_velocity,
        last_inflection_time=row.last_inflection_time,
        last_rsi=row.last_rsi,
        alert_triggered=row.alert_triggered,
        alert_message=row.alert_message,
        country_counts=row.country_counts,
        created_at=row.created_at.isoformat(),
        updated_at=row.updated_at.isoformat(),
    )


@router.post(
    "/global/scan",
    response_model=GlobalScanResponse,
)
@limiter.limit("15/minute")
async def scan_global_spread(
    request: Request,
    payload: GlobalScanRequest,
    _api_key: Annotated[str, Depends(verify_api_key)],
) -> GlobalScanResponse:
    """Scan GDELT, Bluesky, and Reddit live for global narrative spread and SEIZ fitting."""
    del request, _api_key
    scan_data = await scan_global_narrative(
        query=payload.query,
        sources=payload.sources,
        max_per_source=payload.max_per_source,
        timespan=payload.timespan,
    )
    seiz_eval = await run_in_threadpool(
        evaluate_global_scan_with_seiz,
        scan_data,
        payload.calibrate,
        payload.population_size,
        payload.alert_velocity_threshold,
        payload.alert_rsi_threshold,
    )
    return GlobalScanResponse(
        query=scan_data["query"],
        sources_queried=scan_data["sources_queried"],
        source_counts=scan_data["source_counts"],
        total_mentions=scan_data["total_mentions"],
        stance_counts=scan_data["stance_counts"],
        country_stats=[CountrySpreadStat(**item) for item in scan_data["country_stats"]],
        timestamps=scan_data["timestamps"],
        t_observed=scan_data["t_observed"],
        I_observed=scan_data["I_observed"],
        Z_observed=scan_data["Z_observed"],
        mentions=[GlobalMentionItem(**item) for item in scan_data["mentions"]],
        calibrated_parameters=seiz_eval["calibrated_parameters"],
        forecast_summary=seiz_eval["forecast_summary"],
        forecast_series=seiz_eval["forecast_series"],
        alert_triggered=bool(seiz_eval["alert_triggered"]),
        alert_message=seiz_eval["alert_message"],
    )


@router.get(
    "/global/narratives",
    response_model=list[TrackedNarrativeResponse],
)
async def list_tracked_narratives(
    active_only: bool = Query(default=False, description="Only return active watchlist monitors"),
) -> list[TrackedNarrativeResponse]:
    """List persisted global narrative monitors and their latest alert evaluations."""
    rows = await NARRATIVE_REPO.list_narratives(active_only=active_only)
    return [_narrative_to_response(row) for row in rows]


@router.post(
    "/global/narratives",
    response_model=TrackedNarrativeResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_tracked_narrative(
    payload: TrackedNarrativeCreateRequest,
) -> TrackedNarrativeResponse:
    """Register a global narrative monitor and optionally trigger an immediate scan."""
    narrative_id = str(uuid.uuid4())
    row = await NARRATIVE_REPO.create_narrative(
        narrative_id=narrative_id,
        name=payload.name,
        query=payload.query,
        sources=list(payload.sources),
        alert_velocity_threshold=payload.alert_velocity_threshold,
        alert_rsi_threshold=payload.alert_rsi_threshold,
    )
    if payload.run_initial_scan:
        poll_and_evaluate_narratives_task.delay(narrative_id)
        refreshed = await NARRATIVE_REPO.get_narrative(narrative_id)
        if refreshed is not None:
            row = refreshed
    return _narrative_to_response(row)


@router.post(
    "/global/narratives/{narrative_id}/refresh",
    response_model=TrackedNarrativeResponse,
)
async def refresh_tracked_narrative(narrative_id: str) -> TrackedNarrativeResponse:
    """Trigger an on-demand global scan & SEIZ alert evaluation via Celery."""
    existing = await NARRATIVE_REPO.get_narrative(narrative_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tracked narrative '{narrative_id}' not found.",
        )
    poll_and_evaluate_narratives_task.delay(narrative_id)
    updated = await NARRATIVE_REPO.get_narrative(narrative_id)
    assert updated is not None
    return _narrative_to_response(updated)


@router.delete(
    "/global/narratives/{narrative_id}",
)
async def delete_tracked_narrative(narrative_id: str) -> dict[str, str]:
    """Delete a tracked global narrative monitor."""
    deleted = await NARRATIVE_REPO.delete_narrative(narrative_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tracked narrative '{narrative_id}' not found.",
        )
    return {"deleted_id": narrative_id}
