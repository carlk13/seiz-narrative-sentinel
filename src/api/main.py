"""FastAPI application entrypoint and global HTTP 422 exception handlers."""

from __future__ import annotations

import structlog
from fastapi import Depends, FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import ValidationError
from slowapi.errors import RateLimitExceeded

from src.api.routes import router
from src.api.security import limiter, rate_limit_exceeded_handler, verify_api_key
from src.models.schemas import ErrorResponse
from src.simulation.calibration import OptimizationConvergenceError
from src.simulation.engine import configure_structlog

configure_structlog()
logger = structlog.get_logger(__name__)

app = FastAPI(
    title="SEIZ Disinformation Trajectory Simulator API",
    version="0.1.0",
    description=(
        "Asynchronous REST API for SEIZ epidemiological simulation and parameter calibration."
    ),
    dependencies=[Depends(verify_api_key)],
)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    rate_limit_exceeded_handler,  # type: ignore[arg-type]
)
app.include_router(router)


@app.exception_handler(OptimizationConvergenceError)
async def optimization_convergence_exception_handler(
    request: Request,
    exc: OptimizationConvergenceError,
) -> JSONResponse:
    """Return formatted HTTP 422 response when parameter optimization fails."""
    del request
    logger.warning("http_422_optimization_convergence_error", detail=str(exc))
    payload = ErrorResponse(
        error="OptimizationConvergenceError",
        detail=str(exc),
    )
    return JSONResponse(status_code=422, content=payload.model_dump())


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return formatted HTTP 422 response for FastAPI request validation errors."""
    del request
    payload = ErrorResponse(
        error="ValidationError",
        detail=jsonable_encoder(exc.errors()),
    )
    return JSONResponse(status_code=422, content=payload.model_dump())


@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(
    request: Request,
    exc: ValidationError,
) -> JSONResponse:
    """Return formatted HTTP 422 response for direct Pydantic ValidationError exceptions."""
    del request
    payload = ErrorResponse(
        error="ValidationError",
        detail=jsonable_encoder(exc.errors()),
    )
    return JSONResponse(status_code=422, content=payload.model_dump())


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Liveness check endpoint."""
    return {"status": "ok"}


@app.get("/metrics", tags=["Observability"])
async def metrics_endpoint() -> Response:
    """Expose Prometheus metrics for scraping."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
