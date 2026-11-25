"""FastAPI API key authentication and Redis/in-memory rate limiting via slowapi."""

from __future__ import annotations

import os
import secrets
import socket
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Protocol, cast
from urllib.parse import urlparse

import structlog
from fastapi import HTTPException, Request, Response, Security, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.models.schemas import ErrorResponse
from src.simulation.engine import configure_structlog

configure_structlog()
logger = structlog.get_logger(__name__)

API_KEY_HEADER_NAME = "X-API-Key"
DEFAULT_API_KEY = os.getenv("SEIZ_API_KEY", "seiz-secret-api-key")
SIMULATION_RATE_LIMIT = "5/minute"
CALIBRATION_RATE_LIMIT = "5/minute"

api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


class RateLimiterProtocol(Protocol):
    """Typed protocol matching the runtime slowapi.Limiter interface under strict mypy."""

    def limit[**P, R](
        self,
        limit_value: str,
    ) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]: ...

    def reset(self) -> None: ...

    def _inject_headers(self, response: Response, current_limit: Any) -> Response: ...


def get_valid_api_keys() -> set[str]:
    """Return configured valid API keys from environment variables."""
    keys: set[str] = set()
    primary = os.getenv("SEIZ_API_KEY", DEFAULT_API_KEY).strip()
    if primary:
        keys.add(primary)
    extra_raw = os.getenv("SEIZ_API_KEYS", "")
    for item in extra_raw.split(","):
        cleaned = item.strip()
        if cleaned:
            keys.add(cleaned)
    return keys


async def verify_api_key(
    api_key: Annotated[str | None, Security(api_key_header)] = None,
) -> str:
    """Validate incoming X-API-Key header using constant-time comparison."""
    valid_keys = get_valid_api_keys()
    if api_key is not None and any(
        secrets.compare_digest(api_key, candidate) for candidate in valid_keys
    ):
        return api_key

    logger.warning(
        "api_key_authentication_failed",
        has_header=api_key is not None,
    )
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing X-API-Key header.",
        headers={"WWW-Authenticate": "ApiKey"},
    )


def resolve_rate_limit_storage_uri(redis_url: str | None = None) -> str:
    """Return reachable Redis storage URI or gracefully fall back to in-memory storage."""
    candidate = (
        redis_url
        if redis_url is not None
        else (os.getenv("REDIS_URL") or os.getenv("SEIZ_REDIS_URL"))
    )
    if not candidate:
        return "memory://"

    if candidate.startswith(("redis://", "rediss://")):
        parsed = urlparse(candidate)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 6379
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return candidate
        except (OSError, ValueError) as exc:
            logger.warning(
                "rate_limiter_redis_unreachable_fallback_memory",
                redis_url=candidate,
                error=str(exc),
            )
            return "memory://"

    return candidate


def get_rate_limit_key(request: Request) -> str:
    """Derive a per-client rate-limit key combining client IP and X-API-Key when provided."""
    api_key = request.headers.get(API_KEY_HEADER_NAME) or request.headers.get(
        API_KEY_HEADER_NAME.lower()
    )
    client_ip = get_remote_address(request) or "127.0.0.1"
    if api_key:
        return f"{client_ip}:{api_key}"
    return str(client_ip)


limiter: RateLimiterProtocol = cast(
    RateLimiterProtocol,
    Limiter(
        key_func=get_rate_limit_key,
        storage_uri=resolve_rate_limit_storage_uri(),
        in_memory_fallback_enabled=True,
    ),
)


def reset_rate_limiter() -> None:
    """Reset in-memory rate-limit counters (intended for isolated test runs)."""
    limiter.reset()


async def rate_limit_exceeded_handler(
    request: Request,
    exc: RateLimitExceeded,
) -> JSONResponse:
    """Return clean HTTP 429 Too Many Requests response on rate-limit violations."""
    logger.warning(
        "rate_limit_exceeded",
        path=str(request.url.path),
        detail=str(exc.detail),
    )
    payload = ErrorResponse(
        error="RateLimitExceeded",
        detail=f"Rate limit exceeded: {exc.detail}",
    )
    response = JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=payload.model_dump(),
    )
    view_limit = getattr(request.state, "view_rate_limit", None)
    app_limiter = getattr(request.app.state, "limiter", None)
    if view_limit is not None and app_limiter is not None:
        return cast(JSONResponse, app_limiter._inject_headers(response, view_limit))
    return response
