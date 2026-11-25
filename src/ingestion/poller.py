"""Autonomous social metrics ingestion daemon, keyword NLP classifier, and InfluxDB v2 store."""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, TypedDict, cast

import httpx
import structlog
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from src.simulation.engine import configure_structlog

configure_structlog()
logger = structlog.get_logger(__name__)


class CurrentSeriesPayload(TypedDict):
    """Typed dictionary matching CurrentDataResponse fields."""

    keyword: str
    timestamps: list[str]
    t_observed: list[float]
    I_observed: list[float]
    Z_observed: list[float]
    I: list[float]  # noqa: E741
    Z: list[float]

DEFAULT_INFECTED_KEYWORDS: tuple[str, ...] = (
    "breaking",
    "confirmed",
    "leaked",
    "exposed",
    "truth",
    "share",
    "viral",
    "cover-up",
    "coverup",
    "undeniable",
    "wake up",
    "bombshell",
    "proof",
)

DEFAULT_SKEPTIC_KEYWORDS: tuple[str, ...] = (
    "debunk",
    "debunked",
    "fact-check",
    "fact check",
    "false",
    "hoax",
    "misleading",
    "fake",
    "unverified",
    "myth",
    "disproven",
    "fabricated",
    "refuted",
)

DEFAULT_POLL_URL = os.getenv("SEIZ_SOCIAL_API_URL", "http://127.0.0.1:8088/mock/posts")
DEFAULT_INFLUX_URL = os.getenv("INFLUXDB_URL", "http://127.0.0.1:8086")
DEFAULT_INFLUX_TOKEN = os.getenv("INFLUXDB_TOKEN", "seiz-dev-token")
DEFAULT_INFLUX_ORG = os.getenv("INFLUXDB_ORG", "seiz-org")
DEFAULT_INFLUX_BUCKET = os.getenv("INFLUXDB_BUCKET", "seiz_metrics")


def floor_to_minute(dt: datetime | None = None) -> datetime:
    """Normalize a datetime to UTC and truncate seconds/microseconds to the minute."""
    ts = datetime.now(UTC) if dt is None else dt
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    else:
        ts = ts.astimezone(UTC)
    return ts.replace(second=0, microsecond=0)


def classify_post(
    text: str,
    infected_keywords: Sequence[str] = DEFAULT_INFECTED_KEYWORDS,
    skeptic_keywords: Sequence[str] = DEFAULT_SKEPTIC_KEYWORDS,
) -> Literal["Infected", "Skeptic", "Neutral"]:
    """Classify a social post into Infected, Skeptic, or Neutral deterministically."""
    lowered = text.lower()
    infected_hits = sum(1 for kw in infected_keywords if kw.lower() in lowered)
    skeptic_hits = sum(1 for kw in skeptic_keywords if kw.lower() in lowered)

    if skeptic_hits > infected_hits:
        return "Skeptic"
    if infected_hits > skeptic_hits:
        return "Infected"
    if infected_hits > 0 and infected_hits == skeptic_hits:
        # Debunking context takes precedence when quoting a claim to refute it.
        return "Skeptic"
    return "Neutral"


def _extract_post_text(post: str | dict[str, Any]) -> str:
    """Extract searchable text from a raw string or social API post dictionary."""
    if isinstance(post, str):
        return post
    parts: list[str] = []
    for key in ("title", "text", "body", "selftext", "content"):
        val = post.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val)
    return " ".join(parts)


def aggregate_post_counts(
    posts: Sequence[str | dict[str, Any]],
    infected_keywords: Sequence[str] = DEFAULT_INFECTED_KEYWORDS,
    skeptic_keywords: Sequence[str] = DEFAULT_SKEPTIC_KEYWORDS,
) -> dict[str, int]:
    """Classify a batch of posts and return aggregate Infected, Skeptic, and Neutral counts."""
    counts = {"Infected": 0, "Skeptic": 0, "Neutral": 0}
    for post in posts:
        text = _extract_post_text(post)
        label = classify_post(
            text,
            infected_keywords=infected_keywords,
            skeptic_keywords=skeptic_keywords,
        )
        counts[label] += 1
    return counts


class InfluxTimeSeriesStore:
    """InfluxDB v2 time-series client wrapper with in-memory fallback for local development."""

    def __init__(
        self,
        url: str = DEFAULT_INFLUX_URL,
        token: str = DEFAULT_INFLUX_TOKEN,
        org: str = DEFAULT_INFLUX_ORG,
        bucket: str = DEFAULT_INFLUX_BUCKET,
        influx_client: InfluxDBClient | None = None,
    ) -> None:
        """Initialize the InfluxDB v2 client and thread-safe local record mirror."""
        self.url: str = url
        self.token: str = token
        self.org: str = org
        self.bucket: str = bucket
        self._owns_client: bool = influx_client is None
        self._client: InfluxDBClient = (
            influx_client
            if influx_client is not None
            else InfluxDBClient(url=self.url, token=self.token, org=self.org)
        )
        self._lock = threading.Lock()
        self._records: dict[str, dict[datetime, tuple[int, int]]] = {}

    def clear(self, keyword: str | None = None) -> None:
        """Clear mirrored in-memory records (useful for test isolation)."""
        with self._lock:
            if keyword is None:
                self._records.clear()
            else:
                self._records.pop(keyword, None)

    def close(self) -> None:
        """Close the underlying InfluxDB client if owned by this store."""
        if self._owns_client:
            self._client.close()

    def write_minute_counts(
        self,
        keyword: str,
        infected_count: int,
        skeptic_count: int,
        timestamp: datetime | None = None,
    ) -> Point:
        """Write minute-timestamped Infected and Skeptic counts to InfluxDB v2."""
        minute_ts = floor_to_minute(timestamp)
        point = (
            Point("seiz_social_metrics")
            .tag("keyword", keyword)
            .field("Infected", int(infected_count))
            .field("Skeptic", int(skeptic_count))
            .time(minute_ts, WritePrecision.S)
        )

        with self._lock:
            kw_records = self._records.setdefault(keyword, {})
            kw_records[minute_ts] = (int(infected_count), int(skeptic_count))

        try:
            write_api = self._client.write_api(write_options=SYNCHRONOUS)
            write_api.write(bucket=self.bucket, org=self.org, record=point)
        except Exception as exc:
            logger.warning(
                "influxdb_write_failed",
                error=str(exc),
                keyword=keyword,
            )

        return point

    def _parse_flux_tables(self, tables: Any) -> list[tuple[datetime, float, float]]:
        """Parse pivoted or unpivoted InfluxDB FluxTable outputs into sorted rows."""
        grouped: dict[datetime, list[float]] = {}
        for table in tables or []:
            for record in getattr(table, "records", []):
                raw_time = record.get_time() if hasattr(record, "get_time") else None
                if not isinstance(raw_time, datetime):
                    continue
                minute_ts = floor_to_minute(raw_time)
                bucket_vals = grouped.setdefault(minute_ts, [0.0, 0.0])

                values = getattr(record, "values", {})
                if "Infected" in values or "Skeptic" in values:
                    if values.get("Infected") is not None:
                        bucket_vals[0] = float(values["Infected"])
                    if values.get("Skeptic") is not None:
                        bucket_vals[1] = float(values["Skeptic"])
                elif hasattr(record, "get_field") and hasattr(record, "get_value"):
                    field_name = record.get_field()
                    field_val = record.get_value()
                    if field_name == "Infected" and field_val is not None:
                        bucket_vals[0] = float(field_val)
                    elif field_name == "Skeptic" and field_val is not None:
                        bucket_vals[1] = float(field_val)

        return [
            (ts, vals[0], vals[1])
            for ts, vals in sorted(grouped.items(), key=lambda item: item[0])
        ]

    @staticmethod
    def _default_baseline_rows(limit: int) -> list[tuple[datetime, float, float]]:
        """Generate a deterministic multi-minute baseline series if no points are stored."""
        n_points = max(2, min(limit, 10))
        base_ts = floor_to_minute(datetime.now(UTC)) - timedelta(minutes=n_points - 1)
        rows: list[tuple[datetime, float, float]] = []
        for idx in range(n_points):
            ts = base_ts + timedelta(minutes=idx)
            infected = float(15 + 3 * idx)
            skeptic = float(10 + 2 * idx)
            rows.append((ts, infected, skeptic))
        return rows

    def query_series(
        self,
        keyword: str = "disinfo",
        limit: int = 60,
    ) -> CurrentSeriesPayload:
        """Query InfluxDB (or local mirror) and return formatted I(t) and Z(t) arrays."""
        safe_keyword = keyword.replace('"', '\\"')
        flux_query = (
            f'from(bucket: "{self.bucket}")\n'
            f"  |> range(start: -24h)\n"
            f'  |> filter(fn: (r) => r["_measurement"] == "seiz_social_metrics" '
            f'and r["keyword"] == "{safe_keyword}")\n'
            f'  |> filter(fn: (r) => r["_field"] == "Infected" or r["_field"] == "Skeptic")\n'
            f'  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")\n'
            f'  |> sort(columns: ["_time"])\n'
            f"  |> tail(n: {int(limit)})"
        )

        rows: list[tuple[datetime, float, float]] = []
        try:
            query_api = self._client.query_api()
            tables = query_api.query(query=flux_query, org=self.org)
            rows = self._parse_flux_tables(tables)
        except Exception as exc:
            logger.warning(
                "influxdb_query_failed",
                error=str(exc),
                keyword=keyword,
            )

        if not rows:
            with self._lock:
                kw_records = self._records.get(keyword, {})
                sorted_items = sorted(kw_records.items(), key=lambda item: item[0])
                if limit > 0:
                    sorted_items = sorted_items[-limit:]
                rows = [(ts, float(vals[0]), float(vals[1])) for ts, vals in sorted_items]

        if not rows:
            rows = self._default_baseline_rows(limit)

        t0 = rows[0][0]
        timestamps = [ts.isoformat() for ts, _, _ in rows]
        t_observed = [float((ts - t0).total_seconds() / 60.0) for ts, _, _ in rows]
        i_observed = [float(i_val) for _, i_val, _ in rows]
        z_observed = [float(z_val) for _, _, z_val in rows]

        return {
            "keyword": keyword,
            "timestamps": timestamps,
            "t_observed": t_observed,
            "I_observed": i_observed,
            "Z_observed": z_observed,
            "I": i_observed,
            "Z": z_observed,
        }


class SocialMetricsPoller:
    """Autonomous async background daemon polling social APIs and writing to InfluxDB v2."""

    def __init__(
        self,
        api_url: str = DEFAULT_POLL_URL,
        keyword: str = "disinfo",
        poll_interval_seconds: float = 60.0,
        store: InfluxTimeSeriesStore | None = None,
        http_client: httpx.AsyncClient | None = None,
        infected_keywords: Sequence[str] = DEFAULT_INFECTED_KEYWORDS,
        skeptic_keywords: Sequence[str] = DEFAULT_SKEPTIC_KEYWORDS,
    ) -> None:
        """Configure the autonomous social polling daemon."""
        if poll_interval_seconds <= 0.0:
            raise ValueError("poll_interval_seconds must be > 0.")
        self.api_url: str = api_url
        self.keyword: str = keyword
        self.poll_interval_seconds: float = poll_interval_seconds
        self.store: InfluxTimeSeriesStore = (
            store if store is not None else InfluxTimeSeriesStore()
        )
        self._http_client: httpx.AsyncClient | None = http_client
        self.infected_keywords: tuple[str, ...] = tuple(infected_keywords)
        self.skeptic_keywords: tuple[str, ...] = tuple(skeptic_keywords)
        self._stop_event: asyncio.Event | None = None
        self._daemon_task: asyncio.Task[None] | None = None

    @staticmethod
    def _parse_posts_payload(payload: Any) -> list[dict[str, Any] | str]:
        """Normalize mock REST or Reddit/Telegram JSON responses into a post list."""
        if isinstance(payload, list):
            return cast(list[dict[str, Any] | str], payload)
        if isinstance(payload, dict):
            posts_val = payload.get("posts")
            if isinstance(posts_val, list):
                return cast(list[dict[str, Any] | str], posts_val)
            items_val = payload.get("items")
            if isinstance(items_val, list):
                return cast(list[dict[str, Any] | str], items_val)
            data_obj = payload.get("data")
            if isinstance(data_obj, dict) and isinstance(data_obj.get("children"), list):
                return [
                    child.get("data", child)
                    for child in data_obj["children"]
                    if isinstance(child, dict)
                ]
        return []

    def _synthetic_fallback_posts(self) -> list[dict[str, Any] | str]:
        """Generate a deterministic fallback batch when no external social endpoint is reachable."""
        return [
            {"text": f"Breaking proof about {self.keyword}, share this viral report!"},
            {"text": f"Confirmed leaked documents on {self.keyword}."},
            {"text": f"Fact-check: that {self.keyword} rumor is a fabricated hoax."},
            {"text": f"Experts debunked the misleading {self.keyword} claim."},
            {"text": f"General commentary mentioning {self.keyword}."},
        ]

    async def fetch_posts(self) -> list[dict[str, Any] | str]:
        """Asynchronously fetch social posts matching self.keyword via httpx."""
        try:
            if self._http_client is not None:
                resp = await self._http_client.get(
                    self.api_url, params={"q": self.keyword, "limit": 100}
                )
                resp.raise_for_status()
                return self._parse_posts_payload(resp.json())

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    self.api_url, params={"q": self.keyword, "limit": 100}
                )
                resp.raise_for_status()
                return self._parse_posts_payload(resp.json())
        except httpx.HTTPError as exc:
            logger.warning(
                "social_endpoint_fetch_failed",
                error=str(exc),
                keyword=self.keyword,
            )
            return self._synthetic_fallback_posts()

    async def poll_once(self, timestamp: datetime | None = None) -> dict[str, int]:
        """Fetch posts once, classify Infected vs Skeptic counts, and write to InfluxDB."""
        posts = await self.fetch_posts()
        counts = aggregate_post_counts(
            posts,
            infected_keywords=self.infected_keywords,
            skeptic_keywords=self.skeptic_keywords,
        )
        self.store.write_minute_counts(
            keyword=self.keyword,
            infected_count=counts["Infected"],
            skeptic_count=counts["Skeptic"],
            timestamp=timestamp,
        )
        return counts

    async def run_forever(
        self,
        stop_event: asyncio.Event | None = None,
        max_iterations: int | None = None,
    ) -> None:
        """Run the autonomous polling loop every poll_interval_seconds."""
        iteration = 0
        while True:
            if stop_event is not None and stop_event.is_set():
                break

            await self.poll_once()
            iteration += 1
            if max_iterations is not None and iteration >= max_iterations:
                break

            if stop_event is not None:
                try:
                    await asyncio.wait_for(
                        stop_event.wait(), timeout=self.poll_interval_seconds
                    )
                    break
                except TimeoutError:
                    continue
            else:
                await asyncio.sleep(self.poll_interval_seconds)

    def start_background_daemon(self) -> asyncio.Task[None]:
        """Start run_forever as a background asyncio.Task."""
        if self._daemon_task is not None and not self._daemon_task.done():
            return self._daemon_task
        self._stop_event = asyncio.Event()
        self._daemon_task = asyncio.create_task(
            self.run_forever(stop_event=self._stop_event)
        )
        return self._daemon_task

    async def stop_background_daemon(self) -> None:
        """Signal the background daemon to stop and await clean shutdown."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._daemon_task is not None:
            await self._daemon_task
            self._daemon_task = None
