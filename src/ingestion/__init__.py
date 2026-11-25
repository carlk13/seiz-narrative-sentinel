"""Autonomous social-metrics polling and InfluxDB v2 ingestion package."""

from src.ingestion.poller import (
    DEFAULT_INFECTED_KEYWORDS,
    DEFAULT_SKEPTIC_KEYWORDS,
    InfluxTimeSeriesStore,
    SocialMetricsPoller,
    aggregate_post_counts,
    classify_post,
    floor_to_minute,
)

__all__ = [
    "DEFAULT_INFECTED_KEYWORDS",
    "DEFAULT_SKEPTIC_KEYWORDS",
    "InfluxTimeSeriesStore",
    "SocialMetricsPoller",
    "aggregate_post_counts",
    "classify_post",
    "floor_to_minute",
]
