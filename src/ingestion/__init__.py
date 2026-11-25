"""Autonomous social-metrics polling, global connectors, and InfluxDB v2 ingestion package."""

from src.ingestion.global_tracker import (
    ALL_GLOBAL_SOURCES,
    GlobalMention,
    build_observed_timeseries,
    classify_post_multilingual,
    fetch_bluesky_mentions,
    fetch_gdelt_mentions,
    fetch_reddit_mentions,
    normalize_country_iso3,
    scan_global_narrative,
)
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
    "ALL_GLOBAL_SOURCES",
    "DEFAULT_INFECTED_KEYWORDS",
    "DEFAULT_SKEPTIC_KEYWORDS",
    "GlobalMention",
    "InfluxTimeSeriesStore",
    "SocialMetricsPoller",
    "aggregate_post_counts",
    "build_observed_timeseries",
    "classify_post",
    "classify_post_multilingual",
    "fetch_bluesky_mentions",
    "fetch_gdelt_mentions",
    "fetch_reddit_mentions",
    "floor_to_minute",
    "normalize_country_iso3",
    "scan_global_narrative",
]
