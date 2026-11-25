"""Live worldwide misinformation connectors (GDELT, Bluesky, Reddit) and geo-aggregation."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
import structlog

from src.ingestion.poller import DEFAULT_INFECTED_KEYWORDS, DEFAULT_SKEPTIC_KEYWORDS
from src.simulation.engine import configure_structlog

configure_structlog()
logger = structlog.get_logger(__name__)

GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
BLUESKY_SEARCH_API_URL = "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts"
REDDIT_SEARCH_API_URL = "https://www.reddit.com/search.json"

SupportedSource = Literal["gdelt", "bluesky", "reddit"]
ALL_GLOBAL_SOURCES: tuple[SupportedSource, ...] = ("gdelt", "bluesky", "reddit")

MULTILINGUAL_INFECTED_KEYWORDS: tuple[str, ...] = DEFAULT_INFECTED_KEYWORDS + (
    # Spanish
    "urgente",
    "filtrado",
    "encubrimiento",
    "comparte",
    "verdad oculta",
    # French
    "scandale",
    "complot",
    "révélation",
    "partagez",
    # German
    "enthüllt",
    "vertuschung",
    "eilmeldung",
    "teilen",
    # Portuguese
    "vazado",
    "compartilhe",
    "farsa revelada",
    # Arabic (transliterated/script common viral cues)
    "عاجل",
    "فضيحة",
    "مسرب",
)

MULTILINGUAL_SKEPTIC_KEYWORDS: tuple[str, ...] = DEFAULT_SKEPTIC_KEYWORDS + (
    # Spanish
    "bulo",
    "desmentido",
    "falso",
    "verificación",
    "engañoso",
    # French
    "infox",
    "intox",
    "démenti",
    "vérification",
    "faux",
    # German
    "faktencheck",
    "falschmeldung",
    "widerlegt",
    "irreführend",
    "gerücht",
    # Portuguese
    "boato",
    "checagem",
    "desmentido",
    "enganoso",
    # Arabic
    "مضلل",
    "إشاعة",
    "تدقيق الحقائق",
    "كاذب",
)

FACT_CHECK_DOMAINS: tuple[str, ...] = (
    "snopes.com",
    "politifact.com",
    "factcheck.org",
    "fullfact.org",
    "correctiv.org",
    "maldita.es",
    "chequeado.com",
    "factcheck.afp.com",
    "reuters.com",
    "apnews.com",
    "leadstories.com",
    "boomlive.in",
    "africacheck.org",
)

COUNTRY_NAME_TO_ISO3: dict[str, str] = {
    "united states": "USA",
    "usa": "USA",
    "us": "USA",
    "united kingdom": "GBR",
    "uk": "GBR",
    "britain": "GBR",
    "england": "GBR",
    "germany": "DEU",
    "deutschland": "DEU",
    "france": "FRA",
    "spain": "ESP",
    "italy": "ITA",
    "netherlands": "NLD",
    "poland": "POL",
    "ukraine": "UKR",
    "russia": "RUS",
    "canada": "CAN",
    "mexico": "MEX",
    "brazil": "BRA",
    "argentina": "ARG",
    "colombia": "COL",
    "chile": "CHL",
    "india": "IND",
    "china": "CHN",
    "japan": "JPN",
    "south korea": "KOR",
    "australia": "AUS",
    "new zealand": "NZL",
    "south africa": "ZAF",
    "nigeria": "NGA",
    "kenya": "KEN",
    "egypt": "EGY",
    "israel": "ISR",
    "turkey": "TUR",
    "saudi arabia": "SAU",
    "indonesia": "IDN",
    "philippines": "PHL",
    "pakistan": "PAK",
    "sweden": "SWE",
    "norway": "NOR",
    "switzerland": "CHE",
    "austria": "AUT",
    "belgium": "BEL",
    "ireland": "IRL",
    "portugal": "PRT",
    "greece": "GRC",
    "taiwan": "TWN",
    "iran": "IRN",
}

LANGUAGE_FALLBACK_ISO3: dict[str, str] = {
    "en": "USA",
    "english": "USA",
    "de": "DEU",
    "german": "DEU",
    "fr": "FRA",
    "french": "FRA",
    "es": "ESP",
    "spanish": "ESP",
    "pt": "BRA",
    "portuguese": "BRA",
    "it": "ITA",
    "italian": "ITA",
    "ja": "JPN",
    "japanese": "JPN",
    "ko": "KOR",
    "korean": "KOR",
    "pl": "POL",
    "polish": "POL",
    "nl": "NLD",
    "dutch": "NLD",
    "ru": "RUS",
    "russian": "RUS",
    "uk": "UKR",
    "ukrainian": "UKR",
    "hi": "IND",
    "hindi": "IND",
    "ar": "EGY",
    "arabic": "EGY",
}


@dataclass(frozen=True)
class GlobalMention:
    """Normalized global social or news mention returned by public connectors."""

    source: SupportedSource
    title: str
    url: str
    published_at: datetime
    country_iso3: str
    country_name: str
    language: str
    label: Literal["Infected", "Skeptic", "Neutral"]


def normalize_country_iso3(
    country_raw: str | None = None,
    language: str | None = None,
    text: str = "",
) -> tuple[str, str]:
    """Resolve an ISO-3 country code and display name from metadata or text cues."""
    if country_raw:
        cleaned = country_raw.strip()
        lower = cleaned.lower()
        if len(cleaned) == 3 and cleaned.isalpha() and cleaned.isupper():
            return cleaned, cleaned
        if lower in COUNTRY_NAME_TO_ISO3:
            return COUNTRY_NAME_TO_ISO3[lower], cleaned

    lowered_text = text.lower()
    for name, iso3 in COUNTRY_NAME_TO_ISO3.items():
        if len(name) >= 4 and name in lowered_text:
            return iso3, name.title()

    if language:
        lang_key = language.strip().lower().split("-")[0]
        if lang_key in LANGUAGE_FALLBACK_ISO3:
            iso = LANGUAGE_FALLBACK_ISO3[lang_key]
            return iso, iso

    return "USA", "Global / Unattributed"


def classify_post_multilingual(
    text: str,
    url: str = "",
    infected_keywords: Sequence[str] = MULTILINGUAL_INFECTED_KEYWORDS,
    skeptic_keywords: Sequence[str] = MULTILINGUAL_SKEPTIC_KEYWORDS,
) -> Literal["Infected", "Skeptic", "Neutral"]:
    """Deterministic multilingual stance classifier with fact-check domain awareness."""
    lowered = text.lower()
    host = urlparse(url).netloc.lower() if url else ""

    if host and any(domain in host for domain in FACT_CHECK_DOMAINS):
        if any(kw in lowered for kw in ("fact", "check", "false", "debunk", "hoax", "fake")):
            return "Skeptic"

    infected_hits = sum(1 for kw in infected_keywords if kw.lower() in lowered)
    skeptic_hits = sum(1 for kw in skeptic_keywords if kw.lower() in lowered)

    if host and any(domain in host for domain in FACT_CHECK_DOMAINS[:7]):
        skeptic_hits += 2

    if skeptic_hits > infected_hits:
        return "Skeptic"
    if infected_hits > skeptic_hits:
        return "Infected"
    if infected_hits > 0 and infected_hits == skeptic_hits:
        return "Skeptic"
    return "Neutral"


def _parse_gdelt_seendate(raw: str | None) -> datetime:
    """Parse GDELT seendate (YYYYMMDDTHHMMSSZ or YYYYMMDDHHMMSS) into UTC datetime."""
    if not raw:
        return datetime.now(UTC)
    cleaned = raw.strip()
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return datetime.now(UTC)


def _parse_iso_datetime(raw: str | None) -> datetime:
    """Parse ISO-8601 timestamp safely into UTC datetime."""
    if not raw:
        return datetime.now(UTC)
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
    except ValueError:
        return datetime.now(UTC)


async def fetch_gdelt_mentions(
    query: str,
    max_records: int = 50,
    timespan: str = "7d",
    http_client: httpx.AsyncClient | None = None,
) -> list[GlobalMention]:
    """Fetch multilingual, country-tagged global news mentions from the GDELT 2.0 Doc API."""
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": str(max(1, min(max_records, 250))),
        "timespan": timespan,
        "format": "json",
        "sort": "DateDesc",
    }
    try:
        if http_client is not None:
            resp = await http_client.get(GDELT_DOC_API_URL, params=params)
        else:
            async with httpx.AsyncClient(timeout=12.0) as client:
                resp = await client.get(GDELT_DOC_API_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("gdelt_fetch_failed", query=query, error=str(exc))
        return []

    articles = data.get("articles") if isinstance(data, dict) else None
    if not isinstance(articles, list):
        return []

    mentions: list[GlobalMention] = []
    for art in articles:
        if not isinstance(art, dict):
            continue
        title = str(art.get("title") or "").strip()
        if not title:
            continue
        url = str(art.get("url") or "")
        lang = str(art.get("language") or "en")
        country_raw = str(art.get("sourcecountry") or "")
        iso3, cname = normalize_country_iso3(country_raw, lang, title)
        label = classify_post_multilingual(title, url=url)
        mentions.append(
            GlobalMention(
                source="gdelt",
                title=title,
                url=url,
                published_at=_parse_gdelt_seendate(art.get("seendate")),
                country_iso3=iso3,
                country_name=cname,
                language=lang,
                label=label,
            )
        )
    return mentions


async def fetch_bluesky_mentions(
    query: str,
    max_records: int = 40,
    http_client: httpx.AsyncClient | None = None,
) -> list[GlobalMention]:
    """Fetch live public social posts from the Bluesky AppView public search API."""
    params = {
        "q": query,
        "limit": str(max(1, min(max_records, 100))),
        "sort": "latest",
    }
    try:
        if http_client is not None:
            resp = await http_client.get(BLUESKY_SEARCH_API_URL, params=params)
        else:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(BLUESKY_SEARCH_API_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("bluesky_fetch_failed", query=query, error=str(exc))
        return []

    posts = data.get("posts") if isinstance(data, dict) else None
    if not isinstance(posts, list):
        return []

    mentions: list[GlobalMention] = []
    for post in posts:
        if not isinstance(post, dict):
            continue
        record = post.get("record")
        if not isinstance(record, dict):
            continue
        text = str(record.get("text") or "").strip()
        if not text:
            continue
        langs = record.get("langs")
        lang = str(langs[0]) if isinstance(langs, list) and langs else "en"
        uri = str(post.get("uri") or "")
        author = post.get("author")
        handle = str(author.get("handle") or "") if isinstance(author, dict) else ""
        post_id = uri.rsplit("/", 1)[-1] if "/" in uri else ""
        web_url = f"https://bsky.app/profile/{handle}/post/{post_id}" if handle and post_id else ""
        iso3, cname = normalize_country_iso3(None, lang, text)
        label = classify_post_multilingual(text, url=web_url)
        mentions.append(
            GlobalMention(
                source="bluesky",
                title=text[:240],
                url=web_url,
                published_at=_parse_iso_datetime(record.get("createdAt")),
                country_iso3=iso3,
                country_name=cname,
                language=lang,
                label=label,
            )
        )
    return mentions


async def fetch_reddit_mentions(
    query: str,
    max_records: int = 40,
    http_client: httpx.AsyncClient | None = None,
) -> list[GlobalMention]:
    """Fetch public social discussions from Reddit's JSON search endpoint."""
    params = {
        "q": query,
        "limit": str(max(1, min(max_records, 100))),
        "sort": "new",
        "t": "week",
    }
    headers = {"User-Agent": "SEIZ-Disinfo-Tracker/0.1 (research-analytics)"}
    try:
        if http_client is not None:
            resp = await http_client.get(REDDIT_SEARCH_API_URL, params=params, headers=headers)
        else:
            async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
                resp = await client.get(REDDIT_SEARCH_API_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("reddit_fetch_failed", query=query, error=str(exc))
        return []

    root = data.get("data") if isinstance(data, dict) else None
    children = root.get("children") if isinstance(root, dict) else None
    if not isinstance(children, list):
        return []

    mentions: list[GlobalMention] = []
    for child in children:
        if not isinstance(child, dict):
            continue
        item = child.get("data")
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        selftext = str(item.get("selftext") or "").strip()
        full_text = f"{title} {selftext}".strip()
        if not title:
            continue
        permalink = str(item.get("permalink") or "")
        url = f"https://www.reddit.com{permalink}" if permalink else str(item.get("url") or "")
        created_utc = item.get("created_utc")
        published = (
            datetime.fromtimestamp(float(created_utc), tz=UTC)
            if isinstance(created_utc, (int, float))
            else datetime.now(UTC)
        )
        iso3, cname = normalize_country_iso3(None, "en", full_text)
        label = classify_post_multilingual(full_text, url=url)
        mentions.append(
            GlobalMention(
                source="reddit",
                title=title[:240],
                url=url,
                published_at=published,
                country_iso3=iso3,
                country_name=cname,
                language="en",
                label=label,
            )
        )
    return mentions


def build_observed_timeseries(
    mentions: Sequence[GlobalMention],
    num_buckets: int = 12,
) -> tuple[list[str], list[float], list[float], list[float]]:
    """Bucket timestamped global mentions into cumulative (t_hours, I, Z) curves for SEIZ."""
    if not mentions:
        now = datetime.now(UTC)
        t0 = now - timedelta(hours=1)
        return ([t0.isoformat(), now.isoformat()], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0])

    ordered = sorted(mentions, key=lambda m: m.published_at)
    start_ts = ordered[0].published_at
    end_ts = ordered[-1].published_at
    span_seconds = max((end_ts - start_ts).total_seconds(), 3600.0)
    buckets = max(2, min(num_buckets, 48))
    step_seconds = span_seconds / (buckets - 1)

    inf_counts = [0.0] * buckets
    skp_counts = [0.0] * buckets

    for mention in ordered:
        offset = max(0.0, (mention.published_at - start_ts).total_seconds())
        idx = min(buckets - 1, int(offset / step_seconds))
        if mention.label == "Infected":
            inf_counts[idx] += 1.0
        elif mention.label == "Skeptic":
            skp_counts[idx] += 1.0

    timestamps: list[str] = []
    t_hours: list[float] = []
    cum_i: list[float] = []
    cum_z: list[float] = []
    run_i = 0.0
    run_z = 0.0

    for idx in range(buckets):
        ts = start_ts + timedelta(seconds=idx * step_seconds)
        timestamps.append(ts.isoformat())
        t_hours.append(round((idx * step_seconds) / 3600.0, 4))
        run_i += inf_counts[idx]
        run_z += skp_counts[idx]
        cum_i.append(run_i)
        cum_z.append(run_z)

    return timestamps, t_hours, cum_i, cum_z


async def scan_global_narrative(
    query: str,
    sources: Sequence[SupportedSource] = ALL_GLOBAL_SOURCES,
    max_per_source: int = 40,
    timespan: str = "7d",
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Concurrently scan public global feeds, aggregate by country, and build SEIZ series."""
    active_sources = tuple(dict.fromkeys(sources)) or ALL_GLOBAL_SOURCES
    tasks: list[Any] = []
    for src in active_sources:
        if src == "gdelt":
            tasks.append(
                fetch_gdelt_mentions(
                    query,
                    max_records=max_per_source,
                    timespan=timespan,
                    http_client=http_client,
                )
            )
        elif src == "bluesky":
            tasks.append(
                fetch_bluesky_mentions(
                    query,
                    max_records=max_per_source,
                    http_client=http_client,
                )
            )
        elif src == "reddit":
            tasks.append(
                fetch_reddit_mentions(
                    query,
                    max_records=max_per_source,
                    http_client=http_client,
                )
            )

    batches = await asyncio.gather(*tasks)
    all_mentions: list[GlobalMention] = []
    source_counts: dict[str, int] = {}
    for src, batch in zip(active_sources, batches, strict=True):
        source_counts[src] = len(batch)
        all_mentions.extend(batch)

    all_mentions.sort(key=lambda m: m.published_at, reverse=True)

    stance_counts = {"Infected": 0, "Skeptic": 0, "Neutral": 0}
    country_stats_map: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"country_name": "", "total": 0, "infected": 0, "skeptic": 0, "neutral": 0}
    )

    for m in all_mentions:
        stance_counts[m.label] += 1
        c_entry = country_stats_map[m.country_iso3]
        if not c_entry["country_name"]:
            c_entry["country_name"] = m.country_name
        c_entry["total"] += 1
        if m.label == "Infected":
            c_entry["infected"] += 1
        elif m.label == "Skeptic":
            c_entry["skeptic"] += 1
        else:
            c_entry["neutral"] += 1

    country_stats = [
        {
            "iso3": iso3,
            "country_name": stats["country_name"],
            "total": int(stats["total"]),
            "infected": int(stats["infected"]),
            "skeptic": int(stats["skeptic"]),
            "neutral": int(stats["neutral"]),
            "infected_share_pct": round(
                (100.0 * stats["infected"] / stats["total"]) if stats["total"] > 0 else 0.0,
                2,
            ),
        }
        for iso3, stats in sorted(
            country_stats_map.items(),
            key=lambda kv: kv[1]["total"],
            reverse=True,
        )
    ]

    timestamps, t_obs, i_obs, z_obs = build_observed_timeseries(all_mentions)

    return {
        "query": query,
        "sources_queried": list(active_sources),
        "source_counts": source_counts,
        "total_mentions": len(all_mentions),
        "stance_counts": stance_counts,
        "country_stats": country_stats,
        "timestamps": timestamps,
        "t_observed": t_obs,
        "I_observed": i_obs,
        "Z_observed": z_obs,
        "mentions": [
            {
                "source": m.source,
                "title": m.title,
                "url": m.url,
                "published_at": m.published_at.isoformat(),
                "country_iso3": m.country_iso3,
                "country_name": m.country_name,
                "language": m.language,
                "label": m.label,
            }
            for m in all_mentions[:100]
        ],
    }
