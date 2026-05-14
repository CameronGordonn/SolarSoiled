"""Open-Meteo historical weather + air quality client with on-disk caching.

Open-Meteo is free with no API key and exposes ERA5 reanalysis back to 1940 plus
CAMS air quality (PM2.5/PM10) going back several years. Daily aggregates are
sufficient for soiling-risk rolling-window features.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Open-Meteo's free tier uses a sliding window (by minute and by day). When we
# hit the minute window, a 429 is served — the right response is to back off
# and retry, not drop the row. Without this, large panel runs lose ~30% of
# rows to transient limits. Daily-quota exhaustion still fails the row after
# MAX_RETRIES so we don't block forever.
MAX_RETRIES = 1
BASE_BACKOFF_S = 5.0

DEFAULT_DAILY_VARS: tuple[str, ...] = (
    "precipitation_sum",
    "wind_speed_10m_max",
    "relative_humidity_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "shortwave_radiation_sum",
)

DEFAULT_AQ_VARS: tuple[str, ...] = ("pm2_5", "pm10")

# Open-Meteo's CAMS air-quality reanalysis starts hard at 2013-01-01. Requests
# with start_date < this floor 400 with "Parameter 'start_date' is out of
# allowed range". We clamp transparently so a 365d lookback for panel-year
# 2013 (which would ask for 2012-12-31) still works — we lose at most one day
# of AQ history vs the requested window.
AQ_MIN_DATE = date(2013, 1, 1)


def _session(cache_dir: Path, expire_after_days: int = 30):
    """Build a requests-cache Session so repeated lat/lon queries hit disk."""
    try:
        from requests_cache import CachedSession
    except ImportError as err:
        raise ImportError(
            "requests-cache is required. Install via `pip install requests-cache`."
        ) from err
    cache_dir.mkdir(parents=True, exist_ok=True)
    return CachedSession(
        cache_name=str(cache_dir / "openmeteo"),
        backend="sqlite",
        expire_after=timedelta(days=expire_after_days),
    )


def _get_with_retry(sess, url: str, params: dict, timeout: int):
    """GET with exponential backoff on HTTP 429. Cached responses skip the wait."""
    for attempt in range(MAX_RETRIES + 1):
        resp = sess.get(url, params=params, timeout=timeout)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        if attempt >= MAX_RETRIES:
            resp.raise_for_status()  # raises
        wait = BASE_BACKOFF_S * (2 ** attempt)
        logger.info("Open-Meteo 429 on %s; retrying in %.0fs (attempt %d)", url.split("/")[-1], wait, attempt + 1)
        time.sleep(wait)
    return resp  # unreachable


def fetch_weather(
    lat: float,
    lon: float,
    start: date,
    end: date,
    daily_vars: Sequence[str] = DEFAULT_DAILY_VARS,
    cache_dir: Path = Path(".cache/soiling"),
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch daily ERA5 reanalysis for (lat, lon) between start and end dates.

    Returns a DataFrame indexed by date with one column per requested variable.
    """
    sess = _session(Path(cache_dir))
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(daily_vars),
        "timezone": "UTC",
    }
    resp = _get_with_retry(sess, ARCHIVE_URL, params, timeout)
    payload = resp.json()
    if "daily" not in payload:
        raise RuntimeError(f"Open-Meteo returned no daily block: {payload}")
    df = pd.DataFrame(payload["daily"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time").sort_index()


def fetch_air_quality(
    lat: float,
    lon: float,
    start: date,
    end: date,
    aq_vars: Sequence[str] = DEFAULT_AQ_VARS,
    cache_dir: Path = Path(".cache/soiling"),
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch hourly CAMS air quality and downsample to daily means.

    PM2.5 / PM10 are the dust-deposition proxies most correlated with soiling.
    """
    aq_start = max(start, AQ_MIN_DATE)
    if aq_start > end:
        return pd.DataFrame()
    sess = _session(Path(cache_dir))
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": aq_start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(aq_vars),
        "timezone": "UTC",
    }
    resp = _get_with_retry(sess, AIR_QUALITY_URL, params, timeout)
    payload = resp.json()
    if "hourly" not in payload:
        raise RuntimeError(f"Open-Meteo AQ returned no hourly block: {payload}")
    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    return df.resample("D").mean()


def fetch_combined(
    lat: float,
    lon: float,
    start: date,
    end: date,
    cache_dir: Path = Path(".cache/soiling"),
) -> pd.DataFrame:
    """Convenience: weather + AQ joined on the daily index, NaNs where AQ missing."""
    w = fetch_weather(lat, lon, start, end, cache_dir=cache_dir)
    try:
        aq = fetch_air_quality(lat, lon, start, end, cache_dir=cache_dir)
    except Exception as exc:
        logger.warning("AQ fetch failed for (%.4f, %.4f): %s — continuing without it", lat, lon, exc)
        aq = pd.DataFrame(index=w.index)
    return w.join(aq, how="left")
