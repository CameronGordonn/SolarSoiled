"""Open-Meteo historical weather + air quality client with on-disk caching.

Open-Meteo is free with no API key and exposes ERA5 reanalysis back to 1940 plus
CAMS air quality (PM2.5/PM10) going back several years. Daily aggregates are
sufficient for soiling-risk rolling-window features.

MERRA-2 (NASA) extends AQ coverage back to 1980. Set the environment variable
NASA_EARTHDATA_TOKEN to enable it. When set, MERRA-2 replaces CAMS as the AQ
source for the full date range (no 2013 floor). Requires approving the
"NASA GESDISC DATA ARCHIVE" app in your Earthdata profile.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# ── MERRA-2 constants ────────────────────────────────────────────────────────
# GES DISC OPeNDAP base for the 2D hourly aerosol dataset (M2T1NXAER).
MERRA2_BASE = "https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2T1NXAER.5.12.4"

# MERRA-2 grid (fixed 0.5° lat × 0.625° lon).
_M2_LAT_STEP = 0.5
_M2_LON_STEP = 0.625
_M2_LAT_MIN = -90.0
_M2_LON_MIN = -180.0
_M2_N_LON = 576  # wrap-around guard

# Aerosol variables to fetch (all in kg/m³; ×1e9 → µg/m³).
# PM2.5 = dust25 + BC + OC + SO4 + seasalt25
# PM10  = dust_total + seasalt_total + BC + OC + SO4
_M2_VARS = ("DUSMASS25", "BCSMASS", "OCSMASS", "SO4SMASS", "SSSMASS25", "DUSMASS", "SSSMASS")

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


def _merra2_version(year: int) -> str:
    """MERRA-2 stream label embedded in filenames."""
    if year >= 2011:
        return "400"
    if year >= 2001:
        return "300"
    if year >= 1992:
        return "200"
    return "100"


def _merra2_lat_idx(lat: float) -> int:
    return round((lat - _M2_LAT_MIN) / _M2_LAT_STEP)


def _merra2_lon_idx(lon: float) -> int:
    return int(round((lon - _M2_LON_MIN) / _M2_LON_STEP)) % _M2_N_LON


def _parse_merra2_ascii(text: str) -> dict[str, np.ndarray]:
    """Parse OPeNDAP ASCII response into {varname: array of 24 hourly floats}.

    Actual format returned by GES DISC OPeNDAP (one value per line):
      BCSMASS.BCSMASS[BCSMASS.time=0][BCSMASS.lat=32], 1.55524e-10
      BCSMASS.BCSMASS[BCSMASS.time=60][BCSMASS.lat=32], 1.83888e-10
      ...
    """
    result: dict[str, list[float]] = {}
    # Match lines of the form "VARNAME.VARNAME[...], value"
    _data_re = re.compile(r'^([A-Z0-9]+)\.\1\[.*\],\s*([-\d.eE+naN]+)\s*$')
    for line in text.splitlines():
        m = _data_re.match(line.strip())
        if not m:
            continue
        varname = m.group(1)
        try:
            val = float(m.group(2))
        except ValueError:
            val = float("nan")
        result.setdefault(varname, []).append(val)
    return {k: np.array(v) for k, v in result.items()}


def _merra2_session(token: str, cache_dir: Path):
    """CachedSession with Earthdata Bearer token for MERRA-2 requests.

    Uses a custom session subclass that preserves the Authorization header
    across NASA's cross-host redirects (GES DISC → urs.earthdata.nasa.gov).
    """
    try:
        from requests_cache import CachedSession
    except ImportError as err:
        raise ImportError("requests-cache is required: pip install requests-cache") from err

    class _EarthdataCachedSession(CachedSession):
        def __init__(self, _token: str, **kwargs):
            super().__init__(**kwargs)
            self._token = _token
            self.headers.update({"Authorization": f"Bearer {_token}"})

        def rebuild_auth(self, prepared_request, response):  # noqa: ARG002
            prepared_request.headers["Authorization"] = f"Bearer {self._token}"

    cache_dir.mkdir(parents=True, exist_ok=True)
    return _EarthdataCachedSession(
        _token=token,
        cache_name=str(cache_dir / "merra2"),
        backend="sqlite",
        expire_after=timedelta(days=365),  # reanalysis data is immutable
    )


def _fetch_merra2_day(
    sess,
    lat: float,
    lon: float,
    d: date,
) -> dict[str, float]:
    """Fetch daily-mean PM2.5 and PM10 for one day from MERRA-2 OPeNDAP."""
    year, month = d.year, d.month
    date_str = d.strftime("%Y%m%d")
    version = _merra2_version(year)
    lat_idx = _merra2_lat_idx(lat)
    lon_idx = _merra2_lon_idx(lon)

    filename = f"MERRA2_{version}.tavg1_2d_aer_Nx.{date_str}.nc4"
    constraint = ",".join(f"{v}[0:23][{lat_idx}][{lon_idx}]" for v in _M2_VARS)
    url = f"{MERRA2_BASE}/{year}/{month:02d}/{filename}.ascii?{constraint}"

    for attempt in range(MAX_RETRIES + 1):
        resp = sess.get(url, timeout=60)
        if resp.status_code == 429:
            if attempt >= MAX_RETRIES:
                resp.raise_for_status()
            time.sleep(BASE_BACKOFF_S * (2 ** attempt))
            continue
        resp.raise_for_status()
        break

    arrays = _parse_merra2_ascii(resp.text)

    def _mean(varname: str) -> float:
        return float(np.nanmean(arrays[varname])) if varname in arrays else 0.0

    # All values in kg/m³ → µg/m³
    scale = 1e9
    pm25 = (_mean("DUSMASS25") + _mean("BCSMASS") + _mean("OCSMASS")
            + _mean("SO4SMASS") + _mean("SSSMASS25")) * scale
    pm10 = (_mean("DUSMASS") + _mean("SSSMASS") + _mean("BCSMASS")
            + _mean("OCSMASS") + _mean("SO4SMASS")) * scale
    return {"pm2_5": pm25, "pm10": pm10}


def fetch_air_quality_merra2(
    lat: float,
    lon: float,
    start: date,
    end: date,
    token: str,
    cache_dir: Path = Path(".cache/soiling"),
) -> pd.DataFrame:
    """Fetch daily PM2.5 and PM10 from MERRA-2 (1980-present).

    Requires a NASA Earthdata Bearer token. Coverage has no floor date unlike
    CAMS (which starts 2013-01-01), making it suitable for historical NREL rows.
    """
    sess = _merra2_session(token, Path(cache_dir))
    rows = []
    current = start
    while current <= end:
        try:
            row = _fetch_merra2_day(sess, lat, lon, current)
            row["time"] = pd.Timestamp(current)
            rows.append(row)
        except Exception as exc:
            logger.warning("MERRA-2 fetch failed for %s at (%.3f, %.3f): %s", current, lat, lon, exc)
            rows.append({"time": pd.Timestamp(current), "pm2_5": float("nan"), "pm10": float("nan")})
        current += timedelta(days=1)

    if not rows:
        return pd.DataFrame(columns=["pm2_5", "pm10"])
    df = pd.DataFrame(rows).set_index("time").sort_index()
    return df


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
    """Convenience: weather + AQ joined on the daily index, NaNs where AQ missing.

    AQ source priority:
      1. MERRA-2 (1980-present) if NASA_EARTHDATA_TOKEN is set in the environment.
         Consistent source for all rows regardless of date — avoids mixing MERRA-2
         and CAMS aerosol models within the same training set.
      2. CAMS via Open-Meteo (2013-present) otherwise. Pre-2013 rows return NaN
         for PM columns (median-imputed downstream).
    """
    w = fetch_weather(lat, lon, start, end, cache_dir=cache_dir)
    token = os.environ.get("NASA_EARTHDATA_TOKEN")
    try:
        if token:
            aq = fetch_air_quality_merra2(lat, lon, start, end, token=token, cache_dir=cache_dir)
        else:
            aq = fetch_air_quality(lat, lon, start, end, cache_dir=cache_dir)
    except Exception as exc:
        source = "MERRA-2" if token else "CAMS"
        logger.warning("%s AQ fetch failed for (%.4f, %.4f): %s — continuing without it", source, lat, lon, exc)
        aq = pd.DataFrame(index=w.index)
    return w.join(aq, how="left")
