"""Label sources for the soiling risk model.

Three paths, in priority order:

1. `load_nrel_soiling_map` — ingest the NREL PV Soiling Map tabular release
   (Micheli/Deceglie/Muller; 255 US stations with IWSR). Expected schema once
   the CSV is placed under data/external/nrel_soiling_map.csv:
     station_id, latitude, longitude, iwsr, soiling_rate_pct_per_day,
     start_date, end_date
   The file is not redistributable here — request from NREL or extract from
   the published supplementary of Micheli 2019 (Prog. in Photovoltaics).

2. `somosclean_synthetic_labels` — physics labels from the ENEL SOMOSclean
   empirical model (Micheli et al.). Models soiling as complementary exponential
   growth toward a saturation ceiling, with PM10-driven acceleration on dust days
   and partial cleaning proportional to rainfall. Validated on 200 MW of Spanish
   PV plants (MAE 0.71%, below sensor noise floor). Preferred over Kimber.

3. `kimber_synthetic_labels` — simpler physics-proxy labels from Kimber 2007
   (linear PM2.5 accumulation, binary rain reset). Available as an ablation
   baseline; SOMOSclean is the production path.

All paths return a DataFrame with columns:
  station_id, latitude, longitude, as_of, iwsr, label
where `label` is 1 if IWSR < `iwsr_risk_threshold` else 0.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.soiling.weather_client import fetch_combined

logger = logging.getLogger(__name__)


def load_nrel_soiling_map(path: Path, bbox: tuple[float, float, float, float] | None = None) -> pd.DataFrame:
    """Load NREL soiling-map CSV, optionally clipped to a (minx, miny, maxx, maxy) bbox."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"NREL soiling map not found at {path}. "
            "Request from https://www.nrel.gov/pv/soiling or use kimber_synthetic_labels()."
        )
    df = pd.read_csv(path)
    required = {"station_id", "latitude", "longitude", "iwsr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"NREL CSV missing columns: {missing}")
    if bbox is not None:
        minx, miny, maxx, maxy = bbox
        df = df[
            df["longitude"].between(minx, maxx) & df["latitude"].between(miny, maxy)
        ].reset_index(drop=True)
    return df


def load_nrel_panel(path: Path, bbox: tuple[float, float, float, float] | None = None) -> pd.DataFrame:
    """Load the per-(station, year) panel CSV produced by `convert_panel()`.

    Same schema as the summary CSV plus a required `year` integer column.
    Each row represents one annual IWSR observation, enabling per-year weather
    feature matching instead of comparing all stations against a single
    today-centered weather window.
    """
    df = load_nrel_soiling_map(path, bbox=bbox)
    if "year" not in df.columns:
        raise ValueError(f"Expected panel CSV with `year` column at {path}; got summary CSV instead")
    df["year"] = df["year"].astype(int)
    return df


def somosclean_eqd_trajectory(
    daily: pd.DataFrame,
    sl_sat: float = 0.25,
    k: float = 30.0,
    heavy_rain_mm: float = 10.0,
    rain_min_mm: float = 1.0,
    pm10_dust_threshold: float = 50.0,
    pm10_dust_scale: float = 0.02,
) -> tuple[pd.Series, pd.Series]:
    """SOMOSclean (ENEL) daily equivalent-day and soiling-loss trajectories.

    eqD accumulates faster on dust days (PM10 > threshold) and resets fully or
    partially on rain. SL is the complementary exponential:
        SL(d) = sl_sat * (1 - exp(-eqD(d) / k))
        IWSR(d) = 1 - SL(d)

    Returns (eqd_series, sl_series), both indexed by daily.index.
    """
    precip = daily.get("precipitation_sum")
    pm10 = daily.get("pm10")
    if precip is None:
        raise ValueError("SOMOSclean model needs precipitation_sum column")

    precip_v = pd.to_numeric(precip.fillna(0.0), errors="coerce").fillna(0.0).values
    if pm10 is not None:
        pm10_v = pd.to_numeric(pm10.fillna(pm10.median() if not pm10.isna().all() else 0.0), errors="coerce").fillna(0.0).values
    else:
        pm10_v = np.zeros(len(daily))

    eqd = np.zeros(len(daily))
    sl = np.zeros(len(daily))
    eq = 0.0
    for i in range(len(daily)):
        p = precip_v[i]
        if p >= heavy_rain_mm:
            f = 0.0
        elif p >= rain_min_mm:
            # Linear partial cleaning between rain_min_mm and heavy_rain_mm
            f = 1.0 - (p - rain_min_mm) / (heavy_rain_mm - rain_min_mm)
        elif pm10_v[i] > pm10_dust_threshold:
            f = 1.0 + pm10_dust_scale * (pm10_v[i] - pm10_dust_threshold)
        else:
            f = 1.0
        eq = f * (eq + 1.0)
        eqd[i] = eq
        sl[i] = sl_sat * (1.0 - np.exp(-eq / k))

    return (
        pd.Series(eqd, index=daily.index, name="eqD"),
        pd.Series(sl, index=daily.index, name="soiling_loss"),
    )


def somosclean_synthetic_labels(
    stations: Iterable[dict],
    as_of: date,
    lookback_days: int = 365,
    iwsr_risk_threshold: float = 0.90,
    cache_dir: Path = Path(".cache/soiling"),
    sl_sat: float = 0.25,
    k: float = 30.0,
    heavy_rain_mm: float = 10.0,
    rain_min_mm: float = 1.0,
    pm10_dust_threshold: float = 50.0,
    pm10_dust_scale: float = 0.02,
) -> pd.DataFrame:
    """Generate SOMOSclean-based labels for a list of stations.

    IWSR is derived from the terminal soiling loss of the SOMOSclean trajectory:
        IWSR = 1 - SL_at_as_of
    A longer lookback (365d default vs Kimber's 180d) lets the saturation
    dynamics stabilize before the label window ends.

    Each station dict needs keys: station_id, latitude, longitude.
    """
    start = as_of - timedelta(days=lookback_days)
    rows = []
    for s in stations:
        try:
            daily = fetch_combined(s["latitude"], s["longitude"], start, as_of, cache_dir=cache_dir)
        except Exception as exc:
            logger.warning("Skipping %s: weather fetch failed (%s)", s["station_id"], exc)
            continue
        _, sl_series = somosclean_eqd_trajectory(
            daily,
            sl_sat=sl_sat,
            k=k,
            heavy_rain_mm=heavy_rain_mm,
            rain_min_mm=rain_min_mm,
            pm10_dust_threshold=pm10_dust_threshold,
            pm10_dust_scale=pm10_dust_scale,
        )
        iwsr = float(1.0 - sl_series.iloc[-1])
        rows.append(
            {
                "station_id": s["station_id"],
                "latitude": s["latitude"],
                "longitude": s["longitude"],
                "as_of": as_of.isoformat(),
                "iwsr": iwsr,
                "label": int(iwsr < iwsr_risk_threshold),
            }
        )
    return pd.DataFrame(rows)


def somosclean_panel_labels(
    nrel_csv_path: Path,
    iwsr_risk_threshold: float = 0.944,
    cache_dir: Path = Path(".cache/soiling"),
    bbox: tuple[float, float, float, float] | None = None,
    sl_sat: float = 0.10,
    k: float = 30.0,
    heavy_rain_mm: float = 10.0,
    rain_min_mm: float = 1.0,
    pm10_dust_threshold: float = 50.0,
    pm10_dust_scale: float = 0.02,
) -> pd.DataFrame:
    """Generate per-(station, year) SOMOSclean labels using NREL coordinates only.

    Loads the NREL annual CSV for station coordinates and years — the measured
    IWSR column is intentionally ignored. For each (station, year), fetches
    weather for that calendar year and computes SOMOSclean IWSR using the
    calibrated physics parameters. This gives ~891 training rows vs the 37-row
    seed-station fallback, with full geographic diversity across NREL's 15-state
    coverage, while keeping measured NREL soiling out of training.

    as_of for each row = Dec 31 of that year, matching nrel_panel convention so
    the weather lookback aligns with the actual observation period.
    """
    path = Path(nrel_csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"NREL annual CSV not found at {path}. "
            "Run scripts/12_ingest_nrel_soiling_map.py first."
        )
    df = pd.read_csv(path)
    required = {"station_id", "latitude", "longitude", "year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"NREL annual CSV missing columns: {missing}")
    if "iwsr_censored" in df.columns:
        before = len(df)
        df = df[~df["iwsr_censored"].astype(bool)].reset_index(drop=True)
        logger.info("Dropped %d censored rows; %d remain for coordinate source", before - len(df), len(df))
    if bbox is not None:
        minx, miny, maxx, maxy = bbox
        df = df[df["longitude"].between(minx, maxx) & df["latitude"].between(miny, maxy)].reset_index(drop=True)

    rows = []
    n_total = len(df)
    for i, (_, s) in enumerate(df.iterrows(), start=1):
        year = int(s["year"])
        as_of = date(year, 12, 31)
        start = as_of - timedelta(days=364)
        try:
            daily = fetch_combined(float(s["latitude"]), float(s["longitude"]), start, as_of, cache_dir=cache_dir)
        except Exception as exc:
            logger.warning("Skipping %s (%d): weather fetch failed (%s)", s["station_id"], year, exc)
            continue
        _, sl_series = somosclean_eqd_trajectory(
            daily,
            sl_sat=sl_sat,
            k=k,
            heavy_rain_mm=heavy_rain_mm,
            rain_min_mm=rain_min_mm,
            pm10_dust_threshold=pm10_dust_threshold,
            pm10_dust_scale=pm10_dust_scale,
        )
        iwsr = float(1.0 - sl_series.iloc[-1])
        rows.append(
            {
                "station_id": str(s["station_id"]),
                "latitude": float(s["latitude"]),
                "longitude": float(s["longitude"]),
                "year": year,
                "as_of": as_of.isoformat(),
                "iwsr": iwsr,
                "label": int(iwsr < iwsr_risk_threshold),
            }
        )
        if i % 100 == 0:
            logger.info("SOMOSclean panel labels: %d / %d rows", i, n_total)
    return pd.DataFrame(rows)


def kimber_soiling_ratio(
    daily: pd.DataFrame,
    deposition_per_pm25: float = 0.0006,
    rain_clean_mm: float = 1.0,
) -> pd.Series:
    """Kimber-style daily soiling ratio trajectory.

    Starts at 1.0 (clean), accumulates loss proportional to daily PM2.5, and
    resets to 1.0 on any day with precipitation >= rain_clean_mm. The
    `deposition_per_pm25` constant is the Kimber 2007 fit (~0.0006 fractional
    loss per (µg/m³ · day)); keep as a tunable in config.
    """
    pm25 = daily.get("pm2_5")
    precip = daily.get("precipitation_sum")
    if pm25 is None or precip is None:
        raise ValueError("Kimber model needs precipitation_sum and pm2_5 columns")
    ratio = np.ones(len(daily))
    current = 1.0
    pm25_filled = pm25.fillna(pm25.median() if not pm25.isna().all() else 10.0)
    pm25_v = pd.to_numeric(pm25_filled, errors="coerce").fillna(0.0).values
    precip_v = pd.to_numeric(precip.fillna(0.0), errors="coerce").fillna(0.0).values
    for i in range(len(daily)):
        if precip_v[i] >= rain_clean_mm:
            current = 1.0
        else:
            current = max(0.0, current - deposition_per_pm25 * pm25_v[i])
        ratio[i] = current
    return pd.Series(ratio, index=daily.index, name="soiling_ratio")


def kimber_synthetic_labels(
    stations: Iterable[dict],
    as_of: date,
    lookback_days: int = 180,
    iwsr_risk_threshold: float = 0.97,
    cache_dir: Path = Path(".cache/soiling"),
) -> pd.DataFrame:
    """Generate Kimber-based labels for a list of stations.

    Each station dict needs keys: station_id, latitude, longitude.
    """
    start = as_of - timedelta(days=lookback_days)
    rows = []
    for s in stations:
        try:
            daily = fetch_combined(s["latitude"], s["longitude"], start, as_of, cache_dir=cache_dir)
        except Exception as exc:
            logger.warning("Skipping %s: weather fetch failed (%s)", s["station_id"], exc)
            continue
        ratio = kimber_soiling_ratio(daily)
        iwsr = float(ratio.mean())
        rows.append(
            {
                "station_id": s["station_id"],
                "latitude": s["latitude"],
                "longitude": s["longitude"],
                "as_of": as_of.isoformat(),
                "iwsr": iwsr,
                "label": int(iwsr < iwsr_risk_threshold),
            }
        )
    return pd.DataFrame(rows)
