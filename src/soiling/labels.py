"""Label sources for the soiling risk model.

Two paths, in priority order:

1. `load_nrel_soiling_map` — ingest the NREL PV Soiling Map tabular release
   (Micheli/Deceglie/Muller; 255 US stations with IWSR). Expected schema once
   the CSV is placed under data/external/nrel_soiling_map.csv:
     station_id, latitude, longitude, iwsr, soiling_rate_pct_per_day,
     start_date, end_date
   The file is not redistributable here — request from NREL or extract from
   the published supplementary of Micheli 2019 (Prog. in Photovoltaics).

2. `kimber_synthetic_labels` — physics-proxy labels derived from the same
   weather stream used at inference time (precip + PM2.5). Useful for
   bootstrapping the pipeline before real labels are in hand; retrain once
   real IWSR data arrives.

Both paths return a DataFrame with columns:
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
