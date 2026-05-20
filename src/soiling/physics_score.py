"""SOMOSclean physics-based soiling risk scorer.

Replaces the XGBoost model for V1. Each detected array is scored by running
the SOMOSclean trajectory on its trailing weather window and reporting the
terminal soiling loss as the risk score.

Risk score is normalized to [0, 1] by dividing by sl_sat so that the
downstream recommend rule (threshold at 0.6) is on a consistent scale:
  0.0 = clean (no accumulated soiling)
  0.5 = 50% of saturation ceiling reached
  1.0 = at or above saturation ceiling

Output columns added to the array GeoDataFrame:
  risk_score        float [0, 1] — primary signal consumed by recommend
  soiling_loss_pct  float — raw SL in percent (e.g. 6.3)
  eqd               float — equivalent days of accumulation at scoring date
  last_rain_date    str | None — ISO date of last heavy rain event (≥10 mm)
  scored_at         str — ISO datetime of scoring
  scoring_method    str — always "somosclean-physics-v1"
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from soiling.labels import somosclean_eqd_trajectory
from soiling.weather_client import fetch_combined

logger = logging.getLogger(__name__)

# SOMOSclean parameters — mirror configs/soiling/features.yaml:somosclean
_DEFAULT_PARAMS = {
    "sl_sat": 0.10,
    "k": 30.0,
    "heavy_rain_mm": 10.0,
    "rain_min_mm": 1.0,
    "pm10_dust_threshold": 50.0,
    "pm10_dust_scale": 0.02,
}


def score_arrays(
    gdf: gpd.GeoDataFrame,
    *,
    as_of: Optional[date] = None,
    lookback_days: int = 365,
    cache_dir: Path = Path(".cache/soiling"),
    params: Optional[dict] = None,
) -> gpd.GeoDataFrame:
    """Run SOMOSclean on each array and attach risk columns.

    Arrays with missing centroids or failed weather fetches get risk_score=NaN.
    The GeoDataFrame must have a geometry column in any CRS (centroids are
    reprojected to WGS84 for the weather fetch).
    """
    p = {**_DEFAULT_PARAMS, **(params or {})}
    sl_sat = float(p["sl_sat"])
    as_of = as_of or datetime.now(timezone.utc).date()
    start = as_of - timedelta(days=lookback_days)
    scored_at = datetime.now(timezone.utc).isoformat()

    gdf = gdf.copy()
    # Compute centroids in a projected CRS then reproject to WGS84 for lookups.
    centroids_wgs84 = gdf.geometry.to_crs("EPSG:3857").centroid.to_crs("EPSG:4326")

    risk_scores: list[float] = []
    soiling_loss_pcts: list[float] = []
    eqds: list[float] = []
    last_rain_dates: list[Optional[str]] = []

    n = len(gdf)
    for i, (_, geom) in enumerate(zip(gdf.index, centroids_wgs84)):
        lat, lon = geom.y, geom.x
        try:
            daily = fetch_combined(lat, lon, start, as_of, cache_dir=cache_dir)
            traj_kwargs = {k: p[k] for k in (
                "sl_sat", "k", "heavy_rain_mm", "rain_min_mm",
                "pm10_dust_threshold", "pm10_dust_scale",
            )}
            eqd_series, sl_series = somosclean_eqd_trajectory(daily, **traj_kwargs)

            sl_terminal = float(sl_series.iloc[-1])
            risk = min(1.0, sl_terminal / sl_sat) if sl_sat > 0 else 0.0
            eqd = float(eqd_series.iloc[-1])

            # Last heavy rain: last date where precip >= heavy_rain_mm
            precip = daily.get("precipitation_sum")
            last_rain: Optional[str] = None
            if precip is not None:
                heavy = precip[pd.to_numeric(precip, errors="coerce").fillna(0) >= float(p["heavy_rain_mm"])]
                if not heavy.empty:
                    last_rain = heavy.index[-1].date().isoformat()

            risk_scores.append(risk)
            soiling_loss_pcts.append(round(sl_terminal * 100, 3))
            eqds.append(round(eqd, 1))
            last_rain_dates.append(last_rain)

        except Exception as exc:
            logger.warning("Physics score failed for array %d (%.4f, %.4f): %s", i, lat, lon, exc)
            risk_scores.append(float("nan"))
            soiling_loss_pcts.append(float("nan"))
            eqds.append(float("nan"))
            last_rain_dates.append(None)

        if (i + 1) % 10 == 0 or (i + 1) == n:
            logger.info("Scored %d / %d arrays", i + 1, n)

    gdf["risk_score"] = risk_scores
    gdf["soiling_loss_pct"] = soiling_loss_pcts
    gdf["eqd"] = eqds
    gdf["last_rain_date"] = last_rain_dates
    gdf["scored_at"] = scored_at
    gdf["scoring_method"] = "somosclean-physics-v1"
    return gdf
