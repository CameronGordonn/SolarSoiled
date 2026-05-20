"""Rolling-window aggregation over daily weather/AQ time series.

Produces a single feature row per (array, as-of date) that can be joined onto
`array_features.geo.parquet` on `array_id`.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.soiling.labels import kimber_soiling_ratio, somosclean_eqd_trajectory


DEFAULT_WINDOWS: tuple[int, ...] = (7, 30, 90)


def _window_stats(series: pd.Series, days: int, as_of: pd.Timestamp, agg: str) -> float:
    start = as_of - pd.Timedelta(days=days - 1)
    window = series.loc[start:as_of]
    if window.empty or window.isna().all():
        return float("nan")
    if agg == "sum":
        return float(window.sum())
    if agg == "mean":
        return float(window.mean())
    if agg == "max":
        return float(window.max())
    raise ValueError(f"Unknown agg: {agg}")


def _dry_day_streak(precip: pd.Series, as_of: pd.Timestamp, threshold_mm: float = 1.0) -> int:
    """Consecutive days ending at `as_of` with precip < threshold_mm."""
    tail = precip.loc[:as_of].dropna()
    streak = 0
    for v in tail.values[::-1]:
        if v < threshold_mm:
            streak += 1
        else:
            break
    return streak


def _days_since_significant_rain(precip: pd.Series, as_of: pd.Timestamp, threshold_mm: float = 5.0) -> int:
    tail = precip.loc[:as_of].dropna()
    for i, v in enumerate(tail.values[::-1]):
        if v >= threshold_mm:
            return i
    return len(tail)


def build_feature_row(
    daily: pd.DataFrame,
    as_of: date,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    kimber_cfg: Mapping | None = None,
    somosclean_cfg: Mapping | None = None,
) -> dict:
    """Aggregate one weather DataFrame into a single-row feature dict.

    `daily` must have a DatetimeIndex and standard Open-Meteo column names
    (precipitation_sum, pm2_5, wind_speed_10m_max, relative_humidity_2m_mean).

    If `kimber_cfg` has `use_kimber_feature: true` and both `precipitation_sum`
    and `pm2_5` are present, a Kimber physics-proxy IWSR trajectory is computed
    and exposed as `kimber_iwsr_proxy` (full-window mean) plus per-window means.
    Without this, the model sees only raw weather — the Kimber feature gives it
    a dense physics prior so it can learn the NREL residual.
    """
    as_of_ts = pd.Timestamp(as_of)
    row: dict = {"as_of": as_of.isoformat()}

    precip = daily.get("precipitation_sum")
    if precip is not None:
        for w in windows:
            row[f"precip_{w}d_mm"] = _window_stats(precip, w, as_of_ts, "sum")
        row["dry_day_streak"] = _dry_day_streak(precip, as_of_ts)
        row["days_since_rain_5mm"] = _days_since_significant_rain(precip, as_of_ts)

    for col, agg in (
        ("wind_speed_10m_max", "mean"),
        ("relative_humidity_2m_mean", "mean"),
        ("temperature_2m_max", "mean"),
        ("pm2_5", "mean"),
        ("pm10", "mean"),
    ):
        if col not in daily.columns:
            continue
        for w in windows:
            row[f"{col}_{w}d_{agg}"] = _window_stats(daily[col], w, as_of_ts, agg)

    if kimber_cfg and kimber_cfg.get("use_kimber_feature"):
        if "precipitation_sum" in daily.columns and "pm2_5" in daily.columns:
            ratio = kimber_soiling_ratio(
                daily,
                deposition_per_pm25=float(kimber_cfg.get("deposition_per_pm25", 0.0006)),
                rain_clean_mm=float(kimber_cfg.get("rain_clean_mm", 1.0)),
            )
            row["kimber_iwsr_proxy"] = float(ratio.mean())
            for w in windows:
                row[f"kimber_iwsr_{w}d_mean"] = _window_stats(ratio, w, as_of_ts, "mean")

    if somosclean_cfg and somosclean_cfg.get("use_somosclean_feature"):
        if "precipitation_sum" in daily.columns:
            _, sl_series = somosclean_eqd_trajectory(
                daily,
                sl_sat=float(somosclean_cfg.get("sl_sat", 0.25)),
                k=float(somosclean_cfg.get("k", 30.0)),
                heavy_rain_mm=float(somosclean_cfg.get("heavy_rain_mm", 10.0)),
                rain_min_mm=float(somosclean_cfg.get("rain_min_mm", 1.0)),
                pm10_dust_threshold=float(somosclean_cfg.get("pm10_dust_threshold", 50.0)),
                pm10_dust_scale=float(somosclean_cfg.get("pm10_dust_scale", 0.02)),
            )
            row["somosclean_sl_proxy"] = float(sl_series.iloc[-1])
            for w in (7, 30):
                row[f"somosclean_sl_{w}d_mean"] = _window_stats(sl_series, w, as_of_ts, "mean")

    return row


def build_feature_matrix(
    samples: Iterable[dict],
    windows: Sequence[int] = DEFAULT_WINDOWS,
    kimber_cfg: Mapping | None = None,
    somosclean_cfg: Mapping | None = None,
) -> pd.DataFrame:
    """Given an iterable of {"id", "daily": DataFrame, "as_of": date, **static},
    return a joined feature matrix keyed by `id`.
    """
    rows = []
    for s in samples:
        base = {"id": s["id"]}
        base.update({k: v for k, v in s.items() if k not in {"id", "daily", "as_of"}})
        base.update(build_feature_row(s["daily"], s["as_of"], windows=windows, kimber_cfg=kimber_cfg, somosclean_cfg=somosclean_cfg))
        rows.append(base)
    return pd.DataFrame(rows)
