"""v1 cleaning-recommendation engine.

Implements the rule-based recommend stage from
``docs/PRODUCT_VISION.md`` ("Cleaning recommendation engine — staged").
Returns a payload that matches the v0 beta ``/recommend`` API contract.

v1 is intentionally simple: a hard-coded rule over (risk_score, 7-day rain
forecast, days_since_clean). v2 will swap in an ML-based recovery model
once the feedback loop has populated training rows.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable

import geopandas as gpd

from solarsoiled.manifest import write_manifest

logger = logging.getLogger(__name__)


FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


# Risk → recovery range placeholders. Replace once the feedback loop populates.
_RECOVERY_BUCKETS: tuple[tuple[float, str, tuple[float, float]], ...] = (
    (0.50, "low", (2.0, 5.0)),
    (0.75, "medium", (5.0, 12.0)),
    (1.01, "high", (10.0, 20.0)),
)


def _bucket(risk: float) -> tuple[str, tuple[float, float]]:
    for upper, name, recovery in _RECOVERY_BUCKETS:
        if risk < upper:
            return name, recovery
    return "high", _RECOVERY_BUCKETS[-1][2]


def _aggregate_risk(risks: Iterable[float]) -> float:
    """AOI-level risk = 90th percentile of per-array risk (ceiling rank)."""
    vals = sorted(float(r) for r in risks if r is not None)
    if not vals:
        return 0.0
    idx = min(len(vals) - 1, int(0.9 * len(vals)))
    return vals[idx]


def _default_forecast_fn(lat: float, lon: float, days: int) -> list[float]:
    """Hit Open-Meteo /forecast for daily precipitation_sum (mm).

    Caller-injectable so unit tests don't touch the network.
    """
    import requests

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum",
        "forecast_days": int(days),
        "timezone": "UTC",
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    return [float(v or 0.0) for v in payload["daily"]["precipitation_sum"]]


def _longest_dry_stretch_end(
    rain_per_day_mm: list[float],
    start: date,
    *,
    daily_threshold_mm: float,
) -> date | None:
    """Return the last date of the longest run of dry days, or None."""
    best_len = 0
    best_end_idx: int | None = None
    cur_len = 0
    cur_start: int | None = None
    for i, r in enumerate(rain_per_day_mm):
        if r < daily_threshold_mm:
            if cur_start is None:
                cur_start = i
            cur_len = i - cur_start + 1
            if cur_len > best_len:
                best_len = cur_len
                best_end_idx = i
        else:
            cur_len = 0
            cur_start = None
    if best_end_idx is None:
        return None
    return start + timedelta(days=best_end_idx)


def recommend_cleaning(
    risk_geojson: Path,
    last_cleaned: date,
    aoi_centroid: tuple[float, float],
    *,
    risk_threshold: float = 0.6,
    rain_mm_threshold: float = 5.0,
    min_days_since_clean: int = 30,
    forecast_days: int = 7,
    forecast_fn: Callable[[float, float, int], list[float]] | None = None,
    today: date | None = None,
) -> dict:
    """Return a recommendation payload matching the v0 beta /recommend contract.

    ``risk_geojson`` is the output of script 11 (``risk.geojson``), one feature
    per detected array with a ``risk_score`` property. ``aoi_centroid`` is
    ``(lat, lon)`` in WGS84. ``forecast_fn`` is injectable for tests.
    """
    today = today or datetime.utcnow().date()
    forecast_fn = forecast_fn or _default_forecast_fn

    gdf = gpd.read_file(risk_geojson)
    if "risk_score" not in gdf.columns:
        raise ValueError(f"{risk_geojson} has no risk_score column")
    aoi_risk = _aggregate_risk(gdf["risk_score"].tolist())
    confidence, recovery_range = _bucket(aoi_risk)

    days_since_clean = (today - last_cleaned).days
    rain_forecast = forecast_fn(aoi_centroid[0], aoi_centroid[1], forecast_days)
    total_rain_mm = sum(rain_forecast)
    dry_end = _longest_dry_stretch_end(
        rain_forecast,
        start=today,
        daily_threshold_mm=rain_mm_threshold / max(forecast_days, 1),
    )

    rule_fired = "below_risk_threshold"
    window_start: date | None = None
    window_end: date | None = None

    if aoi_risk < risk_threshold:
        rule_fired = "below_risk_threshold"
    elif days_since_clean < min_days_since_clean:
        rule_fired = "below_age_threshold"
    elif total_rain_mm >= rain_mm_threshold:
        rule_fired = "deferred_due_to_rain"
    else:
        rule_fired = "weather_window_open"
        window_start = today + timedelta(days=1)
        window_end = dry_end if dry_end and dry_end > window_start else (today + timedelta(days=forecast_days))

    return {
        "window_start": window_start.isoformat() if window_start else None,
        "window_end": window_end.isoformat() if window_end else None,
        "expected_recovery_pct": list(recovery_range),
        "confidence": confidence,
        "rule_fired": rule_fired,
        "model_version": "recommend-v1-rule",
        "beta": True,
        "known_limitations": [
            "Rule-based v1; expected_recovery_pct is a static placeholder per risk bucket",
            "Single AOI-centroid forecast (no spatial aggregation across arrays)",
        ],
        "inputs": {
            "aoi_risk_p90": float(aoi_risk),
            "n_arrays": int(len(gdf)),
            "days_since_clean": int(days_since_clean),
            "total_rain_7d_mm": float(total_rain_mm),
            "last_cleaned": last_cleaned.isoformat(),
            "today": today.isoformat(),
        },
    }


def write_recommendation(
    out_path: Path,
    payload: dict,
    *,
    upstream_manifest: Path | None = None,
) -> Path:
    """Write the recommend payload + sibling manifest.json."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    inputs: list[str] = []
    if upstream_manifest and Path(upstream_manifest).is_file():
        inputs.append(str(upstream_manifest))
    write_manifest(
        out_path.parent,
        stage="recommend",
        model_version=payload.get("model_version", "recommend-v1-rule"),
        model_weights=None,
        inputs=inputs,
        beta=bool(payload.get("beta", True)),
        metrics={
            "aoi_risk_p90": payload["inputs"]["aoi_risk_p90"],
            "n_arrays": payload["inputs"]["n_arrays"],
            "days_since_clean": payload["inputs"]["days_since_clean"],
            "total_rain_7d_mm": payload["inputs"]["total_rain_7d_mm"],
        },
        known_limitations=list(payload.get("known_limitations", [])),
        extra={
            "rule_fired": payload["rule_fired"],
            "recommendations_json": str(out_path),
        },
        filename="manifest.recommend.json",
    )
    return out_path
