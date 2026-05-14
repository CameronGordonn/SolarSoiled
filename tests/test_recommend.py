"""Unit tests for solarsoiled.recommend.

Network-free: ``forecast_fn`` is injected with deterministic test fixtures
so all five rule_fired branches are covered.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from solarsoiled.recommend import recommend_cleaning, write_recommendation


def _write_risk_geojson(tmp_path: Path, scores: list[float]) -> Path:
    geoms = [Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)]) for i in range(len(scores))]
    gdf = gpd.GeoDataFrame({"risk_score": scores, "geometry": geoms}, crs="EPSG:4326")
    out = tmp_path / "risk.geojson"
    gdf.to_file(out, driver="GeoJSON")
    return out


def _fake_forecast(rain_pattern: list[float]):
    def _fn(lat, lon, days):
        return list(rain_pattern[:days]) + [0.0] * max(days - len(rain_pattern), 0)
    return _fn


def test_below_risk_threshold(tmp_path: Path):
    risk = _write_risk_geojson(tmp_path, [0.1, 0.2, 0.3])
    p = recommend_cleaning(
        risk_geojson=risk,
        last_cleaned=date(2026, 1, 1),
        aoi_centroid=(36.95, -122.05),
        forecast_fn=_fake_forecast([0.0] * 7),
        today=date(2026, 4, 30),
    )
    assert p["rule_fired"] == "below_risk_threshold"
    assert p["window_start"] is None
    assert p["confidence"] == "low"


def test_below_age_threshold(tmp_path: Path):
    risk = _write_risk_geojson(tmp_path, [0.8, 0.9])
    p = recommend_cleaning(
        risk_geojson=risk,
        last_cleaned=date(2026, 4, 20),  # 10 days ago
        aoi_centroid=(36.95, -122.05),
        forecast_fn=_fake_forecast([0.0] * 7),
        today=date(2026, 4, 30),
    )
    assert p["rule_fired"] == "below_age_threshold"
    assert p["window_start"] is None
    assert p["confidence"] == "high"


def test_deferred_due_to_rain(tmp_path: Path):
    risk = _write_risk_geojson(tmp_path, [0.7, 0.8])
    p = recommend_cleaning(
        risk_geojson=risk,
        last_cleaned=date(2026, 1, 1),
        aoi_centroid=(36.95, -122.05),
        forecast_fn=_fake_forecast([5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0]),  # 15mm total
        today=date(2026, 4, 30),
    )
    assert p["rule_fired"] == "deferred_due_to_rain"
    assert p["window_start"] is None


def test_weather_window_open(tmp_path: Path):
    risk = _write_risk_geojson(tmp_path, [0.7, 0.8])
    p = recommend_cleaning(
        risk_geojson=risk,
        last_cleaned=date(2026, 1, 1),
        aoi_centroid=(36.95, -122.05),
        forecast_fn=_fake_forecast([0.0] * 7),
        today=date(2026, 4, 30),
    )
    assert p["rule_fired"] == "weather_window_open"
    assert p["window_start"] == "2026-05-01"
    assert p["window_end"] == "2026-05-06"  # 0..6 indexed from today, end of 7-day stretch
    assert p["confidence"] == "high"
    assert p["expected_recovery_pct"] == [10.0, 20.0]


def test_payload_shape_matches_contract(tmp_path: Path):
    risk = _write_risk_geojson(tmp_path, [0.7])
    p = recommend_cleaning(
        risk_geojson=risk,
        last_cleaned=date(2026, 1, 1),
        aoi_centroid=(36.95, -122.05),
        forecast_fn=_fake_forecast([0.0] * 7),
        today=date(2026, 4, 30),
    )
    for key in (
        "window_start",
        "window_end",
        "expected_recovery_pct",
        "confidence",
        "rule_fired",
        "model_version",
        "beta",
        "known_limitations",
    ):
        assert key in p, f"missing key {key}"
    assert p["beta"] is True


def test_aggregate_uses_p90(tmp_path: Path):
    risk = _write_risk_geojson(tmp_path, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95])
    # p90 of 10 values is the index 9 (0-indexed) — 0.95
    p = recommend_cleaning(
        risk_geojson=risk,
        last_cleaned=date(2026, 1, 1),
        aoi_centroid=(36.95, -122.05),
        forecast_fn=_fake_forecast([0.0] * 7),
        today=date(2026, 4, 30),
    )
    assert p["inputs"]["aoi_risk_p90"] >= 0.9
    assert p["rule_fired"] == "weather_window_open"


def test_write_recommendation_round_trip(tmp_path: Path):
    risk = _write_risk_geojson(tmp_path, [0.7])
    payload = recommend_cleaning(
        risk_geojson=risk,
        last_cleaned=date(2026, 1, 1),
        aoi_centroid=(36.95, -122.05),
        forecast_fn=_fake_forecast([0.0] * 7),
        today=date(2026, 4, 30),
    )
    out = write_recommendation(tmp_path / "recommendations.json", payload)
    on_disk = json.loads(out.read_text())
    assert on_disk["rule_fired"] == payload["rule_fired"]
    # Sibling manifest written alongside.
    assert (tmp_path / "manifest.recommend.json").is_file()
