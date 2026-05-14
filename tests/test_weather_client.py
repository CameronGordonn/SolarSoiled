"""Tests for src.soiling.weather_client — HTTP mocked so they run offline."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.soiling import weather_client as wc


def _mock_weather_response(n_days: int = 7) -> dict:
    dates = [f"2026-03-{i+1:02d}" for i in range(n_days)]
    return {
        "daily": {
            "time": dates,
            "precipitation_sum": [0.0, 2.1, 0.0, 0.0, 0.5, 0.0, 10.3][:n_days],
            "wind_speed_10m_max": [8.0] * n_days,
            "relative_humidity_2m_mean": [65.0] * n_days,
            "temperature_2m_max": [18.0] * n_days,
            "temperature_2m_min": [10.0] * n_days,
            "shortwave_radiation_sum": [20.0] * n_days,
        }
    }


def _mock_aq_response(n_hours: int = 48) -> dict:
    times = pd.date_range("2026-03-01", periods=n_hours, freq="h")
    return {
        "hourly": {
            "time": [t.isoformat() for t in times],
            "pm2_5": [12.0] * n_hours,
            "pm10": [25.0] * n_hours,
        }
    }


def _patched_session(json_payload):
    resp = MagicMock()
    resp.json.return_value = json_payload
    resp.raise_for_status.return_value = None
    sess = MagicMock()
    sess.get.return_value = resp
    return sess


def test_fetch_weather_returns_daily_indexed_frame():
    with patch.object(wc, "_session", return_value=_patched_session(_mock_weather_response(7))):
        df = wc.fetch_weather(36.97, -122.03, date(2026, 3, 1), date(2026, 3, 7))
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) == 7
    assert "precipitation_sum" in df.columns
    assert df["precipitation_sum"].sum() == pytest.approx(12.9)


def test_fetch_weather_raises_on_missing_daily_block():
    with patch.object(wc, "_session", return_value=_patched_session({"error": "nope"})):
        with pytest.raises(RuntimeError, match="no daily block"):
            wc.fetch_weather(0.0, 0.0, date(2026, 1, 1), date(2026, 1, 2))


def test_fetch_air_quality_resamples_to_daily():
    with patch.object(wc, "_session", return_value=_patched_session(_mock_aq_response(48))):
        df = wc.fetch_air_quality(36.97, -122.03, date(2026, 3, 1), date(2026, 3, 2))
    assert len(df) == 2  # two days from 48 hours
    assert df["pm2_5"].iloc[0] == pytest.approx(12.0)


def test_fetch_combined_continues_when_aq_fails():
    def fake_session_factory(cache_dir, expire_after_days=30):
        # Raise on the AQ URL, succeed on the archive URL.
        sess = MagicMock()
        def get(url, params=None, timeout=None):
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            if "air-quality" in url:
                raise RuntimeError("network flake")
            resp.json.return_value = _mock_weather_response(3)
            return resp
        sess.get.side_effect = get
        return sess

    with patch.object(wc, "_session", side_effect=fake_session_factory):
        df = wc.fetch_combined(36.97, -122.03, date(2026, 3, 1), date(2026, 3, 3))
    assert len(df) == 3
    assert "precipitation_sum" in df.columns
    # AQ columns absent but the call did not raise.
    assert "pm2_5" not in df.columns
