"""Tests for rolling-window aggregation + Kimber physics proxy."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.soiling.feature_engineering import (
    build_feature_row,
    _days_since_significant_rain,
    _dry_day_streak,
    _window_stats,
)
from src.soiling.labels import kimber_soiling_ratio
from src.soiling.risk_model import (
    SpatialCVConfig,
    impute_with_feature_medians,
    load_feature_medians,
    save_model,
    train_risk_model,
)


def _synth_daily(n: int = 90, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "precipitation_sum": rng.choice([0.0, 0.0, 0.0, 5.0], size=n),
            "wind_speed_10m_max": rng.uniform(5, 15, size=n),
            "relative_humidity_2m_mean": rng.uniform(40, 90, size=n),
            "temperature_2m_max": rng.uniform(10, 30, size=n),
            "pm2_5": rng.uniform(5, 40, size=n),
            "pm10": rng.uniform(10, 80, size=n),
        },
        index=idx,
    )


def test_window_stats_sum_matches_tail():
    idx = pd.date_range("2026-01-01", periods=10, freq="D")
    s = pd.Series(range(10), index=idx, dtype=float)
    # Last 3 days: 7+8+9 = 24
    assert _window_stats(s, 3, pd.Timestamp("2026-01-10"), "sum") == pytest.approx(24.0)


def test_dry_day_streak_counts_trailing_dry_days():
    idx = pd.date_range("2026-01-01", periods=6, freq="D")
    precip = pd.Series([5.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=idx)
    assert _dry_day_streak(precip, pd.Timestamp("2026-01-06")) == 5


def test_dry_day_streak_resets_on_rain():
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    precip = pd.Series([0.0, 0.0, 2.0, 0.0], index=idx)
    assert _dry_day_streak(precip, pd.Timestamp("2026-01-04")) == 1


def test_days_since_significant_rain():
    idx = pd.date_range("2026-01-01", periods=5, freq="D")
    precip = pd.Series([10.0, 0.0, 0.0, 0.0, 0.0], index=idx)
    assert _days_since_significant_rain(precip, pd.Timestamp("2026-01-05"), threshold_mm=5.0) == 4


def test_build_feature_row_emits_expected_columns():
    daily = _synth_daily(60)
    row = build_feature_row(daily, date(2026, 2, 28), windows=(7, 30))
    assert "precip_7d_mm" in row and "precip_30d_mm" in row
    assert "dry_day_streak" in row and "days_since_rain_5mm" in row
    assert "pm2_5_30d_mean" in row
    assert "wind_speed_10m_max_7d_mean" in row


def test_build_feature_row_omits_kimber_by_default():
    daily = _synth_daily(60)
    row = build_feature_row(daily, date(2026, 2, 28), windows=(7, 30))
    assert "kimber_iwsr_proxy" not in row
    assert "kimber_iwsr_7d_mean" not in row


def test_build_feature_row_emits_kimber_when_enabled():
    daily = _synth_daily(60)
    kimber_cfg = {"use_kimber_feature": True, "deposition_per_pm25": 0.001, "rain_clean_mm": 1.0}
    row = build_feature_row(daily, date(2026, 2, 28), windows=(7, 30), kimber_cfg=kimber_cfg)
    assert "kimber_iwsr_proxy" in row
    assert "kimber_iwsr_7d_mean" in row and "kimber_iwsr_30d_mean" in row
    # IWSR is bounded in [0, 1]
    assert 0.0 <= row["kimber_iwsr_proxy"] <= 1.0
    assert 0.0 <= row["kimber_iwsr_7d_mean"] <= 1.0


def test_build_feature_row_kimber_skipped_when_columns_missing():
    idx = pd.date_range("2026-01-01", periods=30, freq="D")
    daily = pd.DataFrame({"wind_speed_10m_max": np.full(30, 5.0)}, index=idx)
    kimber_cfg = {"use_kimber_feature": True}
    row = build_feature_row(daily, date(2026, 1, 30), windows=(7,), kimber_cfg=kimber_cfg)
    assert "kimber_iwsr_proxy" not in row


def test_train_risk_model_returns_calibrator_when_cv_runs():
    """Confirm the (model, metrics, calibrator) triple shape; calibrator should
    appear once spatial CV produces out-of-fold predictions on a multi-class y.
    """

    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "f3": rng.normal(size=n),
    })
    # Make labels somewhat learnable — split on f1 + noise
    y = ((X["f1"].values + 0.3 * rng.normal(size=n)) > 0).astype(int)
    # Spread points across multiple lat/lon clusters (>=2 needed)
    lats = np.tile(np.arange(8.0, 18.0), n // 10 + 1)[:n]
    lons = np.tile(np.arange(-120.0, -110.0), n // 10 + 1)[:n]

    model, metrics, calibrator = train_risk_model(
        X, y, lats=lats, lons=lons, cv=SpatialCVConfig(n_folds=3, cluster_km=120),
        target_mode="binary",
    )
    assert metrics["n_folds"] >= 2
    assert calibrator is not None
    # Calibrator monotonic in [0,1]
    pred_lo = calibrator.predict([0.05])[0]
    pred_hi = calibrator.predict([0.95])[0]
    assert 0.0 <= pred_lo <= 1.0 and 0.0 <= pred_hi <= 1.0


def test_train_risk_model_regression_target_mode_emits_spearman():
    """Regression mode reports a Spearman rank correlation alongside AUC."""

    rng = np.random.default_rng(1)
    n = 80
    X = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
    })
    iwsr = 0.95 + 0.04 * rng.random(n) - 0.02 * (X["f1"].values > 0)
    y = (iwsr < 0.97).astype(int)
    lats = np.tile(np.arange(8.0, 18.0), n // 10 + 1)[:n]
    lons = np.tile(np.arange(-120.0, -110.0), n // 10 + 1)[:n]

    model, metrics, _calibrator = train_risk_model(
        X, y, lats=lats, lons=lons, cv=SpatialCVConfig(n_folds=3, cluster_km=120),
        target_mode="regression", iwsr_target=iwsr,
    )
    assert metrics["target_mode"] == "regression"
    assert "mean_spearman" in metrics


def test_kimber_ratio_resets_on_heavy_rain():
    idx = pd.date_range("2026-01-01", periods=5, freq="D")
    daily = pd.DataFrame(
        {
            "precipitation_sum": [0.0, 0.0, 0.0, 10.0, 0.0],
            "pm2_5": [50.0, 50.0, 50.0, 50.0, 50.0],
        },
        index=idx,
    )
    ratio = kimber_soiling_ratio(daily, deposition_per_pm25=0.01, rain_clean_mm=1.0)
    # Days 1-3: accumulating loss. Day 4: rain reset to 1.0. Day 5: loss resumes.
    assert ratio.iloc[3] == pytest.approx(1.0)
    assert ratio.iloc[2] < 1.0
    assert ratio.iloc[4] < 1.0
    assert ratio.iloc[4] > ratio.iloc[2]


def test_kimber_ratio_floors_at_zero():
    idx = pd.date_range("2026-01-01", periods=50, freq="D")
    daily = pd.DataFrame(
        {
            "precipitation_sum": [0.0] * 50,
            "pm2_5": [100.0] * 50,
        },
        index=idx,
    )
    ratio = kimber_soiling_ratio(daily, deposition_per_pm25=0.05, rain_clean_mm=1.0)
    assert ratio.min() >= 0.0
    assert ratio.iloc[-1] == pytest.approx(0.0)


def test_train_risk_model_persists_feature_medians(tmp_path):
    rng = np.random.default_rng(2)
    n = 60
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
        }
    )
    X.loc[::5, "f2"] = np.nan
    y = (X["f1"].fillna(0.0) > 0).astype(int).to_numpy()
    lats = np.tile(np.arange(8.0, 18.0), n // 10 + 1)[:n]
    lons = np.tile(np.arange(-120.0, -110.0), n // 10 + 1)[:n]

    model, _metrics, _calibrator = train_risk_model(
        X, y, lats=lats, lons=lons, cv=SpatialCVConfig(n_folds=3, cluster_km=120)
    )
    medians = getattr(model, "feature_medians_", None)
    assert medians is not None
    assert "f2" in medians

    out_dir = tmp_path / "run"
    save_model(model, out_dir, feature_names=X.columns, metrics={})
    loaded = load_feature_medians(out_dir)
    assert loaded == medians


def test_impute_with_feature_medians_uses_training_values():
    X = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, np.nan]})
    filled = impute_with_feature_medians(X, {"a": 2.5, "b": 7.0})
    assert filled["a"].tolist() == [1.0, 2.5]
    assert filled["b"].tolist() == [7.0, 7.0]
