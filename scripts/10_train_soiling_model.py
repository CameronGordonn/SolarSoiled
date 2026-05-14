"""Train the XGBoost soiling risk classifier on CA station labels.

Label source is selected via configs/soiling/california.yaml:
  - kimber_proxy       → synthetic IWSR from weather (bootstrap, default)
  - nrel_soiling_map   → real IWSR from NREL CSV (requires data/external file)
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.soiling.feature_engineering import build_feature_row
from src.soiling.labels import (
    kimber_synthetic_labels,
    load_nrel_panel,
    load_nrel_soiling_map,
)
from src.soiling.location_features import load_static_lookup, location_feature_vector
from src.soiling.risk_model import (
    SpatialCVConfig,
    impute_with_feature_medians,
    save_model,
    train_risk_model,
)
from src.soiling.weather_client import fetch_combined
from solarsoiled.manifest import write_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Seed CA station coordinates for the kimber_proxy fallback. Spread across
# six climate regimes so spatial CV has non-degenerate folds:
#   coastal NorCal · SF Bay / Central Coast · Central Valley · North interior
#   · Sierra Nevada · Mojave · Colorado/Imperial desert · SoCal coastal · SoCal inland
# Swap for NREL CA stations once data/external/nrel_soiling_map.csv lands.
DEFAULT_CA_STATIONS = [
    # Coastal NorCal — wet, foggy, low dust
    {"station_id": "ca_crescent_city", "latitude": 41.756, "longitude": -124.202},
    {"station_id": "ca_eureka",        "latitude": 40.802, "longitude": -124.164},
    {"station_id": "ca_point_reyes",   "latitude": 38.070, "longitude": -122.800},
    {"station_id": "ca_santa_cruz",    "latitude": 36.974, "longitude": -122.030},
    # Bay / Central Coast
    {"station_id": "ca_half_moon_bay", "latitude": 37.463, "longitude": -122.428},
    {"station_id": "ca_san_jose",      "latitude": 37.339, "longitude": -121.895},
    {"station_id": "ca_san_luis_obispo","latitude":35.283, "longitude": -120.659},
    {"station_id": "ca_santa_barbara", "latitude": 34.420, "longitude": -119.698},
    {"station_id": "ca_ventura",       "latitude": 34.275, "longitude": -119.229},
    # Central Valley — ag dust, long dry season, high PM
    {"station_id": "ca_sacramento",    "latitude": 38.582, "longitude": -121.494},
    {"station_id": "ca_stockton",      "latitude": 37.958, "longitude": -121.290},
    {"station_id": "ca_modesto",       "latitude": 37.639, "longitude": -120.997},
    {"station_id": "ca_merced",        "latitude": 37.302, "longitude": -120.483},
    {"station_id": "ca_fresno",        "latitude": 36.748, "longitude": -119.772},
    {"station_id": "ca_visalia",       "latitude": 36.330, "longitude": -119.292},
    {"station_id": "ca_bakersfield",   "latitude": 35.373, "longitude": -119.019},
    # North interior
    {"station_id": "ca_redding",       "latitude": 40.586, "longitude": -122.391},
    {"station_id": "ca_chico",         "latitude": 39.728, "longitude": -121.837},
    {"station_id": "ca_yreka",         "latitude": 41.735, "longitude": -122.634},
    # Sierra Nevada — elevation, winter snow/rain
    {"station_id": "ca_truckee",       "latitude": 39.328, "longitude": -120.183},
    {"station_id": "ca_tahoe_south",   "latitude": 38.934, "longitude": -119.984},
    {"station_id": "ca_yosemite",      "latitude": 37.748, "longitude": -119.592},
    {"station_id": "ca_mammoth_lakes", "latitude": 37.649, "longitude": -118.972},
    # Mojave — extreme dry, dust, high PM10
    {"station_id": "ca_mojave",        "latitude": 35.054, "longitude": -118.176},
    {"station_id": "ca_ridgecrest",    "latitude": 35.622, "longitude": -117.670},
    {"station_id": "ca_barstow",       "latitude": 34.896, "longitude": -117.017},
    {"station_id": "ca_twentynine_palms","latitude":34.135,"longitude": -116.054},
    # Colorado / Imperial Desert — irrigation ag + desert dust
    {"station_id": "ca_palm_springs",  "latitude": 33.823, "longitude": -116.546},
    {"station_id": "ca_indio",         "latitude": 33.721, "longitude": -116.216},
    {"station_id": "ca_blythe",        "latitude": 33.610, "longitude": -114.597},
    {"station_id": "ca_el_centro",     "latitude": 32.792, "longitude": -115.563},
    # SoCal coastal — marine layer, moderate PM
    {"station_id": "ca_long_beach",    "latitude": 33.770, "longitude": -118.194},
    {"station_id": "ca_oceanside",     "latitude": 33.196, "longitude": -117.379},
    {"station_id": "ca_san_diego",     "latitude": 32.716, "longitude": -117.161},
    # SoCal inland — high PM, dry
    {"station_id": "ca_los_angeles",   "latitude": 34.052, "longitude": -118.244},
    {"station_id": "ca_santa_ana",     "latitude": 33.746, "longitude": -117.867},
    {"station_id": "ca_riverside",     "latitude": 33.953, "longitude": -117.396},
]


def _window_start(as_of: date, lookback_days: int) -> date:
    """Inclusive lookback: 365 days ending Dec 31 starts on Jan 1, not Dec 31."""
    return as_of - timedelta(days=max(lookback_days - 1, 0))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/soiling/california.yaml")
    p.add_argument("--features-config", default="configs/soiling/features.yaml")
    p.add_argument("--model-config", default="configs/soiling/model.yaml")
    p.add_argument("--run-name", default=None)
    p.add_argument("--dry-run", action="store_true", help="Build labels + features, skip training")
    p.add_argument("--holdout-year", type=int, default=None,
                   help="Panel-mode only: hold out this label year for temporal validation")
    return p.parse_args()


def _load_labels(region_cfg: dict, repo_root: Path) -> pd.DataFrame:
    source = region_cfg.get("label_source", "kimber_proxy")
    as_of = datetime.utcnow().date()
    if source in ("nrel_soiling_map", "nrel_panel"):
        bbox_cfg = region_cfg.get("bbox")
        bbox = tuple(bbox_cfg) if bbox_cfg else None
        if source == "nrel_panel":
            path = repo_root / region_cfg.get("nrel_panel_csv_path", "data/external/nrel_soiling_map_annual.csv")
            df = load_nrel_panel(path, bbox=bbox)
        else:
            path = repo_root / region_cfg["nrel_csv_path"]
            df = load_nrel_soiling_map(path, bbox=bbox)
        if region_cfg.get("drop_censored", True) and "iwsr_censored" in df.columns:
            before = len(df)
            df = df[~df["iwsr_censored"].astype(bool)].reset_index(drop=True)
            logger.info("Dropped %d censored (>0.99) rows; %d remain", before - len(df), len(df))
        df["label"] = (df["iwsr"] < region_cfg["iwsr_risk_threshold"]).astype(int)
        if source == "nrel_panel":
            # Panel rows have a year; as_of becomes Dec 31 of that year so weather features match.
            df["as_of"] = df["year"].apply(lambda y: date(int(y), 12, 31).isoformat())
        else:
            df["as_of"] = as_of.isoformat()
        keep = ["station_id", "latitude", "longitude", "as_of", "iwsr", "label"]
        for c in ("year", "iwsr_lower", "iwsr_upper", "tilt_deg", "months_in_data_set",
                  "mounting", "measurement_type"):
            if c in df.columns:
                keep.append(c)
        return df[keep]
    if source == "kimber_proxy":
        logger.info("Using kimber_proxy labels on %d seed stations", len(DEFAULT_CA_STATIONS))
        return kimber_synthetic_labels(
            DEFAULT_CA_STATIONS,
            as_of=as_of,
            lookback_days=int(region_cfg.get("weather_history_days", 180)),
            iwsr_risk_threshold=float(region_cfg["iwsr_risk_threshold"]),
            cache_dir=repo_root / region_cfg.get("cache_dir", ".cache/soiling"),
        )
    raise ValueError(f"Unknown label_source: {source}")


def _station_features(
    labels: pd.DataFrame,
    feat_cfg: dict,
    region_cfg: dict,
    repo_root: Path,
) -> pd.DataFrame:
    panel_mode = "year" in labels.columns
    # Panel mode uses a year-long lookback ending Dec 31 of each label's year.
    # Summary mode keeps the original 180-day rolling window from today.
    default_lookback = 365 if panel_mode else int(region_cfg.get("weather_history_days", 180))
    cache_dir = repo_root / region_cfg.get("cache_dir", ".cache/soiling")
    windows = tuple(feat_cfg["rolling_windows_days"])
    kimber_cfg = feat_cfg.get("kimber")
    static_path = region_cfg.get("static_features_csv")
    static_lookup = load_static_lookup(repo_root / static_path) if static_path else None

    covariates_cfg = feat_cfg.get("nrel_covariates", {}) or {}
    cov_numeric = [c for c in covariates_cfg.get("numeric", []) if c in labels.columns]
    cov_categorical = [c for c in covariates_cfg.get("categorical", []) if c in labels.columns]

    n_total = len(labels)
    rows = []
    for i, (_, s) in enumerate(labels.iterrows(), start=1):
        as_of_row = date.fromisoformat(s["as_of"])
        start_row = _window_start(as_of_row, default_lookback)
        try:
            daily = fetch_combined(s["latitude"], s["longitude"], start_row, as_of_row, cache_dir=cache_dir)
            loc = location_feature_vector(
                s["latitude"], s["longitude"],
                cache_dir=cache_dir, static_lookup=static_lookup,
            )
        except Exception as exc:
            logger.warning("Skipping %s%s: %s",
                           s["station_id"],
                           f" ({as_of_row.year})" if panel_mode else "",
                           exc)
            continue
        row = {
            "station_id": s["station_id"],
            "latitude": float(s["latitude"]),
            "longitude": float(s["longitude"]),
            "label": int(s["label"]),
            "iwsr": float(s["iwsr"]),
        }
        if panel_mode:
            row["year"] = int(s["year"])
        for c in ("iwsr_lower", "iwsr_upper"):
            if c in labels.columns:
                row[c] = s.get(c)
        for c in cov_numeric:
            row[c] = s.get(c)
        for c in cov_categorical:
            row[c] = s.get(c)
        row.update(loc)
        row.update(build_feature_row(daily, as_of_row, windows=windows, kimber_cfg=kimber_cfg))
        rows.append(row)
        if panel_mode and i % 50 == 0:
            logger.info("Built features for %d / %d panel rows", i, n_total)

    df = pd.DataFrame(rows)
    # One-hot encode categoricals to match the XGBoost numeric input.
    for c in cov_categorical:
        if c in df.columns:
            dummies = pd.get_dummies(df[c].fillna("unknown"), prefix=c, dtype=float)
            df = pd.concat([df.drop(columns=[c]), dummies], axis=1)
    return df


def _log_aq_coverage(features: pd.DataFrame) -> None:
    aq_cols = [c for c in features.columns if c.startswith(("pm2_5_", "pm10_"))]
    if not aq_cols:
        return
    coverage = features[aq_cols].notna().mean().sort_values()
    logger.info(
        "AQ feature coverage: %s",
        ", ".join(f"{col}={frac:.1%}" for col, frac in coverage.items()),
    )
    if "year" in features.columns:
        by_year = features.groupby("year")[aq_cols].apply(lambda g: float(g.notna().mean().mean()))
        logger.info("AQ mean coverage by year:\n%s", by_year.to_string())
    if coverage.max() < 0.5:
        logger.warning(
            "AQ features are sparse across the training matrix (<50%% non-null). "
            "Historical Open-Meteo AQ backfill may be incomplete for this dataset."
        )


def _drop_sparse_features(features: pd.DataFrame, min_non_null_fraction: float | None) -> tuple[pd.DataFrame, dict[str, float]]:
    if min_non_null_fraction is None:
        return features, {}
    meta_cols = {"station_id", "latitude", "longitude", "label", "iwsr",
                 "iwsr_lower", "iwsr_upper", "as_of", "year"}
    feature_cols = [c for c in features.columns if c not in meta_cols]
    coverage = features[feature_cols].notna().mean().sort_values()
    sparse = coverage[coverage < float(min_non_null_fraction)]
    if sparse.empty:
        return features, {}
    logger.warning(
        "Dropping %d sparse features with <%.0f%% coverage: %s",
        len(sparse), 100.0 * float(min_non_null_fraction), ", ".join(sparse.index.tolist()),
    )
    return features.drop(columns=sparse.index.tolist()), {str(k): float(v) for k, v in sparse.items()}


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    region_cfg = yaml.safe_load((repo_root / args.config).read_text())
    feat_cfg = yaml.safe_load((repo_root / args.features_config).read_text())
    model_cfg = yaml.safe_load((repo_root / args.model_config).read_text())

    labels = _load_labels(region_cfg, repo_root)
    if labels.empty:
        raise RuntimeError("No labels produced — check network access and label source")
    logger.info("Label distribution: %s", labels["label"].value_counts().to_dict())

    features = _station_features(labels, feat_cfg, region_cfg, repo_root)
    _log_aq_coverage(features)
    features, dropped_sparse = _drop_sparse_features(
        features,
        (feat_cfg.get("feature_quality", {}) or {}).get("min_non_null_fraction"),
    )
    out_train = repo_root / "outputs" / "soiling" / "training_matrix.parquet"
    out_train.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_train, index=False)
    logger.info("Wrote training matrix → %s (%d rows)", out_train, len(features))

    if args.dry_run:
        logger.info("--dry-run: skipping training")
        return

    meta_cols = {"station_id", "latitude", "longitude", "label", "iwsr",
                 "iwsr_lower", "iwsr_upper", "as_of", "year"}
    feature_cols = [c for c in features.columns if c not in meta_cols]
    X = features[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = features["label"].values.astype(int)
    iwsr = features["iwsr"].values.astype(float)

    sample_weight = _compute_sample_weights(features, model_cfg.get("sample_weights", {}))

    target_mode = model_cfg.get("target_mode", "binary")
    cv_cfg = SpatialCVConfig(**model_cfg["spatial_cv"])
    iwsr_thr = float(region_cfg["iwsr_risk_threshold"])

    holdout_year = args.holdout_year
    holdout_mask = None
    if holdout_year is not None:
        if "year" not in features.columns:
            raise ValueError("--holdout-year requires panel mode (label_source: nrel_panel)")
        holdout_mask = (features["year"].values == holdout_year)
        n_hold = int(holdout_mask.sum())
        if n_hold == 0:
            raise ValueError(f"No rows with year={holdout_year} in training matrix")
        logger.info("Temporal holdout: training on %d rows, evaluating on %d (%d)",
                    int((~holdout_mask).sum()), n_hold, holdout_year)

    if holdout_mask is not None:
        X_train = X.loc[~holdout_mask].reset_index(drop=True)
        X_test = X.loc[holdout_mask].reset_index(drop=True)
        y_train, y_test = y[~holdout_mask], y[holdout_mask]
        iwsr_train = iwsr[~holdout_mask] if target_mode == "regression" else None
        sw_train = sample_weight[~holdout_mask] if sample_weight is not None else None
        feats_train = features.loc[~holdout_mask].reset_index(drop=True)
        model, metrics, calibrator = train_risk_model(
            X_train, y_train,
            lats=feats_train["latitude"].values, lons=feats_train["longitude"].values,
            xgb_params=model_cfg["xgboost"], cv=cv_cfg,
            target_mode=target_mode, iwsr_target=iwsr_train,
            sample_weight=sw_train, iwsr_risk_threshold=iwsr_thr,
        )
        # Score the held-out year.
        from sklearn.metrics import roc_auc_score, average_precision_score
        X_test_imputed = impute_with_feature_medians(X_test, getattr(model, "feature_medians_", None))
        if target_mode == "binary":
            scores_h = model.predict_proba(X_test_imputed)[:, 1]
        else:
            scores_h = -model.predict(X_test_imputed)
        if len(np.unique(y_test)) > 1:
            metrics["holdout_year"] = holdout_year
            metrics["holdout_auc"] = float(roc_auc_score(y_test, scores_h))
            metrics["holdout_ap"] = float(average_precision_score(y_test, scores_h))
            metrics["holdout_n"] = int(len(y_test))
        else:
            logger.warning("Holdout year %d has single-class y — skipping holdout AUC", holdout_year)
    else:
        model, metrics, calibrator = train_risk_model(
            X,
            y,
            lats=features["latitude"].values,
            lons=features["longitude"].values,
            xgb_params=model_cfg["xgboost"],
            cv=cv_cfg,
            target_mode=target_mode,
            iwsr_target=iwsr if target_mode == "regression" else None,
            sample_weight=sample_weight,
            iwsr_risk_threshold=iwsr_thr,
        )
    if dropped_sparse:
        metrics["dropped_sparse_features"] = dropped_sparse
    logger.info("Spatial CV metrics: %s", metrics)

    run_name = args.run_name or datetime.utcnow().strftime("soiling_%Y%m%d_%H%M%S")
    out_dir = repo_root / "runs" / "soiling" / run_name
    save_model(model, out_dir, feature_names=feature_cols, metrics=metrics, calibrator=calibrator)
    logger.info("Model saved → %s%s", out_dir, " (with calibrator)" if calibrator else "")

    cv_auc = float(metrics.get("mean_auc", 0.0))
    holdout_auc = float(metrics.get("holdout_auc", 0.0)) if "holdout_auc" in metrics else None
    manifest_metrics: dict[str, float] = {
        "cv_auc": cv_auc,
        "n_train": int(len(features) - (int(holdout_mask.sum()) if holdout_mask is not None else 0)),
    }
    if holdout_auc is not None:
        manifest_metrics["holdout_auc"] = holdout_auc
        manifest_metrics["holdout_year"] = int(metrics.get("holdout_year", -1))
        manifest_metrics["n_holdout"] = int(metrics.get("holdout_n", 0))
    write_manifest(
        out_dir,
        stage="stage2_train",
        model_version=run_name,
        model_weights=out_dir / "model.ubj",
        inputs=[str(out_train)],
        beta=cv_auc < 0.70 and (holdout_auc is None or holdout_auc < 0.70),
        metrics=manifest_metrics,
        known_limitations=[
            "AQ features sparse pre-2022; PM features dropped via min_non_null_fraction gate",
            "Below 0.70 AUC GA bar",
        ],
        extra={
            "label_source": region_cfg.get("label_source", "kimber_proxy"),
            "calibrator": calibrator is not None,
        },
    )

    min_auc = model_cfg.get("acceptance", {}).get("min_mean_auc")
    if min_auc is not None and metrics.get("mean_auc", 0) < min_auc:
        logger.warning("Mean CV AUC %.3f below acceptance %.2f — iterate features/labels before shipping",
                       metrics.get("mean_auc", float("nan")), min_auc)


def _compute_sample_weights(features: pd.DataFrame, cfg: dict) -> np.ndarray | None:
    mode = cfg.get("mode", "uniform")
    if mode == "uniform":
        return None
    if mode != "iwsr_ci_width":
        raise ValueError(f"Unknown sample_weights.mode: {mode}")
    if "iwsr_lower" not in features.columns or "iwsr_upper" not in features.columns:
        logger.warning("sample_weights=iwsr_ci_width requested but CI columns missing; using uniform")
        return None
    width = (features["iwsr_upper"] - features["iwsr_lower"]).astype(float)
    width = width.where(width > 0).fillna(width[width > 0].median() if (width > 0).any() else 0.01)
    w = 1.0 / np.maximum(width.values, 1e-3)
    lo_pct, hi_pct = cfg.get("clip_percentiles", [5, 95])
    lo, hi = np.percentile(w, [lo_pct, hi_pct])
    return np.clip(w, lo, hi)


if __name__ == "__main__":
    main()
