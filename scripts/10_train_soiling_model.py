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
    somosclean_panel_labels,
    somosclean_synthetic_labels,
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
    p.add_argument("--feedback-dir", default=None,
                   help="Root of partner AOI outputs (default: outputs/aoi/). "
                        "Globs */feedback.json and joins lat/lon from */risk.geojson.")
    return p.parse_args()


def load_feedback_labels(feedback_dir: Path, iwsr_risk_threshold: float) -> pd.DataFrame:
    """Load partner clean-event records and convert to training rows.

    For each feedback.json found under feedback_dir/*/feedback.json:
      - Joins array centroid lat/lon from the sibling risk.geojson
      - Derives pseudo_iwsr = 1 - actual_recovery_pct/100
      - Applies iwsr_risk_threshold to produce a binary label
      - Sets as_of = cleaned_at so weather features match the soiling event

    Returns an empty DataFrame (not None) when no feedback files are found.
    """
    import glob
    import json as _json

    import geopandas as gpd

    pattern = str(feedback_dir / "*" / "feedback.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()

    all_rows: list[dict] = []
    for fpath in files:
        partner_id = Path(fpath).parent.name
        risk_path = Path(fpath).parent / "risk.geojson"
        if not risk_path.exists():
            logger.warning("feedback/%s: risk.geojson missing — skipping (run score first)", partner_id)
            continue

        with open(fpath) as fh:
            records = _json.load(fh)

        # Build array_id → (lat, lon) WGS84 centroid lookup via geopandas.
        gdf = gpd.read_file(risk_path)
        if gdf.crs and not gdf.crs.equals("EPSG:4326"):
            gdf = gdf.to_crs("EPSG:4326")
        centroid_map: dict[int, tuple[float, float]] = {}
        for _, row in gdf.iterrows():
            aid = row.get("array_id")
            if aid is None:
                continue
            c = row.geometry.centroid
            centroid_map[int(aid)] = (float(c.y), float(c.x))

        for rec in records:
            aid = int(rec.get("array_id", -1))
            if aid not in centroid_map:
                logger.warning(
                    "feedback/%s array_id=%d not found in risk.geojson — skipping",
                    partner_id, aid,
                )
                continue
            lat, lon = centroid_map[aid]
            recovery_pct = float(rec["actual_recovery_pct"])
            pseudo_iwsr = max(0.0, 1.0 - recovery_pct / 100.0)
            cleaned_at = rec["cleaned_at"]  # YYYY-MM-DD
            year = int(cleaned_at[:4])
            all_rows.append({
                "station_id": f"{partner_id}_array_{aid}",
                "latitude": lat,
                "longitude": lon,
                "as_of": cleaned_at,
                "iwsr": pseudo_iwsr,
                "label": int(pseudo_iwsr < iwsr_risk_threshold),
                "year": year,
                "is_feedback": True,
            })

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    logger.info(
        "Loaded %d feedback rows from %d partner(s) — label dist: %s",
        len(df), len(files), df["label"].value_counts().to_dict(),
    )
    return df


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
    if source == "somosclean":
        feat_cfg_path = repo_root / "configs/soiling/features.yaml"
        feat_cfg = yaml.safe_load(feat_cfg_path.read_text())
        sc_params = feat_cfg.get("somosclean", {})
        sc_kwargs = dict(
            iwsr_risk_threshold=float(region_cfg["iwsr_risk_threshold"]),
            cache_dir=repo_root / region_cfg.get("cache_dir", ".cache/soiling"),
            sl_sat=float(sc_params.get("sl_sat", 0.10)),
            k=float(sc_params.get("k", 30.0)),
            heavy_rain_mm=float(sc_params.get("heavy_rain_mm", 10.0)),
            rain_min_mm=float(sc_params.get("rain_min_mm", 1.0)),
            pm10_dust_threshold=float(sc_params.get("pm10_dust_threshold", 50.0)),
            pm10_dust_scale=float(sc_params.get("pm10_dust_scale", 0.02)),
        )
        nrel_annual = repo_root / region_cfg.get(
            "nrel_panel_csv_path", "data/external/nrel_soiling_map_annual.csv"
        )
        if nrel_annual.exists():
            logger.info("Using somosclean_panel_labels with NREL coordinates from %s", nrel_annual)
            bbox_cfg = region_cfg.get("bbox")
            return somosclean_panel_labels(
                nrel_csv_path=nrel_annual,
                bbox=tuple(bbox_cfg) if bbox_cfg else None,
                **sc_kwargs,
            )
        logger.info("NREL annual CSV not found — falling back to %d seed stations", len(DEFAULT_CA_STATIONS))
        return somosclean_synthetic_labels(
            DEFAULT_CA_STATIONS,
            as_of=as_of,
            lookback_days=int(region_cfg.get("weather_history_days", 365)),
            **sc_kwargs,
        )
    if source == "nrel_merged":
        # Panel rows (year-matched weather) + summary-only rows (today-centered weather).
        # Summary rows add geographic diversity (9 extra states) at the cost of imperfect
        # weather alignment; they are down-weighted in _compute_sample_weights via is_summary.
        bbox_cfg = region_cfg.get("bbox")
        bbox = tuple(bbox_cfg) if bbox_cfg else None
        iwsr_thr = float(region_cfg["iwsr_risk_threshold"])
        drop_censored = region_cfg.get("drop_censored", True)

        panel_path = repo_root / region_cfg.get("nrel_panel_csv_path", "data/external/nrel_soiling_map_annual.csv")
        panel = load_nrel_panel(panel_path, bbox=bbox)
        if drop_censored and "iwsr_censored" in panel.columns:
            before = len(panel)
            panel = panel[~panel["iwsr_censored"].astype(bool)].reset_index(drop=True)
            logger.info("Panel: dropped %d censored rows; %d remain", before - len(panel), len(panel))
        panel["label"] = (panel["iwsr"] < iwsr_thr).astype(int)
        panel["as_of"] = panel["year"].apply(lambda y: date(int(y), 12, 31).isoformat())
        panel["is_summary"] = False

        summary_path = repo_root / region_cfg.get("nrel_csv_path", "data/external/nrel_soiling_map.csv")
        summary_all = load_nrel_soiling_map(summary_path, bbox=bbox)
        panel_stations = set(panel["station_id"].unique())
        summary = summary_all[~summary_all["station_id"].isin(panel_stations)].reset_index(drop=True)
        # Summary-only stations are all censored (IWSR ">0.99" → coerced 0.995, label=0).
        # We keep them despite drop_censored: the label direction is known (not at risk),
        # and they add 9 states of geographic coverage absent from the panel set.
        # They are down-weighted via summary_weight_factor in _compute_sample_weights.
        n_censored_summary = int(summary["iwsr_censored"].astype(bool).sum()) if "iwsr_censored" in summary.columns else 0
        if n_censored_summary:
            logger.info(
                "Summary-only: keeping %d censored rows (all label=0; skipping drop_censored for geographic coverage)",
                n_censored_summary,
            )
        summary["label"] = (summary["iwsr"] < iwsr_thr).astype(int)
        summary["as_of"] = as_of.isoformat()
        summary["year"] = as_of.year
        summary["is_summary"] = True

        keep = ["station_id", "latitude", "longitude", "as_of", "iwsr", "label", "year", "is_summary"]
        for c in ("iwsr_lower", "iwsr_upper", "tilt_deg", "months_in_data_set", "mounting", "measurement_type"):
            if c in panel.columns or c in summary.columns:
                keep.append(c)
        merged = pd.concat([panel, summary], ignore_index=True)
        merged = merged[[c for c in keep if c in merged.columns]]
        logger.info(
            "nrel_merged: %d panel rows (%d stations) + %d summary-only rows (%d stations) = %d total",
            len(panel), panel["station_id"].nunique(),
            len(summary), summary["station_id"].nunique(),
            len(merged),
        )
        return merged
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
    somosclean_cfg = feat_cfg.get("somosclean")
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
            "is_feedback": bool(s.get("is_feedback", False)),
            "is_summary": bool(s.get("is_summary", False)),
            # Month of observation — captures seasonal soiling cycle.
            # lat/lon flow through as explicit features (not in meta_cols).
            "month_of_year": as_of_row.month,
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
        row.update(build_feature_row(daily, as_of_row, windows=windows, kimber_cfg=kimber_cfg, somosclean_cfg=somosclean_cfg))
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
    meta_cols = {"station_id", "label", "iwsr",
                 "iwsr_lower", "iwsr_upper", "as_of", "year", "is_feedback", "is_summary",
                 "months_in_data_set"}  # lat/lon included as features; months_in_data_set is NREL-only metadata
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

    feedback_dir = Path(args.feedback_dir) if args.feedback_dir else repo_root / "outputs" / "aoi"
    feedback_labels = load_feedback_labels(feedback_dir, float(region_cfg["iwsr_risk_threshold"]))
    if not feedback_labels.empty:
        labels = pd.concat([labels, feedback_labels], ignore_index=True)
        logger.info("Combined: %d NREL + %d feedback rows = %d total",
                    len(labels) - len(feedback_labels), len(feedback_labels), len(labels))

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

    meta_cols = {"station_id", "label", "iwsr",
                 "iwsr_lower", "iwsr_upper", "as_of", "year", "is_feedback", "is_summary",
                 "months_in_data_set"}  # lat/lon included as features; months_in_data_set is NREL-only metadata
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
    feedback_multiplier = float(cfg.get("feedback_weight_multiplier", 3.0))
    summary_factor = float(cfg.get("summary_weight_factor", 0.5))
    has_feedback = "is_feedback" in features.columns and features["is_feedback"].any()
    has_summary = "is_summary" in features.columns and features["is_summary"].any()

    if mode == "uniform":
        if not has_feedback and not has_summary:
            return None
        w = np.ones(len(features), dtype=float)
    elif mode == "iwsr_ci_width":
        if "iwsr_lower" not in features.columns or "iwsr_upper" not in features.columns:
            logger.warning("sample_weights=iwsr_ci_width requested but CI columns missing; using uniform")
            w = np.ones(len(features), dtype=float)
        else:
            width = (features["iwsr_upper"] - features["iwsr_lower"]).astype(float)
            width = width.where(width > 0).fillna(width[width > 0].median() if (width > 0).any() else 0.01)
            w = 1.0 / np.maximum(width.values, 1e-3)
            lo_pct, hi_pct = cfg.get("clip_percentiles", [5, 95])
            lo, hi = np.percentile(w, [lo_pct, hi_pct])
            w = np.clip(w, lo, hi)
    else:
        raise ValueError(f"Unknown sample_weights.mode: {mode}")

    if has_summary:
        sum_mask = features["is_summary"].fillna(False).values
        w[sum_mask] *= summary_factor
        logger.info(
            "Applied summary_weight_factor=%.2f to %d summary-only rows",
            summary_factor, int(sum_mask.sum()),
        )

    if has_feedback:
        fb_mask = features["is_feedback"].fillna(False).values
        w[fb_mask] *= feedback_multiplier
        logger.info(
            "Applied feedback_weight_multiplier=%.1f to %d feedback rows",
            feedback_multiplier, int(fb_mask.sum()),
        )

    return w


if __name__ == "__main__":
    main()
