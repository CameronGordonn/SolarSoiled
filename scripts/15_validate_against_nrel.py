"""Validate a trained soiling risk model against measured NREL IWSR values.

NREL stations are held out of training entirely. This script scores them with
the trained model and computes Spearman rank correlation between predicted risk
and measured IWSR. Because risk score and IWSR are inversely related (high risk
= low IWSR), the expected correlation is negative; the script reports |r|.

Usage:
    PYTHONPATH=. python scripts/15_validate_against_nrel.py \
        --model runs/soiling/<run>/model.ubj \
        --run-name <run_name>

Output: outputs/eval/nrel_validation_<run_name>.csv
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from src.soiling.feature_engineering import build_feature_row
from src.soiling.location_features import load_static_lookup, location_feature_vector
from src.soiling.risk_model import (
    impute_with_feature_medians,
    load_calibrator,
    load_feature_medians,
    load_model,
    predict_risk,
)
from src.soiling.weather_client import fetch_combined

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Climate zone assignment by longitude/latitude for per-zone breakdown.
# Rough buckets — coastal west of -120, desert east of -116, valley in between.
def _climate_zone(lat: float, lon: float) -> str:
    if lon < -120.0:
        return "coastal"
    if lon > -116.0 or (lat < 34.5 and lon > -117.5):
        return "desert"
    return "valley"


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to runs/soiling/<run>/model.ubj")
    p.add_argument("--run-name", required=True, help="Label for output file")
    p.add_argument("--nrel-csv", default="data/external/nrel_soiling_map_annual.csv",
                   help="Per-(station, year) NREL panel CSV")
    p.add_argument("--config", default="configs/soiling/california.yaml")
    p.add_argument("--features-config", default="configs/soiling/features.yaml")
    p.add_argument("--lookback-days", type=int, default=365)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]

    nrel_path = repo_root / args.nrel_csv
    if not nrel_path.exists():
        raise FileNotFoundError(
            f"NREL annual CSV not found at {nrel_path}. "
            "Run scripts/12_ingest_nrel_soiling_map.py first."
        )

    region_cfg = yaml.safe_load((repo_root / args.config).read_text())
    feat_cfg = yaml.safe_load((repo_root / args.features_config).read_text())

    model_path = Path(args.model)
    model, feature_names = load_model(model_path)
    calibrator = load_calibrator(model_path.parent)
    feature_medians = load_feature_medians(model_path.parent)
    if feature_names is None:
        raise RuntimeError(f"No feature_names.json next to {args.model}")

    nrel = pd.read_csv(nrel_path)
    required = {"station_id", "latitude", "longitude", "iwsr", "year"}
    missing_cols = required - set(nrel.columns)
    if missing_cols:
        raise ValueError(f"NREL CSV missing columns: {missing_cols}")

    # Drop censored rows (iwsr reported as ">0.99") if present.
    if "iwsr_censored" in nrel.columns:
        before = len(nrel)
        nrel = nrel[~nrel["iwsr_censored"].astype(bool)].reset_index(drop=True)
        logger.info("Dropped %d censored rows; %d remain", before - len(nrel), len(nrel))

    cache_dir = repo_root / region_cfg.get("cache_dir", ".cache/soiling")
    static_path = region_cfg.get("static_features_csv")
    static_lookup = load_static_lookup(repo_root / static_path) if static_path else None
    windows = tuple(feat_cfg["rolling_windows_days"])
    kimber_cfg = feat_cfg.get("kimber")
    somosclean_cfg = feat_cfg.get("somosclean")

    rows = []
    n_total = len(nrel)
    for i, (_, s) in enumerate(nrel.iterrows(), start=1):
        year = int(s["year"])
        as_of = date(year, 12, 31)
        start = as_of - timedelta(days=args.lookback_days - 1)
        try:
            daily = fetch_combined(float(s["latitude"]), float(s["longitude"]), start, as_of, cache_dir=cache_dir)
            loc = location_feature_vector(float(s["latitude"]), float(s["longitude"]),
                                          cache_dir=cache_dir, static_lookup=static_lookup)
        except Exception as exc:
            logger.warning("Skipping %s (%d): %s", s["station_id"], year, exc)
            continue

        feat_row = build_feature_row(daily, as_of, windows=windows, kimber_cfg=kimber_cfg, somosclean_cfg=somosclean_cfg)
        feat_row.update(loc)
        feat_row["station_id"] = s["station_id"]
        feat_row["latitude"] = float(s["latitude"])
        feat_row["longitude"] = float(s["longitude"])
        feat_row["year"] = year
        feat_row["measured_iwsr"] = float(s["iwsr"])
        rows.append(feat_row)

        if i % 50 == 0:
            logger.info("Built features for %d / %d NREL rows", i, n_total)

    if not rows:
        raise RuntimeError("No NREL rows scored — check network access and CSV")

    df = pd.DataFrame(rows)
    meta_cols = {"station_id", "latitude", "longitude", "year", "measured_iwsr", "as_of"}
    feat_cols = [c for c in feature_names if c in df.columns]
    missing_feats = [c for c in feature_names if c not in df.columns]
    if missing_feats and feature_medians:
        logger.warning("Filling %d missing features from training medians: %s",
                       len(missing_feats), missing_feats[:10])
        for col in missing_feats:
            df[col] = feature_medians.get(col, float("nan"))
        feat_cols = feature_names

    X = df[feat_cols].copy()
    if feature_medians:
        X = impute_with_feature_medians(X, feature_medians)

    raw_scores = predict_risk(model, X, calibrator=None)
    calibrated_scores = predict_risk(model, X, calibrator=calibrator)
    df["predicted_risk_score"] = calibrated_scores
    df["predicted_risk_raw"] = raw_scores

    # Spearman correlation: risk score vs IWSR (expect negative — high risk = low IWSR)
    mask = df["measured_iwsr"].notna() & df["predicted_risk_score"].notna()
    r, pval = stats.spearmanr(df.loc[mask, "predicted_risk_score"], df.loc[mask, "measured_iwsr"])
    abs_r = abs(r)
    logger.info("Overall Spearman |r| = %.3f (p=%.4f, n=%d)", abs_r, pval, mask.sum())
    logger.info("Note: negative r expected (high risk score → low measured IWSR)")

    # Per climate zone breakdown
    df["climate_zone"] = df.apply(lambda row: _climate_zone(row["latitude"], row["longitude"]), axis=1)
    for zone in sorted(df["climate_zone"].unique()):
        sub = df[df["climate_zone"] == zone]
        z_mask = sub["measured_iwsr"].notna() & sub["predicted_risk_score"].notna()
        if z_mask.sum() < 3:
            continue
        zr, zp = stats.spearmanr(sub.loc[z_mask, "predicted_risk_score"], sub.loc[z_mask, "measured_iwsr"])
        logger.info("  %s: |r| = %.3f (n=%d)", zone, abs(zr), z_mask.sum())

    out_dir = repo_root / "outputs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"nrel_validation_{args.run_name}.csv"
    out_cols = ["station_id", "year", "latitude", "longitude", "climate_zone",
                "measured_iwsr", "predicted_risk_score", "predicted_risk_raw"]
    df[out_cols].to_csv(out_path, index=False)
    logger.info("Wrote validation results → %s (%d rows)", out_path, len(df))

    # Summary line for easy copy-paste into results tables
    gate = "PASS" if abs_r >= 0.40 else "BELOW_GATE"
    logger.info("Validation summary: run=%s  spearman_abs_r=%.3f  n=%d  gate=%s",
                args.run_name, abs_r, mask.sum(), gate)


if __name__ == "__main__":
    main()
