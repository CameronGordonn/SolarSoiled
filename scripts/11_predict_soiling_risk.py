"""Score detected arrays with a trained soiling risk model.

Input:  outputs/soiling/inference_matrix.parquet (from script 09)
        runs/soiling/<run_name>/model.ubj + feature_names.json
Output: outputs/soiling_risk.geojson (original polygons + risk_score column)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.soiling.risk_model import (
    impute_with_feature_medians,
    load_calibrator,
    load_feature_medians,
    load_model,
    predict_risk,
)
from solarsoiled.manifest import write_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to runs/soiling/<run>/model.ubj")
    p.add_argument("--features", default="outputs/soiling/inference_matrix.parquet")
    p.add_argument("--arrays", default="outputs/array_features.geo.parquet")
    p.add_argument("--out", default="outputs/soiling_risk.geojson")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]

    model, feature_names = load_model(Path(args.model))
    if feature_names is None:
        raise RuntimeError(f"No feature_names.json next to {args.model}")
    calibrator = load_calibrator(Path(args.model).parent)
    feature_medians = load_feature_medians(Path(args.model).parent)

    features = pd.read_parquet(repo_root / args.features)
    missing = [c for c in feature_names if c not in features.columns]
    if missing:
        if feature_medians is not None:
            # Fill columns that the training set had but inference set lacks using training medians.
            logger.warning("Inference matrix missing %d features; imputing from training medians: %s",
                           len(missing), missing[:10])
            for col in missing:
                features[col] = feature_medians.get(col, float("nan"))
        else:
            raise RuntimeError(f"Inference matrix missing features and no feature_medians.json: {missing[:10]}...")

    X = features[feature_names].apply(pd.to_numeric, errors="coerce")
    if feature_medians is not None:
        X = impute_with_feature_medians(X, feature_medians)
    else:
        logger.warning("feature_medians.json missing; falling back to inference-set medians")
        X = X.fillna(X.median(numeric_only=True))
    features["risk_score"] = predict_risk(model, X, calibrator=calibrator)
    logger.info("Calibrator: %s", "loaded" if calibrator is not None else "none (raw scores)")

    arrays = gpd.read_parquet(repo_root / args.arrays)
    enriched = arrays.merge(
        features[["array_id", "risk_score"]], on="array_id", how="left"
    )
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_file(out_path, driver="GeoJSON")
    logger.info("Wrote %d arrays with risk scores → %s", len(enriched), out_path)
    logger.info("risk_score stats: mean=%.3f, p90=%.3f, max=%.3f",
                enriched["risk_score"].mean(),
                enriched["risk_score"].quantile(0.9),
                enriched["risk_score"].max())

    model_path = Path(args.model)
    write_manifest(
        out_path.parent,
        stage="stage2_score",
        model_version=model_path.parent.name,
        model_weights=model_path,
        inputs=[str(repo_root / args.arrays), str(repo_root / args.features)],
        metrics={
            "n_arrays": int(len(enriched)),
            "risk_score_mean": float(enriched["risk_score"].mean()),
            "risk_score_p90": float(enriched["risk_score"].quantile(0.9)),
            "risk_score_max": float(enriched["risk_score"].max()),
        },
        known_limitations=[
            "AQ features sparse pre-2022; PM features dropped via min_non_null_fraction gate",
            "Below 0.70 AUC GA bar",
        ],
        extra={
            "calibrator": calibrator is not None,
            "output_geojson": str(out_path),
        },
    )


if __name__ == "__main__":
    main()
