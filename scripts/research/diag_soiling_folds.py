"""Diagnose Stage 2 spatial CV folds for a saved soiling run.

Reproduces the spatial clustering + GroupKFold split used during training,
joins fold assignments back to stations, and reports per-fold composition
plus feature importance from the saved XGBoost model.

Usage:
  PYTHONPATH=. python scripts/research/diag_soiling_folds.py --run runs/soiling/run_f_holdout2022
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.soiling.risk_model import (  # noqa: E402
    _spatial_clusters,
    impute_with_feature_medians,
    load_feature_medians,
    load_model,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="outputs/soiling/training_matrix.parquet")
    ap.add_argument("--run", default="runs/soiling/run_f_holdout2022")
    ap.add_argument("--cluster-km", type=float, default=50.0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--holdout-year", type=int, default=2022,
                    help="Year removed during training (so we re-fold on the same train set)")
    args = ap.parse_args()

    from sklearn.model_selection import GroupKFold

    df = pd.read_parquet(args.matrix)
    print(f"loaded {len(df)} rows from {args.matrix}")

    if args.holdout_year is not None and "year" in df.columns:
        before = len(df)
        df = df[df["year"] != args.holdout_year].reset_index(drop=True)
        print(f"removed {before - len(df)} holdout-year ({args.holdout_year}) rows; "
              f"{len(df)} remain (training set)")

    clusters = _spatial_clusters(df["latitude"].values, df["longitude"].values, args.cluster_km)
    df["_cluster"] = clusters
    print(f"spatial clusters at {args.cluster_km}km: {df['_cluster'].nunique()} unique")

    feature_names = json.loads(Path(args.run, "feature_names.json").read_text())
    X = df[feature_names].apply(pd.to_numeric, errors="coerce")
    feature_medians = load_feature_medians(Path(args.run))
    if feature_medians is not None:
        X = impute_with_feature_medians(X, feature_medians)
    else:
        X = X.fillna(X.median(numeric_only=True))
    y = df["label"].values.astype(int)

    metrics = json.loads(Path(args.run, "metrics.json").read_text())
    fold_aucs = metrics.get("fold_aucs", [None] * args.n_folds)

    model, _ = load_model(Path(args.run, "model.ubj"))

    print("\n=== PER-FOLD BREAKDOWN ===")
    gkf = GroupKFold(n_splits=args.n_folds)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=clusters)):
        sub = df.iloc[te]
        stations = sub.groupby("station_id").size().sort_values(ascending=False)
        pos_rate = float(sub["label"].mean())
        n_pos = int(sub["label"].sum())
        n_neg = int((1 - sub["label"]).sum())
        lat_min, lat_max = sub["latitude"].min(), sub["latitude"].max()
        lon_min, lon_max = sub["longitude"].min(), sub["longitude"].max()
        auc_str = f"{fold_aucs[fold]:.3f}" if fold_aucs[fold] is not None else "n/a"
        print(f"\n--- Fold {fold} (CV AUC = {auc_str}) ---")
        print(f"  n_rows     = {len(sub)}")
        print(f"  n_stations = {sub['station_id'].nunique()}")
        print(f"  n_clusters = {sub['_cluster'].nunique()}")
        print(f"  pos_rate   = {pos_rate:.3f}  ({n_pos} pos / {n_neg} neg)")
        print(f"  lat range  = ({lat_min:.2f}, {lat_max:.2f})")
        print(f"  lon range  = ({lon_min:.2f}, {lon_max:.2f})")
        print(f"  top stations:")
        for sid, count in stations.head(8).items():
            print(f"    {sid:35s} {count:3d} rows")

    print("\n=== FEATURE IMPORTANCE (XGBoost gain) ===")
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(imp.head(15).to_string(index=False))
    print("\n  bottom features (low or zero signal):")
    print(imp.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
