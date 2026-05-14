"""Build per-array soiling feature matrix from detected arrays + weather history.

Joins onto outputs/array_features.geo.parquet (from script 07) on `array_id`.
Writes outputs/soiling/inference_matrix.parquet.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.soiling.feature_engineering import build_feature_row
from src.soiling.location_features import load_static_lookup, location_feature_vector
from src.soiling.weather_client import fetch_combined
from solarsoiled.manifest import write_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _window_start(as_of: date, lookback_days: int) -> date:
    """Inclusive lookback: 180 days ending today includes today as day 180."""
    return as_of - timedelta(days=max(lookback_days - 1, 0))


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/soiling/california.yaml")
    p.add_argument("--features-config", default="configs/soiling/features.yaml")
    p.add_argument("--arrays", default="outputs/array_features.geo.parquet")
    p.add_argument("--as-of", default=None, help="YYYY-MM-DD; default = today UTC")
    p.add_argument("--out", default="outputs/soiling/inference_matrix.parquet")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]

    region_cfg = yaml.safe_load((repo_root / args.config).read_text())
    feat_cfg = yaml.safe_load((repo_root / args.features_config).read_text())

    arrays_path = repo_root / args.arrays
    if not arrays_path.exists():
        raise FileNotFoundError(
            f"{arrays_path} missing — run scripts/07_extract_array_features.py first."
        )
    gdf = gpd.read_parquet(arrays_path)
    if gdf.empty:
        raise ValueError("array_features.geo.parquet has no rows")

    # Reproject to WGS84 for lat/lon-keyed weather lookup.
    gdf_ll = gdf.to_crs("EPSG:4326")
    centroids_ll = gdf_ll.geometry.centroid

    as_of = date.fromisoformat(args.as_of) if args.as_of else datetime.utcnow().date()
    lookback = int(region_cfg.get("weather_history_days", 180))
    start = _window_start(as_of, lookback)
    cache_dir = repo_root / region_cfg.get("cache_dir", ".cache/soiling")
    nlcd_path = region_cfg.get("nlcd_path")
    nlcd_path = Path(nlcd_path) if nlcd_path else None

    windows = tuple(feat_cfg["rolling_windows_days"])
    kimber_cfg = feat_cfg.get("kimber")
    geom_cols = [c for c in feat_cfg["geometric_features"] if c in gdf.columns]
    static_path = region_cfg.get("static_features_csv")
    static_lookup = load_static_lookup(repo_root / static_path) if static_path else None

    rows = []
    for i, (_, arr) in enumerate(gdf.iterrows()):
        lon = float(centroids_ll.iloc[i].x)
        lat = float(centroids_ll.iloc[i].y)
        try:
            daily = fetch_combined(lat, lon, start, as_of, cache_dir=cache_dir)
            loc = location_feature_vector(
                lat, lon, nlcd_path=nlcd_path,
                cache_dir=cache_dir, static_lookup=static_lookup,
            )
        except Exception as exc:
            logger.warning("array_id=%s weather/loc fetch failed: %s", arr["array_id"], exc)
            continue

        row = {"array_id": int(arr["array_id"]), "latitude": lat, "longitude": lon}
        row.update({c: arr[c] for c in geom_cols})
        row.update(loc)
        row.update(build_feature_row(daily, as_of, windows=windows, kimber_cfg=kimber_cfg))
        rows.append(row)

        if (i + 1) % 50 == 0:
            logger.info("Built features for %d / %d arrays", i + 1, len(gdf))

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    logger.info("Wrote %d rows → %s", len(rows), out_path)

    write_manifest(
        out_path.parent,
        stage="stage2_score",
        model_version="features-v1",
        model_weights=None,
        inputs=[str(arrays_path)],
        metrics={"n_arrays_in": int(len(gdf)), "n_rows_out": int(len(rows))},
        known_limitations=[
            "Open-Meteo + AQ daily aggregation; missing days dropped",
            "Static features (NLCD/elevation/OSM) skipped if static_features_csv absent",
        ],
        extra={
            "as_of": as_of.isoformat(),
            "lookback_days": int(lookback),
            "rolling_windows_days": list(windows),
            "out_matrix": str(out_path),
        },
    )


if __name__ == "__main__":
    main()
