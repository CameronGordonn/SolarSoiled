"""Pre-compute static (non-time-varying) per-station features:
elevation, ESA WorldCover land-cover class, distance to highway, distance to
agriculture. Cache to data/external/static_features.csv keyed on station_id.

Per-row training/inference then becomes a parquet lookup instead of N live API
calls — keeps weather as the only network-bound step.

Usage:
    python scripts/13_build_static_features.py                       # all NREL stations
    python scripts/13_build_static_features.py --input <csv>         # any (station_id, lat, lon) CSV
    python scripts/13_build_static_features.py --no-worldcover       # skip WorldCover (slow first run)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.soiling.location_features import (
    distance_to_agriculture_m,
    distance_to_highway_m,
    fetch_elevation,
)

logger = logging.getLogger(__name__)

# WorldCover class buckets relevant to soiling. See:
# https://esa-worldcover.org/en/data-access — class legend.
WORLDCOVER_BUCKETS = {
    10: "tree",
    20: "shrub",
    30: "grass",
    40: "cropland",
    50: "built_up",
    60: "bare",
    70: "snow_ice",
    80: "water",
    90: "wetland",
    95: "mangrove",
    100: "moss",
}


def sample_worldcover(lat: float, lon: float) -> Optional[int]:
    """Sample ESA WorldCover 2021 v200 at (lat, lon) via Microsoft Planetary
    Computer. Returns None if PC is unreachable. PC asset URLs are signed so
    this can't be requests-cached — call once per unique location.
    """
    try:
        import pystac_client
        import planetary_computer
        import rasterio
    except ImportError:
        logger.warning("pystac_client/planetary_computer/rasterio missing; skipping WorldCover")
        return None

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["esa-worldcover"],
        bbox=[lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001],
    )
    items = [it for it in search.items() if "v200" in it.id]
    if not items:
        return None
    asset = items[0].assets.get("map")
    if asset is None:
        return None
    try:
        with rasterio.open(asset.href) as src:
            for v in src.sample([(lon, lat)]):
                return int(v[0])
    except Exception as exc:
        logger.warning("WorldCover sample failed at (%.4f, %.4f): %s", lat, lon, exc)
    return None


def build_lookup(df: pd.DataFrame, cache_dir: Path, do_worldcover: bool) -> pd.DataFrame:
    rows = []
    n = len(df)
    for i, (_, s) in enumerate(df.iterrows(), start=1):
        sid = s["station_id"]
        lat = float(s["latitude"])
        lon = float(s["longitude"])
        elev = wc = hwy = ag = None
        try:
            elev = fetch_elevation(lat, lon, cache_dir=cache_dir)
        except Exception as exc:
            logger.warning("%s elevation failed: %s", sid, exc)
        if do_worldcover:
            wc = sample_worldcover(lat, lon)
        try:
            hwy = distance_to_highway_m(lat, lon, cache_dir=cache_dir)
        except Exception as exc:
            logger.warning("%s highway dist failed: %s", sid, exc)
        try:
            ag = distance_to_agriculture_m(lat, lon, cache_dir=cache_dir)
        except Exception as exc:
            logger.warning("%s ag dist failed: %s", sid, exc)
        rows.append({
            "station_id": sid,
            "latitude": lat,
            "longitude": lon,
            "elevation_m": elev,
            "worldcover_class": wc,
            "worldcover_bucket": WORLDCOVER_BUCKETS.get(wc, "unknown") if wc is not None else None,
            "distance_to_highway_m": hwy,
            "distance_to_agriculture_m": ag,
        })
        if i % 25 == 0 or i == n:
            logger.info("Processed %d / %d stations", i, n)
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=repo_root / "data/external/nrel_soiling_map.csv")
    parser.add_argument("--output", type=Path, default=repo_root / "data/external/static_features.csv")
    parser.add_argument("--cache-dir", type=Path, default=repo_root / ".cache/soiling")
    parser.add_argument("--no-worldcover", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df.drop_duplicates(subset=["station_id"]).reset_index(drop=True)
    logger.info("Building static features for %d unique stations", len(df))

    out = build_lookup(df, args.cache_dir, do_worldcover=not args.no_worldcover)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    logger.info("Wrote %d rows → %s", len(out), args.output)
    print(out.describe(include="all").T[["count", "mean", "min", "max"]])
    if "worldcover_bucket" in out.columns:
        print("\nWorldCover bucket counts:")
        print(out["worldcover_bucket"].value_counts())


if __name__ == "__main__":
    main()
