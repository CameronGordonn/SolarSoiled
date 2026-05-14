"""Extract per-array geometric and spatial features from outputs/solar_arrays.geojson."""

from __future__ import annotations

import argparse
from pathlib import Path
import math

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon


TARGET_CRS = "EPSG:32610"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "outputs" / "solar_arrays.geojson"
DEFAULT_OUT_TABLE = REPO_ROOT / "outputs" / "array_features.parquet"
DEFAULT_OUT_GEO = REPO_ROOT / "outputs" / "array_features.geo.parquet"


def _ensure_valid(geom):
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = geom.buffer(0)
    return None if geom.is_empty else geom


def _compactness(area: float, perimeter: float) -> float:
    return (4.0 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else float("nan")


def _bbox_aspect_ratio(geom) -> float:
    minx, miny, maxx, maxy = geom.bounds
    w, h = maxx - minx, maxy - miny
    return max(w, h) / min(w, h) if w > 0 and h > 0 else float("nan")


def _convex_hull_ratio(area: float, geom) -> float:
    hull_area = geom.convex_hull.area if geom.convex_hull else 0.0
    return area / hull_area if hull_area > 0 else float("nan")


def _principal_axis_orientation_deg(geom) -> float:
    """PCA on convex hull exterior coords → principal axis azimuth in [0, 180)."""
    g = geom.convex_hull
    if g is None or g.is_empty:
        return float("nan")
    if isinstance(g, Polygon):
        coords = np.asarray(g.exterior.coords)
    elif isinstance(g, MultiPolygon):
        polys = list(g.geoms)
        coords = np.asarray(max(polys, key=lambda x: x.area).exterior.coords) if polys else None
        if coords is None:
            return float("nan")
    else:
        return float("nan")
    if coords.shape[0] < 3:
        return float("nan")
    xy = coords[:, :2] - coords[:, :2].mean(axis=0, keepdims=True)
    vals, vecs = np.linalg.eigh(np.cov(xy.T))
    v = vecs[:, np.argmax(vals)]
    return math.degrees(math.atan2(v[1], v[0])) % 180.0


def _neighbor_features(gdf: gpd.GeoDataFrame, radii_m=(50.0, 100.0)) -> pd.DataFrame:
    """Neighbor counts within radii, nearest-neighbor distance, mean neighbor area at 100m."""
    sindex = gdf.sindex
    geoms = gdf.geometry.values
    areas = gdf["area_m2"].values
    r_max = max(radii_m)

    neighbor_counts = {r: np.zeros(len(gdf), dtype=int) for r in radii_m}
    nn_dist = np.full(len(gdf), np.nan)
    mean_nb_area_100m = np.full(len(gdf), np.nan)

    for i, geom in enumerate(geoms):
        if geom is None or geom.is_empty:
            continue
        cand_idx = [j for j in sindex.intersection(geom.buffer(r_max).bounds) if j != i]
        if not cand_idx:
            continue
        dists, areas_100 = [], []
        for j in cand_idx:
            g2 = geoms[j]
            if g2 is None or g2.is_empty:
                continue
            d = geom.distance(g2)
            dists.append(d)
            if d <= 100.0:
                areas_100.append(areas[j])
            for r in radii_m:
                if d <= r:
                    neighbor_counts[r][i] += 1
        if dists:
            nn_dist[i] = float(np.min(dists))
        if areas_100:
            mean_nb_area_100m[i] = float(np.mean(areas_100))

    out = {"distance_to_nearest_array_m": nn_dist, "mean_neighbor_area_100m": mean_nb_area_100m}
    for r in radii_m:
        out[f"neighbor_count_{int(r)}m"] = neighbor_counts[r]
    return pd.DataFrame(out)


def extract_features(
    input_geojson: Path,
    out_table: Path,
    out_geo: Path,
) -> None:
    """Extract per-array geometric/spatial features → parquet pair.

    Reads ``input_geojson`` polygons, reprojects to ``TARGET_CRS``, computes
    geometry + neighbor features, and writes both a tabular parquet and a
    geo-parquet alongside it.
    """
    input_geojson = Path(input_geojson)
    out_table = Path(out_table)
    out_geo = Path(out_geo)

    if not input_geojson.exists():
        raise FileNotFoundError(f"Missing input GeoJSON: {input_geojson}")

    gdf = gpd.read_file(input_geojson)
    if gdf.empty:
        raise ValueError(f"No features in: {input_geojson}")

    if gdf.crs is None:
        gdf = gdf.set_crs(TARGET_CRS)
    elif gdf.crs.to_string() != TARGET_CRS:
        gdf = gdf.to_crs(TARGET_CRS)

    gdf["geometry"] = gdf["geometry"].apply(_ensure_valid)
    gdf = gdf[~gdf["geometry"].isna() & ~gdf.geometry.is_empty].reset_index(drop=True)
    gdf["array_id"] = np.arange(len(gdf), dtype=int)
    if "class_id" not in gdf.columns:
        gdf["class_id"] = 0
    if "class_name" not in gdf.columns:
        gdf["class_name"] = "solar_array"

    gdf["area_m2"] = gdf.geometry.area.astype(float)
    gdf["perimeter_m"] = gdf.geometry.length.astype(float)
    gdf["compactness"] = [_compactness(a, p) for a, p in zip(gdf["area_m2"], gdf["perimeter_m"])]
    gdf["bbox_aspect_ratio"] = gdf.geometry.apply(_bbox_aspect_ratio).astype(float)
    gdf["convex_hull_ratio"] = [_convex_hull_ratio(a, g) for a, g in zip(gdf["area_m2"], gdf.geometry)]
    gdf["orientation_deg"] = gdf.geometry.apply(_principal_axis_orientation_deg).astype(float)

    ctx = _neighbor_features(gdf)
    for col in ctx.columns:
        gdf[col] = ctx[col].values

    feature_cols = [
        "array_id", "class_id", "class_name", "area_m2", "perimeter_m", "compactness",
        "bbox_aspect_ratio", "convex_hull_ratio", "orientation_deg",
        "neighbor_count_50m", "neighbor_count_100m", "distance_to_nearest_array_m", "mean_neighbor_area_100m",
    ]

    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_geo.parent.mkdir(parents=True, exist_ok=True)
    gdf[feature_cols + ["geometry"]].to_parquet(out_geo, index=False)
    pd.DataFrame(gdf[feature_cols]).to_parquet(out_table, index=False)
    print(f"Table: {out_table}\nGeoTable: {out_geo}\nCRS: {gdf.crs}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-table", type=Path, default=DEFAULT_OUT_TABLE)
    parser.add_argument("--out-geo", type=Path, default=DEFAULT_OUT_GEO)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    extract_features(args.input, args.out_table, args.out_geo)


if __name__ == "__main__":
    main()
