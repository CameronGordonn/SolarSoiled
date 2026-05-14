#!/usr/bin/env python3
"""Convert YOLO segmentation label outputs to a GeoJSON polygon file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
import geojson

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solarsoiled.manifest import write_manifest


DEFAULT_LABELS_DIR = REPO_ROOT / "runs" / "segment" / "predict" / "labels"
DEFAULT_TILE_INDEX = REPO_ROOT / "data" / "interim" / "tile_index.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "solar_arrays.geojson"


def export_polygons(
    labels_dir: Path,
    tile_index_path: Path,
    output_geojson: Path,
    *,
    write_manifest_sibling: bool = True,
) -> Path:
    """Convert YOLO segmentation labels into a merged GeoJSON polygon file.

    Returns the path written. Writes a sibling ``manifest.json`` next to
    ``output_geojson`` when ``write_manifest_sibling`` is true.
    """
    labels_dir = Path(labels_dir)
    tile_index_path = Path(tile_index_path)
    output_geojson = Path(output_geojson)

    with open(tile_index_path) as f:
        raw = json.load(f)
    tile_index = raw.get("tiles", raw)

    polygons = []
    for label_file in labels_dir.glob("*.txt"):
        tile_name = label_file.stem + ".png"
        if tile_name not in tile_index:
            continue
        meta = tile_index[tile_name]
        transform = rasterio.Affine.from_gdal(*meta["transform"])
        w, h = meta["width"], meta["height"]

        for line in label_file.read_text().splitlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            # YOLO native save_txt has trailing conf (even total fields).
            # SAHI export in 04_infer omits conf (odd total fields).
            if len(parts) % 2 == 0:
                confidence = float(parts[-1])
                coords = list(map(float, parts[1:-1]))
            else:
                confidence = 1.0
                coords = list(map(float, parts[1:]))
            pts = [(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords), 2)]
            poly = Polygon([transform * pt for pt in pts])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.area > 1.0:
                polygons.append({"geometry": poly, "class_id": class_id, "confidence": confidence})

    if not polygons:
        print("No polygons found — exiting")
        return output_geojson

    # Determine CRS from the first polygon's tile entry (all tiles share one CRS)
    tile_crs = list(tile_index.values())[0].get("crs", "EPSG:3857")

    merged = unary_union([p["geometry"] for p in polygons])
    geoms = [merged] if merged.geom_type == "Polygon" else list(merged.geoms)

    # Reproject to WGS84 (GeoJSON spec) so downstream scripts see valid lat/lon
    gdf = gpd.GeoDataFrame(
        [{"id": i, "class_id": 0, "class_name": "solar_array"} for i in range(len(geoms))],
        geometry=geoms,
        crs=tile_crs,
    ).to_crs("EPSG:4326")

    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_geojson, driver="GeoJSON")
    print(f"GeoJSON written to: {output_geojson} ({len(gdf)} polygons, WGS84)")

    if write_manifest_sibling:
        upstream_manifest = labels_dir.parent / "manifest.json"
        upstream_weights = None
        upstream_version = "stage1-unknown"
        if upstream_manifest.is_file():
            try:
                up = json.loads(upstream_manifest.read_text())
                upstream_weights = up.get("model_weights_path")
                upstream_version = up.get("model_version", upstream_version)
            except json.JSONDecodeError:
                pass

        label_files = sorted(labels_dir.glob("*.txt"))
        write_manifest(
            output_geojson.parent,
            stage="stage1_detect",
            model_version=upstream_version,
            model_weights=upstream_weights,
            inputs=[str(p) for p in label_files] + [str(tile_index_path)],
            metrics={"n_polygons": len(gdf), "n_label_files": len(label_files)},
            known_limitations=["Polygons merged via unary_union; per-detection confidence dropped"],
            extra={"output_geojson": str(output_geojson)},
        )

    return output_geojson


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--tile-index", type=Path, default=DEFAULT_TILE_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    export_polygons(args.labels_dir, args.tile_index, args.output)


if __name__ == "__main__":
    main()
