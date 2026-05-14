"""Tile metadata and index management for tile_index.json."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from rasterio.transform import Affine

logger = logging.getLogger(__name__)


# Source filenames look like: naip_sc_2022_512_42.tif
# Pattern: naip_<region>_<vintage>_<size>_<scene>.tif
SOURCE_PATTERN = re.compile(r"naip_(?P<region>[a-z]+)_(?P<vintage>\d{4})_(?P<size>\d+)_(?P<scene>\d+)")

# Roboflow renames tiles to: tile_000042_png.rf.HASH.png
# Original tile_index key is the part before "_png.rf.": tile_000042.png
ROBOFLOW_SUFFIX_RE = re.compile(r"_png\.rf\.[A-Za-z0-9]+(?=\.[a-zA-Z]+$)")


def parse_source_filename(source: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """Extract (region, vintage_year) from a NAIP source filename.

    Returns (None, None) if the source is missing or doesn't match the expected pattern.
    """
    if not source:
        return None, None
    match = SOURCE_PATTERN.search(str(source))
    if not match:
        return None, None
    return match.group("region"), int(match.group("vintage"))


def strip_roboflow_suffix(name: str) -> str:
    """Reverse Roboflow's filename mangling: tile_000042_png.rf.HASH.jpg -> tile_000042.png.

    Handles both suffix variants Roboflow produces:
      - tile_000042_png.rf.HASH.png  (same extension)
      - tile_000042_png.rf.HASH.jpg  (converted to JPEG on re-export)
    Idempotent on already-clean names. Always normalizes to .png to match
    tile_index.json keys (which record the original PNG tile names).
    """
    stripped = ROBOFLOW_SUFFIX_RE.sub("", str(name))
    # Normalize extension to .png regardless of what Roboflow exported
    stem = stripped.rsplit(".", 1)[0] if "." in stripped else stripped
    return stem + ".png"


class TileIndex:
    """Manager for tile spatial metadata."""

    def __init__(self, tile_index_path: Optional[Path] = None):
        self.tile_index_path = Path(tile_index_path) if tile_index_path else None
        self.data: Dict[str, Dict[str, Any]] = {}
        if self.tile_index_path and self.tile_index_path.exists():
            self.load(self.tile_index_path)

    def add_tile(self, tile_name: str, transform: Affine, crs: str, width: int, height: int,
                 source_file: Optional[str] = None, bounds: Optional[Dict[str, float]] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        transform_gdal = list(transform.to_gdal()) if isinstance(transform, Affine) else list(transform)
        entry: Dict[str, Any] = {"transform": transform_gdal, "crs": crs, "width": width, "height": height}
        if source_file:
            entry["source"] = str(source_file)
        if bounds:
            entry["bounds"] = bounds
        if metadata:
            entry["metadata"] = metadata
        self.data[tile_name] = entry

    def get_tile(self, tile_name: str) -> Optional[Dict[str, Any]]:
        return self.data.get(tile_name)

    def get_transform(self, tile_name: str) -> Optional[Affine]:
        entry = self.get_tile(tile_name)
        if not entry or "transform" not in entry:
            return None
        return Affine(*entry["transform"])

    def get_crs(self, tile_name: str) -> Optional[str]:
        entry = self.get_tile(tile_name)
        return entry.get("crs") if entry else None

    def get_region(self, tile_name: str) -> Optional[str]:
        """Return the region code (e.g. 'sc' for Santa Cruz) parsed from the source filename."""
        entry = self.get_tile(tile_name)
        if not entry:
            return None
        if "region" in entry:
            return entry["region"]
        region, _ = parse_source_filename(entry.get("source"))
        return region

    def get_vintage(self, tile_name: str) -> Optional[int]:
        """Return the NAIP vintage year parsed from the source filename, or None if unknown."""
        entry = self.get_tile(tile_name)
        if not entry:
            return None
        if "vintage" in entry:
            return entry["vintage"]
        _, vintage = parse_source_filename(entry.get("source"))
        return vintage

    def lookup_by_stripped_name(self, raw_name: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Look up a tile entry given a raw filename, stripping any Roboflow suffix first.

        Returns (canonical_key, entry) — both None if not found.
        """
        candidate = strip_roboflow_suffix(raw_name)
        if candidate in self.data:
            return candidate, self.data[candidate]
        return None, None

    def validate(self) -> bool:
        if not self.data:
            raise ValueError("Tile index is empty")
        required_keys = {"transform", "crs", "width", "height"}
        for tile_name, entry in self.data.items():
            missing = required_keys - set(entry.keys())
            if missing:
                raise ValueError(f"Tile {tile_name} missing keys: {missing}")
            if not isinstance(entry["crs"], str) or ":" not in entry["crs"]:
                logger.warning(f"Tile {tile_name} has unusual CRS: {entry['crs']}")
            if not isinstance(entry["transform"], list) or len(entry["transform"]) != 6:
                raise ValueError(f"Tile {tile_name} transform invalid: {entry['transform']}")
            if entry["width"] <= 0 or entry["height"] <= 0:
                raise ValueError(f"Tile {tile_name} has invalid dimensions: {entry['width']}x{entry['height']}")
        logger.info(f"Tile index validation passed ({len(self.data)} tiles)")
        return True

    def to_geodataframe(self):
        """Convert tile index to a GeoPandas GeoDataFrame with polygon geometries."""
        try:
            import geopandas as gpd
            from shapely.geometry import box
        except ImportError:
            raise ImportError("geopandas and shapely are required")

        records = []
        for tile_name, info in self.data.items():
            bounds = info.get("bounds")
            if bounds:
                geom = box(bounds["minx"], bounds["miny"], bounds["maxx"], bounds["maxy"])
            else:
                transform = Affine(*info["transform"])
                minx, maxy = transform.c, transform.f
                maxx = minx + info["width"] * transform.a
                miny = maxy + info["height"] * transform.e
                geom = box(minx, miny, maxx, maxy)
            record = {"tile_name": tile_name, "geometry": geom}
            for k, v in info.items():
                if k != "bounds":
                    record[k] = v
            records.append(record)

        crs = records[0].get("crs") if records else None
        return gpd.GeoDataFrame(records, crs=crs)

    def save(self, output_path: Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.validate()
        index_data = {"tiles": self.data, "metadata": {
            "created": datetime.now().isoformat(), "total_tiles": len(self.data), "format_version": "1.0"
        }}
        with open(output_path, "w") as f:
            json.dump(index_data, f, indent=2)
        logger.info(f"Tile index saved to: {output_path}")

    def load(self, input_path: Path) -> None:
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Tile index not found: {input_path}")
        with open(input_path) as f:
            index_data = json.load(f)
        self.data = index_data["tiles"] if "tiles" in index_data else index_data
        logger.info(f"Tile index loaded from: {input_path} ({len(self.data)} tiles)")

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return self.data.copy()


def create_roboflow_metadata(tile_index: TileIndex, split: Dict[str, List[str]],
                              source: str = "NAIP", additional_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build metadata dict for Roboflow upload/download traceability."""
    metadata = {
        "tile_index": tile_index.to_dict(), "split": split,
        "export_date": datetime.now().isoformat(), "source": source,
        "total_tiles": len(tile_index.to_dict()), "split_counts": {k: len(v) for k, v in split.items()},
    }
    if additional_info:
        metadata.update(additional_info)
    return metadata


def save_roboflow_metadata(metadata: Dict[str, Any], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Roboflow metadata saved to: {output_path}")


def load_roboflow_metadata(metadata_path: Path) -> Dict[str, Any]:
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Roboflow metadata not found: {metadata_path}")
    with open(metadata_path) as f:
        metadata = json.load(f)
    logger.info(f"Roboflow metadata loaded from: {metadata_path}")
    return metadata


def merge_tile_indices(indices: List[TileIndex], allow_duplicates: bool = False) -> TileIndex:
    """Merge multiple TileIndex objects into one."""
    merged = TileIndex()
    seen_tiles: set = set()
    for index in indices:
        for tile_name, tile_data in index.to_dict().items():
            if tile_name in seen_tiles and not allow_duplicates:
                raise ValueError(f"Duplicate tile name across indices: {tile_name}")
            merged.data[tile_name] = tile_data
            seen_tiles.add(tile_name)
    logger.info(f"Merged {len(indices)} indices into {len(merged.data)} tiles")
    return merged
