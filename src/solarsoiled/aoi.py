"""AOI parsing for the CLI.

Accept a bbox string ``"minx,miny,maxx,maxy"`` (WGS84) or a path to a GeoJSON
polygon. Produce a normalized polygon + stable AOI ID for the per-AOI output
namespace.

Tier 2 hardening: WGS84 coordinate-range checks, explicit rejection of
non-WGS84 GeoJSON, and shapely validity checks so partner footguns
(self-intersecting polygons, projected CRS files mislabeled as lon/lat) fail
loud at parse time instead of silently producing nonsense downstream.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from shapely.geometry import Polygon, mapping, shape


# RFC 7946 mandates GeoJSON be WGS84, but real partner files often violate it.
# Accept the explicit WGS84 spellings; reject anything else.
_WGS84_CRS_NAMES = {
    "urn:ogc:def:crs:ogc:1.3:crs84",
    "urn:ogc:def:crs:ogc::crs84",
    "urn:ogc:def:crs:epsg::4326",
    "urn:ogc:def:crs:epsg:6.9:4326",
    "epsg:4326",
    "wgs84",
    "wgs 84",
    "crs84",
}


class AOIValidationError(ValueError):
    """Raised when an AOI fails geometric or CRS validation."""


@dataclass(frozen=True)
class Aoi:
    polygon: Polygon
    aoi_id: str
    source: str  # "bbox" or "geojson"


def _check_wgs84_range(minx: float, miny: float, maxx: float, maxy: float, *, context: str) -> None:
    for name, value in (("minx/lon", minx), ("maxx/lon", maxx)):
        if not -180.0 <= value <= 180.0:
            raise AOIValidationError(
                f"{context}: {name}={value} out of WGS84 longitude range [-180, 180]"
            )
    for name, value in (("miny/lat", miny), ("maxy/lat", maxy)):
        if not -90.0 <= value <= 90.0:
            raise AOIValidationError(
                f"{context}: {name}={value} out of WGS84 latitude range [-90, 90]"
            )


def _bbox_to_polygon(spec: str) -> Polygon:
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4:
        raise AOIValidationError(
            f"bbox AOI must be 'minx,miny,maxx,maxy'; got {spec!r}"
        )
    try:
        minx, miny, maxx, maxy = (float(p) for p in parts)
    except ValueError as exc:
        raise AOIValidationError(f"bbox AOI components must be numeric: {spec!r}") from exc
    if not (minx < maxx and miny < maxy):
        raise AOIValidationError(f"bbox AOI degenerate: {spec!r}")
    _check_wgs84_range(minx, miny, maxx, maxy, context=f"bbox {spec!r}")
    return Polygon(
        [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    )


def _crs_is_wgs84(crs: Any) -> bool:
    """Best-effort check that a GeoJSON-style CRS member names WGS84."""
    if crs is None:
        return True  # absent CRS is RFC 7946 default (WGS84)
    if isinstance(crs, str):
        return crs.strip().lower() in _WGS84_CRS_NAMES
    if isinstance(crs, dict):
        props = crs.get("properties") or {}
        name = props.get("name") or crs.get("name")
        if isinstance(name, str):
            return name.strip().lower() in _WGS84_CRS_NAMES
    return False


def _geojson_to_polygon(path: Path) -> Polygon:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not _crs_is_wgs84(raw.get("crs")):
        raise AOIValidationError(
            f"AOI GeoJSON {path} declares a non-WGS84 CRS ({raw.get('crs')!r}); "
            "reproject to EPSG:4326 before submitting."
        )
    if raw.get("type") == "FeatureCollection":
        features = raw.get("features") or []
        if not features:
            raise AOIValidationError(f"FeatureCollection at {path} is empty")
        geom = features[0]["geometry"]
    elif raw.get("type") == "Feature":
        geom = raw["geometry"]
    else:
        geom = raw
    poly = shape(geom)
    if poly.geom_type != "Polygon":
        raise AOIValidationError(
            f"AOI GeoJSON must be a single Polygon; got {poly.geom_type}"
        )
    if not poly.is_valid:
        raise AOIValidationError(
            f"AOI polygon at {path} is invalid: {poly.is_valid_reason if hasattr(poly, 'is_valid_reason') else 'self-intersecting or malformed'}"
        )
    minx, miny, maxx, maxy = poly.bounds
    _check_wgs84_range(minx, miny, maxx, maxy, context=f"GeoJSON {path}")
    return poly


def _stable_id(polygon: Polygon) -> str:
    canonical = json.dumps(
        mapping(polygon), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha1(canonical).hexdigest()[:10]


_SCENES_PATH = Path(__file__).resolve().parents[2] / "configs" / "scenes.yaml"
_scenes_cache: dict[str, dict] | None = None


def _load_scenes() -> dict[str, dict]:
    global _scenes_cache
    if _scenes_cache is None:
        if _SCENES_PATH.exists():
            _scenes_cache = yaml.safe_load(_SCENES_PATH.read_text()) or {}
        else:
            _scenes_cache = {}
    return _scenes_cache


def _normalize_scene_key(s: str) -> str:
    return re.sub(r"[\s\-]+", "_", s.strip().lower())


def _resolve_named_scene(spec: str) -> Polygon | None:
    key = _normalize_scene_key(spec)
    scenes = _load_scenes()
    entry = scenes.get(key)
    if entry is None:
        return None
    minx, miny, maxx, maxy = entry["bbox"]
    _check_wgs84_range(minx, miny, maxx, maxy, context=f"scene {key!r}")
    return Polygon(
        [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    )


def parse_aoi(spec: str, *, partner_id: str | None = None) -> Aoi:
    """Parse a CLI ``--aoi`` argument into a normalized AOI.

    Accepts (in order):
    - A named scene from ``configs/scenes.yaml`` (e.g. ``"santa_cruz"``)
    - A path to a GeoJSON file
    - A comma-separated bbox ``"minx,miny,maxx,maxy"`` (WGS84)

    ``partner_id`` overrides the auto-generated content hash so partner runs
    land in a stable directory.
    """
    # 1. Named scene lookup
    polygon = _resolve_named_scene(spec)
    if polygon is not None:
        aoi_id = partner_id or _normalize_scene_key(spec)
        return Aoi(polygon=polygon, aoi_id=aoi_id, source="scene")

    # 2. GeoJSON file
    candidate = Path(spec)
    if candidate.exists() and candidate.is_file():
        polygon = _geojson_to_polygon(candidate)
        aoi_id = partner_id or _stable_id(polygon)
        return Aoi(polygon=polygon, aoi_id=aoi_id, source="geojson")

    # 3. Bbox string — if it doesn't look like a bbox, give a better error
    if "," not in spec:
        known = sorted(_load_scenes().keys())
        raise AOIValidationError(
            f"Unknown named scene {spec!r}. Known scenes: {', '.join(known)}. "
            f"Also accepts a GeoJSON file path or 'minx,miny,maxx,maxy' bbox."
        )
    polygon = _bbox_to_polygon(spec)
    aoi_id = partner_id or _stable_id(polygon)
    return Aoi(polygon=polygon, aoi_id=aoi_id, source="bbox")


def write_aoi_geojson(aoi: Aoi, path: Path) -> Path:
    """Write the normalized AOI polygon to ``path`` as GeoJSON Feature."""
    path.parent.mkdir(parents=True, exist_ok=True)
    feature: dict[str, Any] = {
        "type": "Feature",
        "properties": {"aoi_id": aoi.aoi_id, "source": aoi.source},
        "geometry": mapping(aoi.polygon),
    }
    path.write_text(json.dumps(feature, indent=2) + "\n", encoding="utf-8")
    return path
