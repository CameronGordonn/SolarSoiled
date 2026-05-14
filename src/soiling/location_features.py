"""Static geographic context per (lat, lon): elevation, land-cover, OSM proximity.

Elevation: Open-Meteo (free, no key). NLCD: local CONUS GeoTIFF if present,
sampled via rasterio. OSM distance: Overpass-API queries cached on disk so
re-runs are instant.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Hard rate limit for Overpass — be polite. Free tier accepts ~2 req/sec.
_LAST_OVERPASS_CALL: float = 0.0
_OVERPASS_MIN_INTERVAL_S: float = 1.1

# Treat anything beyond this radius as "far enough not to matter" — cheaper
# to cap the query rather than pull a giant bbox of OSM data.
HIGHWAY_RADIUS_M = 5000.0
AGRICULTURE_RADIUS_M = 5000.0
EARTH_RADIUS_M = 6_371_000.0


def _session(cache_dir: Path, name: str = "location"):
    try:
        from requests_cache import CachedSession
    except ImportError as err:
        raise ImportError(
            "requests-cache is required. Install via `pip install requests-cache`."
        ) from err
    cache_dir.mkdir(parents=True, exist_ok=True)
    return CachedSession(
        cache_name=str(cache_dir / name),
        backend="sqlite",
        expire_after=None,  # elevation/land-cover/OSM do not expire on relevant timescales
        allowable_methods=("GET", "HEAD", "POST"),  # Overpass-API only accepts POST
        match_headers=False,
    )


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def _overpass_query(
    query: str,
    cache_dir: Path,
    timeout: int = 60,
) -> Optional[dict]:
    """POST an Overpass-QL query, return parsed JSON, or None on failure.

    Cached by query body so identical re-runs are instant. Rate-limited via a
    process-global last-call timestamp to stay within Overpass's fair-use limits.
    Overpass returns 406 to default Python user agents — set an explicit one.
    """
    global _LAST_OVERPASS_CALL
    sess = _session(Path(cache_dir), name="overpass")
    sess.headers.update({"User-Agent": "solar-soiling-ml/1.0 (research; contact via github)"})
    elapsed = time.monotonic() - _LAST_OVERPASS_CALL
    if elapsed < _OVERPASS_MIN_INTERVAL_S:
        time.sleep(_OVERPASS_MIN_INTERVAL_S - elapsed)
    try:
        resp = sess.post(OVERPASS_URL, data={"data": query}, timeout=timeout)
        _LAST_OVERPASS_CALL = time.monotonic()
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Overpass query failed: %s", exc)
        return None


def _bbox_for_radius(lat: float, lon: float, radius_m: float) -> Tuple[float, float, float, float]:
    """Approximate (south, west, north, east) bbox for a radius in meters."""
    dlat = radius_m / 111_000.0
    dlon = radius_m / (111_000.0 * max(0.1, math.cos(math.radians(lat))))
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)


def _min_distance_to_elements(lat: float, lon: float, elements: Iterable[dict]) -> Optional[float]:
    """Min haversine distance from (lat, lon) to any node/way coordinate in elements."""
    best: Optional[float] = None
    for el in elements:
        pts: List[Tuple[float, float]] = []
        if "lat" in el and "lon" in el:
            pts.append((el["lat"], el["lon"]))
        for g in el.get("geometry", []) or []:
            if "lat" in g and "lon" in g:
                pts.append((g["lat"], g["lon"]))
        for plat, plon in pts:
            d = _haversine_m(lat, lon, plat, plon)
            if best is None or d < best:
                best = d
    return best


def fetch_elevation(
    lat: float,
    lon: float,
    cache_dir: Path = Path(".cache/soiling"),
    timeout: int = 15,
) -> float:
    sess = _session(Path(cache_dir))
    resp = sess.get(ELEVATION_URL, params={"latitude": lat, "longitude": lon}, timeout=timeout)
    resp.raise_for_status()
    elevations = resp.json().get("elevation", [])
    if not elevations:
        raise RuntimeError(f"Open-Meteo elevation returned empty for ({lat}, {lon})")
    return float(elevations[0])


def sample_nlcd(lat: float, lon: float, nlcd_path: Optional[Path] = None) -> Optional[int]:
    """Sample NLCD land-cover class at (lat, lon). Returns None if raster unavailable.

    Expected classes of interest for soiling:
      81 / 82 — cultivated crops / pasture (dust from tilling)
      23 / 24 — developed high intensity (urban PM from traffic)
      52      — shrub/scrub (arid, dust-prone)
    """
    if nlcd_path is None or not Path(nlcd_path).exists():
        return None
    try:
        import rasterio
        from rasterio.warp import transform
    except ImportError:
        logger.warning("rasterio unavailable; skipping NLCD sample")
        return None
    with rasterio.open(nlcd_path) as src:
        xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
        row, col = src.index(xs[0], ys[0])
        if row < 0 or col < 0 or row >= src.height or col >= src.width:
            return None
        return int(src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])


def distance_to_highway_m(
    lat: float,
    lon: float,
    cache_dir: Path = Path(".cache/soiling"),
    radius_m: float = HIGHWAY_RADIUS_M,
) -> Optional[float]:
    """Min distance to a major OSM highway (motorway|trunk|primary|secondary)
    within `radius_m` of (lat, lon). Returns radius_m if none found in bbox.
    """
    s, w, n, e = _bbox_for_radius(lat, lon, radius_m)
    query = (
        f"[out:json][timeout:30];"
        f"way[\"highway\"~\"motorway|trunk|primary|secondary\"]({s},{w},{n},{e});"
        f"out geom;"
    )
    payload = _overpass_query(query, cache_dir=cache_dir)
    if payload is None:
        return None
    d = _min_distance_to_elements(lat, lon, payload.get("elements", []))
    return float(d) if d is not None else float(radius_m)


def distance_to_agriculture_m(
    lat: float,
    lon: float,
    cache_dir: Path = Path(".cache/soiling"),
    radius_m: float = AGRICULTURE_RADIUS_M,
) -> Optional[float]:
    """Min distance to OSM cropland / orchard / vineyard / farmland polygon
    within `radius_m`. Returns radius_m if none found.
    """
    s, w, n, e = _bbox_for_radius(lat, lon, radius_m)
    query = (
        f"[out:json][timeout:30];"
        f"("
        f"way[\"landuse\"~\"farmland|orchard|vineyard|farmyard\"]({s},{w},{n},{e});"
        f"way[\"natural\"=\"farmland\"]({s},{w},{n},{e});"
        f");"
        f"out geom;"
    )
    payload = _overpass_query(query, cache_dir=cache_dir)
    if payload is None:
        return None
    d = _min_distance_to_elements(lat, lon, payload.get("elements", []))
    return float(d) if d is not None else float(radius_m)


_STATIC_LOOKUP_CACHE: Dict[str, Dict[Tuple[float, float], dict]] = {}


def load_static_lookup(path: Path, decimals: int = 3) -> Dict[Tuple[float, float], dict]:
    """Load `static_features.csv` (from scripts/13_build_static_features.py) keyed
    by rounded (lat, lon). Decimals=3 → ~110 m precision, fine for land-cover
    bucketing. Cached by path so callers can call repeatedly cheaply.
    """
    key = str(path)
    if key in _STATIC_LOOKUP_CACHE:
        return _STATIC_LOOKUP_CACHE[key]
    import pandas as pd

    if not Path(path).exists():
        _STATIC_LOOKUP_CACHE[key] = {}
        return {}
    df = pd.read_csv(path)
    lookup: Dict[Tuple[float, float], dict] = {}
    for _, r in df.iterrows():
        k = (round(float(r["latitude"]), decimals), round(float(r["longitude"]), decimals))
        lookup[k] = r.to_dict()
    _STATIC_LOOKUP_CACHE[key] = lookup
    logger.info("Loaded %d static-feature rows from %s", len(lookup), path)
    return lookup


def _worldcover_one_hot(bucket: Optional[str]) -> Dict[str, float]:
    """Expand the WorldCover bucket label into a small set of soiling-relevant
    one-hot columns. Unknown maps to all-zero so downstream XGBoost handles it
    as missing-equivalent."""
    keys = ["worldcover_cropland", "worldcover_built_up", "worldcover_bare", "worldcover_tree", "worldcover_grass"]
    out = {k: 0.0 for k in keys}
    mapping = {"cropland": "worldcover_cropland", "built_up": "worldcover_built_up",
               "bare": "worldcover_bare", "tree": "worldcover_tree", "grass": "worldcover_grass"}
    if bucket in mapping:
        out[mapping[bucket]] = 1.0
    return out


def location_feature_vector(
    lat: float,
    lon: float,
    nlcd_path: Optional[Path] = None,
    cache_dir: Path = Path(".cache/soiling"),
    static_lookup: Optional[Dict[Tuple[float, float], dict]] = None,
) -> Dict[str, Optional[float]]:
    """Build the static feature dict for one location.

    If `static_lookup` is provided (from `load_static_lookup`), use the
    pre-computed values instead of hitting the network. Falls back to live
    API calls on cache miss so behavior is graceful.
    """
    if static_lookup:
        cached = static_lookup.get((round(lat, 3), round(lon, 3)))
        if cached is not None:
            base = {
                "elevation_m": cached.get("elevation_m"),
                "nlcd_class": cached.get("worldcover_class"),  # backward-compat name
                "distance_to_highway_m": cached.get("distance_to_highway_m"),
                "distance_to_agriculture_m": cached.get("distance_to_agriculture_m"),
            }
            base.update(_worldcover_one_hot(cached.get("worldcover_bucket")))
            return base
    return {
        "elevation_m": fetch_elevation(lat, lon, cache_dir=cache_dir),
        "nlcd_class": sample_nlcd(lat, lon, nlcd_path=nlcd_path),
        "distance_to_highway_m": distance_to_highway_m(lat, lon, cache_dir=cache_dir),
        "distance_to_agriculture_m": distance_to_agriculture_m(lat, lon, cache_dir=cache_dir),
    }
