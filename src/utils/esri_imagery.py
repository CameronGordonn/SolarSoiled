"""Fetch Esri World Imagery basemap tiles for a bbox.

Used to attach a sub-meter visual reference next to NAIP 60cm tiles when
adjudicating borderline labels. Adjudication only — Esri's standard ToS
prohibits using these images to train ML or redistribute them. We never
write Esri pixels into label files; we only look at them.

REST endpoint:
  https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export

Bbox can be supplied in any CRS supported by Esri (default EPSG:4326).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import requests
from PIL import Image

logger = logging.getLogger(__name__)

ESRI_EXPORT_URL = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/export"
)
DEFAULT_CACHE_DIR = Path("outputs/esri_cache")
REQUEST_TIMEOUT = 20  # seconds
USER_AGENT = "solarsoiled-ml/0.1 (label adjudication; not for redistribution)"


def _cache_key(bbox: Tuple[float, float, float, float], size: int, bbox_crs: str) -> str:
    payload = json.dumps([list(bbox), size, bbox_crs], sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _crs_to_wkid(crs: str) -> int:
    """Convert 'EPSG:4326' -> 4326. Esri also accepts 102100 for web-mercator."""
    crs = crs.strip().upper()
    if crs.startswith("EPSG:"):
        return int(crs.split(":", 1)[1])
    return int(crs)


def fetch_esri_overlay(
    bbox: Tuple[float, float, float, float],
    out_path: Optional[Path] = None,
    size: int = 512,
    bbox_crs: str = "EPSG:4326",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    overwrite: bool = False,
) -> Path:
    """Download an Esri World Imagery render of `bbox` to `out_path` (or cache).

    Args:
        bbox: (minx, miny, maxx, maxy) in `bbox_crs` coordinates.
        out_path: where to save the PNG. If None, a deterministic cache filename is used.
        size: output image dimensions (square).
        bbox_crs: CRS of the bbox tuple (e.g. 'EPSG:4326' or 'EPSG:3857').
        cache_dir: directory for cached responses.
        overwrite: if True, force a re-download even if the cache file exists.

    Returns the path to the saved PNG. Raises on HTTP failure.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if out_path is None:
        key = _cache_key(bbox, size, bbox_crs)
        out_path = cache_dir / f"esri_{key}.png"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        return out_path

    wkid = _crs_to_wkid(bbox_crs)
    params = {
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "bboxSR": str(wkid),
        "imageSR": str(wkid),
        "size": f"{size},{size}",
        "format": "png",
        "transparent": "false",
        "f": "image",
    }

    response = requests.get(
        ESRI_EXPORT_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "image" not in content_type:
        raise RuntimeError(
            f"Esri returned non-image response (Content-Type={content_type!r}). "
            f"First 200 bytes: {response.content[:200]!r}"
        )

    out_path.write_bytes(response.content)
    logger.info("Cached Esri tile %s (%d bytes)", out_path.name, len(response.content))
    return out_path


def fetch_with_retry(
    bbox: Tuple[float, float, float, float],
    out_path: Optional[Path] = None,
    size: int = 512,
    bbox_crs: str = "EPSG:4326",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    retries: int = 3,
    backoff_seconds: float = 2.0,
) -> Optional[Path]:
    """Same as fetch_esri_overlay but swallows transient errors (returns None on final failure)."""
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return fetch_esri_overlay(
                bbox=bbox, out_path=out_path, size=size,
                bbox_crs=bbox_crs, cache_dir=cache_dir,
            )
        except (requests.RequestException, RuntimeError) as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(backoff_seconds * (2 ** attempt))
    logger.warning("Esri fetch failed after %d attempts: %s", retries, last_exc)
    return None


def load_image(path: Path) -> Image.Image:
    """Convenience loader for downstream PIL composition."""
    return Image.open(Path(path)).convert("RGB")
