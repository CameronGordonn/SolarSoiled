"""Tile a NAIP RGB GeoTIFF into 640px YOLO-ready tiles and generate tile_index.json.

Supports optional data acquisition via GeoAI (STAC search or bbox) when in the conda env.
"""

from pathlib import Path
import json
import logging

import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.naip_preprocessing import detect_rgb_band_order, tone_map_naip_clahe
from utils.tile_metadata import TileIndex
from solarsoiled.manifest import write_manifest

try:
    import geoai.download as dl
    GEOAI_AVAILABLE = True
except ImportError:
    GEOAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TILE_SIZE = 640


def download_naip(aoi, output_dir):
    if not GEOAI_AVAILABLE:
        raise ImportError("GeoAI required for downloading. Install: pip install geoai-py")
    if isinstance(aoi, str) and " " in aoi:
        try:
            import geoai.geo_agents as ga
            results = ga.search_stac(collection="naip", bbox=aoi)
            if results:
                bbox = results[0].get("bbox") or aoi
                aoi = ",".join(map(str, bbox)) if isinstance(bbox, (list, tuple)) else bbox
        except Exception as e:
            logger.warning(f"STAC search failed: {e}")
    logger.info(f"Downloading NAIP for AOI: {aoi}")
    dl.download_naip(aoi=aoi, output_dir=str(output_dir), year=2023)


def main(
    download_aoi: str | None = None,
    out_tiles_dir: Path | None = None,
    out_tile_index: Path | None = None,
):
    repo_root = Path(__file__).resolve().parents[1]

    if download_aoi:
        raw_dir = repo_root / "data" / "raw" / "naip"
        raw_dir.mkdir(parents=True, exist_ok=True)
        download_naip(download_aoi, raw_dir)

    raw_dir = repo_root / "data" / "raw"
    naip_subdir = raw_dir / "naip"
    naip_files = sorted(naip_subdir.glob("*.tif")) if naip_subdir.exists() else sorted(raw_dir.glob("*NAIP*.tif"))

    if not naip_files:
        raise FileNotFoundError(f"No NAIP GeoTIFF files found in {naip_subdir} or {raw_dir}")

    tiles_dir = Path(out_tiles_dir) if out_tiles_dir else repo_root / "data" / "tiles"
    tile_index_path = (
        Path(out_tile_index) if out_tile_index else repo_root / "data" / "interim" / "tile_index.json"
    )
    tiles_dir.mkdir(parents=True, exist_ok=True)
    tile_index_path.parent.mkdir(parents=True, exist_ok=True)

    tile_index = TileIndex()
    global_tile_counter = 0
    source_crs: str | None = None

    for naip_path in naip_files:
        logger.info(f"Processing: {naip_path.name}")
        with rasterio.open(naip_path) as src:
            width, height = src.width, src.height
            crs = src.crs.to_string()
            if source_crs is None:
                source_crs = crs
            r_band, g_band, b_band = detect_rgb_band_order(src)
            n_cols = (width + TILE_SIZE - 1) // TILE_SIZE
            n_rows = (height + TILE_SIZE - 1) // TILE_SIZE
            logger.info(f"  {width}x{height} → {n_rows}x{n_cols} tiles  CRS={crs}")

            for row in range(n_rows):
                for col in range(n_cols):
                    x_off, y_off = col * TILE_SIZE, row * TILE_SIZE
                    win_w = min(TILE_SIZE, width - x_off)
                    win_h = min(TILE_SIZE, height - y_off)
                    window = Window(x_off, y_off, win_w, win_h)

                    try:
                        rgb = src.read(indexes=[r_band, g_band, b_band], window=window)
                    except Exception as e:
                        logger.warning(f"  Tile {col},{row} read failed: {e}")
                        continue

                    if rgb.size == 0:
                        continue
                    if rgb.dtype == np.uint16:
                        rgb = (rgb / 256).astype(np.uint8)

                    rgb_mapped = tone_map_naip_clahe(rgb, clip_limit=0.03)
                    tile_name = f"tile_{global_tile_counter:06d}.png"
                    global_tile_counter += 1

                    try:
                        Image.fromarray(np.transpose(rgb_mapped, (1, 2, 0)), mode="RGB").save(tiles_dir / tile_name)
                    except Exception as e:
                        logger.error(f"  Failed to save {tile_name}: {e}")
                        continue

                    transform = src.window_transform(window)
                    tile_index.add_tile(
                        tile_name=tile_name, transform=transform, crs=crs,
                        width=win_w, height=win_h, source_file=naip_path.name,
                        bounds={
                            "minx": transform.c, "miny": transform.f + win_h * transform.e,
                            "maxx": transform.c + win_w * transform.a, "maxy": transform.f,
                        },
                    )

    try:
        tile_index.validate()
    except ValueError as e:
        logger.error(f"Tile index validation failed: {e}")
        raise

    tile_index.save(tile_index_path)
    n_tiles = len(tile_index.to_dict())
    logger.info(f"Created {n_tiles} tiles → {tile_index_path}")

    write_manifest(
        tiles_dir,
        stage="tile",
        model_version="naip-tiler-v1",
        model_weights=None,
        inputs=[str(p) for p in naip_files],
        metrics={"tile_count": n_tiles, "tile_size_px": TILE_SIZE},
        known_limitations=["NAIP RGB tone-mapped via CLAHE (clip_limit=0.03)"],
        extra={"source_crs": source_crs, "tile_index": str(tile_index_path)},
    )


def _parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Tile NAIP imagery with optional GeoAI download.")
    parser.add_argument("--download-aoi", help="AOI bbox (minx,miny,maxx,maxy) or free-text query")
    parser.add_argument("--out-tiles-dir", type=Path, default=None, help="Override default tiles output dir")
    parser.add_argument("--out-tile-index", type=Path, default=None, help="Override default tile_index.json path")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    main(
        download_aoi=args.download_aoi,
        out_tiles_dir=args.out_tiles_dir,
        out_tile_index=args.out_tile_index,
    )
