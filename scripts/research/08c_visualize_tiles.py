#!/usr/bin/env python3
"""Interactive tile footprint map using GeoAI. Exits gracefully if GeoAI is not installed."""

from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    import geoai
    GEOAI_AVAILABLE = True
except ImportError:
    GEOAI_AVAILABLE = False

from src.utils.tile_metadata import TileIndex


def visualize(index_path: Path) -> None:
    if not index_path.exists():
        logger.error(f"Tile index not found: {index_path}")
        return

    gdf = TileIndex(index_path).to_geodataframe()
    if gdf.empty:
        logger.warning("Tile index is empty — nothing to display")
        return

    if not GEOAI_AVAILABLE:
        logger.warning("GeoAI not installed. Install: conda install -c conda-forge geoai")
        return

    try:
        if hasattr(geoai, "view_vector_interactive"):
            geoai.view_vector_interactive(gdf, title="NAIP Tile Footprints")
        else:
            m = geoai.Map()
            m.add_geodataframe(gdf, name="tiles")
            m.show()
    except Exception as e:
        logger.error(f"Visualization failed: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize NAIP tile footprints (requires GeoAI)")
    parser.add_argument(
        "--index", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "interim" / "tile_index.json",
    )
    args = parser.parse_args()
    visualize(args.index)


if __name__ == "__main__":
    main()
