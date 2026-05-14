#!/usr/bin/env python3
"""Optional SAM pre-labeling utility — run GeoAI SAM on tiles to produce mask JSON suggestions."""

from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    import geoai.sam as sam
    GEOAI_SAM = True
except ImportError:
    GEOAI_SAM = False


def generate_masks(tiles_dir: Path, out_dir: Path, prompt: str = "solar panels") -> None:
    if not GEOAI_SAM:
        raise ImportError("GeoAI SAM not available. Install geoai-py or use conda-forge.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for tile_path in Path(tiles_dir).glob("*.png"):
        logger.info(f"Processing {tile_path.name}")
        try:
            masks = sam.segment_image(str(tile_path), text_prompt=prompt)
            (out_dir / f"{tile_path.stem}_sam_masks.json").write_text(json.dumps(masks, indent=2))
        except Exception as e:
            logger.error(f"Failed {tile_path.name}: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate SAM masks for PNG tiles")
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="solar panels")
    args = parser.parse_args()
    generate_masks(args.tiles, args.output, args.prompt)


if __name__ == "__main__":
    main()
