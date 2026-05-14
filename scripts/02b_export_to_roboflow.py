"""Prepare and package tiles for Roboflow upload (70/15/15 train/val/test split)."""

from pathlib import Path
import logging
import random
import shutil

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.tile_metadata import TileIndex, create_roboflow_metadata, save_roboflow_metadata

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
RANDOM_SEED = 42


def copy_split(tiles_dir: Path, split: dict, output_dir: Path) -> None:
    for split_name, tile_names in split.items():
        out = output_dir / "images" / split_name
        out.mkdir(parents=True, exist_ok=True)
        for name in tile_names:
            src = tiles_dir / name
            if not src.exists():
                logger.warning(f"Tile not found: {src}")
                continue
            shutil.copy2(src, out / name)
        logger.info(f"  {split_name}/: {len(tile_names)} tiles")


def create_data_yaml(output_dir: Path) -> None:
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({
            "path": str(output_dir.resolve()),
            "train": "images/train", "val": "images/val", "test": "images/test",
            "nc": 1, "names": {0: "solar_array"},
        }, f, default_flow_style=False)
    logger.info(f"Created: {yaml_path}")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    tiles_dir = repo_root / "data" / "tiles"
    tile_index_path = repo_root / "data" / "interim" / "tile_index.json"
    roboflow_dir = repo_root / "roboflow_upload"

    if not tiles_dir.exists():
        raise FileNotFoundError(f"Tiles directory not found: {tiles_dir}")
    if not tile_index_path.exists():
        raise FileNotFoundError(f"Tile index not found: {tile_index_path}")

    tile_index = TileIndex(tile_index_path)
    tile_names = list(tile_index.to_dict().keys())
    total = len(tile_names)
    if total == 0:
        raise ValueError("Tile index is empty")

    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)
    random.seed(RANDOM_SEED)
    random.shuffle(tile_names)
    split = {
        "train": tile_names[:n_train],
        "val":   tile_names[n_train:n_train + n_val],
        "test":  tile_names[n_train + n_val:],
    }
    logger.info(f"Split: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")

    roboflow_dir.mkdir(parents=True, exist_ok=True)
    copy_split(tiles_dir, split, roboflow_dir)
    create_data_yaml(roboflow_dir)

    metadata = create_roboflow_metadata(tile_index=tile_index, split=split, source="NAIP",
                                        additional_info={"random_seed": RANDOM_SEED})
    save_roboflow_metadata(metadata, roboflow_dir / "roboflow_metadata.json")
    logger.info(f"Done. Output: {roboflow_dir}")


if __name__ == "__main__":
    main()
