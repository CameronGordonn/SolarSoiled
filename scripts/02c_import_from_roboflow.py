"""Import labeled dataset from Roboflow and prepare for YOLO training under data/yolo/naip/.

Supports two Roboflow export formats:
  - YOLOv8 Segmentation (preferred): images + .txt polygon labels, data.yaml
  - COCO Segmentation: images + _annotations.coco.json per split (use when YOLOv8 Seg unavailable)
"""

from pathlib import Path
import json
import logging
import shutil
import random
from datetime import datetime

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.tile_metadata import load_roboflow_metadata, save_roboflow_metadata

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def detect_export_layout(download_dir: Path) -> str:
    if (download_dir / "images" / "train").exists():
        return "images_first"
    if (download_dir / "train" / "images").exists():
        return "split_first"
    raise ValueError("Unrecognized Roboflow export layout. Expected images/train or train/images structure.")


def split_aliases(split: str) -> list:
    return ["val", "valid"] if split == "val" else [split]


def resolve_split_dirs(download_dir: Path, split: str) -> tuple[Path, Path, str | None]:
    layout = detect_export_layout(download_dir)
    for alias in split_aliases(split):
        if layout == "images_first":
            images_dir = download_dir / "images" / alias
            labels_dir = download_dir / "labels" / alias
        else:
            images_dir = download_dir / alias / "images"
            labels_dir = download_dir / alias / "labels"
        if images_dir.exists():
            return images_dir, labels_dir, alias
    return Path(), Path(), None


def list_image_stems(images_dir: Path) -> set:
    if not images_dir.exists():
        return set()
    return {p.stem for p in images_dir.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS}


def collect_paired_samples(images_dir: Path, labels_dir: Path) -> list:
    img_by_stem = {p.stem: p for p in images_dir.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    lbl_by_stem = {p.stem: p for p in labels_dir.glob("*.txt") if p.is_file()}
    shared = sorted(set(img_by_stem) & set(lbl_by_stem))
    return [(img_by_stem[s], lbl_by_stem[s]) for s in shared]


def split_train_only_dataset(download_dir: Path, output_dir: Path, seed: int = 42) -> dict:
    """Split train-only Roboflow export into train/val/test and copy to output."""
    for d in (output_dir / "images", output_dir / "labels"):
        if d.exists():
            shutil.rmtree(d)

    train_images, train_labels, _ = resolve_split_dirs(download_dir, "train")
    if not train_images.exists() or not train_labels.exists():
        raise ValueError("Train-only split requires train/images and train/labels.")

    pairs = collect_paired_samples(train_images, train_labels)
    if not pairs:
        raise ValueError("No image/label pairs found in train-only export.")

    random.seed(seed)
    random.shuffle(pairs)
    n = len(pairs)
    n_train = max(1, int(n * 0.7))
    n_val = max(1, int(n * 0.15))
    n_test = max(1, n - n_train - n_val)
    n_train = max(1, n_train - (1 if n_test < 1 else 0))

    split_map = {
        "train": pairs[:n_train],
        "val":   pairs[n_train:n_train + n_val],
        "test":  pairs[n_train + n_val:],
    }
    for split, split_pairs in split_map.items():
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img_file, lbl_file in split_pairs:
            shutil.copy2(img_file, output_dir / "images" / split / img_file.name)
            shutil.copy2(lbl_file, output_dir / "labels" / split / lbl_file.name)

    return {s: {"total_images": len(p), "images_with_labels": len(p), "missing_labels": 0}
            for s, p in split_map.items()}


def validate_yolo_format(download_dir: Path) -> dict:
    train_images, train_labels, _ = resolve_split_dirs(download_dir, "train")
    val_images, val_labels, val_alias = resolve_split_dirs(download_dir, "val")

    if not train_images.exists() or not train_labels.exists():
        raise ValueError("Missing required train images/labels directories.")

    train_only_mode = (
        not val_images.exists() or not val_labels.exists()
        or len(list_image_stems(val_images)) == 0
    )
    if train_only_mode:
        logger.warning("Validation split not found — treating as train-only export.")

    if not (download_dir / "data.yaml").exists():
        raise ValueError(f"Missing data.yaml in {download_dir}")

    results = {}
    for split in (["train"] if train_only_mode else ["train", "val"]):
        if split == "train":
            images_dir, labels_dir = train_images, train_labels
        else:
            images_dir, labels_dir = val_images, val_labels
        img_stems = list_image_stems(images_dir)
        lbl_stems = {p.stem for p in labels_dir.glob("*.txt")} if labels_dir.exists() else set()
        missing = img_stems - lbl_stems
        if missing:
            logger.warning(f"{split}: missing labels for {len(missing)} images")
        results[split] = {
            "total_images": len(img_stems),
            "images_with_labels": len(img_stems & lbl_stems),
            "missing_labels": len(missing),
        }
        logger.info(f"  {split}: {len(img_stems)} images, {len(img_stems & lbl_stems)} with labels")

    test_images_dir, test_labels_dir, _ = resolve_split_dirs(download_dir, "test")
    if test_images_dir.exists():
        n_img = len(list_image_stems(test_images_dir))
        n_lbl = len(list(test_labels_dir.glob("*.txt"))) if test_labels_dir.exists() else 0
        results["test"] = {"total_images": n_img, "images_with_labels": n_lbl, "missing_labels": max(0, n_img - n_lbl)}

    results["train_only_mode"] = train_only_mode
    return results


def copy_split_to_yolo_structure(download_dir: Path, output_dir: Path, splits: list = None) -> None:
    for split in (splits or ["train", "val", "test"]):
        src_images, src_labels, _ = resolve_split_dirs(download_dir, split)
        if not src_images.exists():
            logger.warning(f"Split '{split}' not found — skipping.")
            continue

        dst_images = output_dir / "images" / split
        dst_labels = output_dir / "labels" / split
        for d in (dst_images, dst_labels):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

        for img_file in src_images.glob("*"):
            if not img_file.is_file() or img_file.suffix.lower() not in IMAGE_EXTS:
                continue
            shutil.copy2(img_file, dst_images / img_file.name)

        image_stems = {p.stem for p in dst_images.glob("*") if p.is_file()}
        if src_labels.exists():
            for lbl_file in src_labels.glob("*.txt"):
                if lbl_file.stem in image_stems:
                    shutil.copy2(lbl_file, dst_labels / lbl_file.name)

        logger.info(f"  {split}/: {len(list(dst_images.glob('*.*')))} images")


def is_coco_format(download_dir: Path) -> bool:
    """Return True if the download looks like a COCO Segmentation export (JSON annotations, no .txt labels)."""
    for split_name in ["train", "valid", "val", "test"]:
        split_dir = download_dir / split_name
        if split_dir.exists() and list(split_dir.glob("*.json")):
            return True
    return False


def coco_to_yolo_labels(coco_json: Path, dst_labels: Path) -> int:
    """Convert COCO segmentation JSON → YOLO polygon .txt files. Returns annotation count."""
    with open(coco_json) as f:
        coco = json.load(f)

    img_info = {img["id"]: img for img in coco["images"]}
    ann_by_image: dict[int, list] = {}
    for ann in coco.get("annotations", []):
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    dst_labels.mkdir(parents=True, exist_ok=True)
    written = 0
    for img_id, meta in img_info.items():
        w, h = meta["width"], meta["height"]
        stem = Path(meta["file_name"]).stem
        lines = []
        for ann in ann_by_image.get(img_id, []):
            seg_field = ann.get("segmentation", [])
            # RLE format (dict with 'counts'/'size') — skip; needs pycocotools to decode
            if isinstance(seg_field, dict):
                continue
            for seg in seg_field:
                if not isinstance(seg, list) or len(seg) < 6:
                    continue
                coords = [float(seg[i]) / w if i % 2 == 0 else float(seg[i]) / h for i in range(len(seg))]
                lines.append("0 " + " ".join(f"{c:.6f}" for c in coords))
        (dst_labels / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        written += len(lines)
    return written


def import_coco_format(download_dir: Path, output_dir: Path) -> dict:
    """Import COCO Segmentation export into YOLO directory structure."""
    split_alias = {"valid": "val"}
    results = {}

    for src_split in ["train", "valid", "test"]:
        src_dir = download_dir / src_split
        if not src_dir.exists():
            continue

        json_files = sorted(src_dir.glob("*annotations*.json")) or sorted(src_dir.glob("*.json"))
        if not json_files:
            logger.warning(f"No COCO JSON in {src_dir} — skipping.")
            continue

        dst_split = split_alias.get(src_split, src_split)
        dst_images = output_dir / "images" / dst_split
        dst_labels = output_dir / "labels" / dst_split
        for d in (dst_images, dst_labels):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

        n_images = 0
        for img_file in src_dir.glob("*"):
            if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTS:
                shutil.copy2(img_file, dst_images / img_file.name)
                n_images += 1

        n_anns = coco_to_yolo_labels(json_files[0], dst_labels)
        results[dst_split] = {"total_images": n_images, "images_with_labels": n_images, "missing_labels": 0}
        logger.info(f"  {dst_split}/: {n_images} images, {n_anns} annotations → YOLO .txt")

    return results


def create_training_data_yaml(output_dir: Path) -> None:
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
    download_dir = repo_root / "roboflow_download"
    alt_dir = repo_root / "roboflow_upload" / "YOLOv8 (Segmentation)"
    upload_dir = repo_root / "roboflow_upload"
    output_dir = repo_root / "data" / "yolo" / "naip"

    if not download_dir.exists():
        if alt_dir.exists():
            download_dir = alt_dir
        else:
            raise FileNotFoundError(
                f"Roboflow download not found: {download_dir}\n"
                "Download from roboflow.com → Download Dataset → YOLOv8 (Segmentation) or COCO Segmentation → save to roboflow_download/"
            )

    logger.info(f"Importing from: {download_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_coco_format(download_dir):
        logger.info("Detected COCO Segmentation format — converting to YOLO polygon labels.")
        validation = import_coco_format(download_dir, output_dir)
    else:
        validation = validate_yolo_format(download_dir)
        if validation.get("train_only_mode"):
            logger.info("Train-only export — splitting 70/15/15...")
            validation = split_train_only_dataset(download_dir, output_dir)
        else:
            copy_split_to_yolo_structure(download_dir, output_dir)

    create_training_data_yaml(output_dir)

    metadata_src = upload_dir / "roboflow_metadata.json"
    if metadata_src.exists():
        try:
            metadata = load_roboflow_metadata(metadata_src)
            metadata["import_date"] = datetime.now().isoformat()
            metadata["validation_results"] = validation
            save_roboflow_metadata(metadata, output_dir / "roboflow_metadata.json")
        except Exception as e:
            logger.warning(f"Could not preserve metadata: {e}")

    train_n = validation.get("train", {}).get("images_with_labels", "?")
    val_n = validation.get("val", {}).get("images_with_labels", "?")
    logger.info(f"Done. Train={train_n} Val={val_n} → {output_dir}")


if __name__ == "__main__":
    main()
