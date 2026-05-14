"""Remove black, empty, and tiny-object tiles from a YOLO Duke dataset."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def parse_label_file(label_path: Path) -> list[str]:
    text = label_path.read_text().strip()
    if not text:
        return []
    return [ln for ln in text.splitlines() if ln.strip()]


def polygon_bbox_px(line: str, tile_w: int = 640, tile_h: int = 640) -> float:
    """Return max(bbox_width, bbox_height) in pixels for a YOLO polygon line."""
    parts = line.split()
    try:
        coords = [float(p) for p in parts[1:]]
    except ValueError:
        return 0.0
    if len(coords) < 4:
        return 0.0
    xs = [coords[i] * tile_w for i in range(0, len(coords), 2)]
    ys = [coords[i] * tile_h for i in range(1, len(coords), 2)]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def is_dark_tile(image_path: Path, min_brightness: float) -> bool:
    try:
        from PIL import Image as PILImage
        arr = np.asarray(PILImage.open(image_path).convert("RGB"), dtype=np.float32)
        return float(arr.mean()) < min_brightness
    except Exception as exc:
        logger.warning(f"Could not read {image_path}: {exc}")
        return False


def process_split(dataset_dir: Path, split: str, min_brightness: float, min_obj_px: float, dry_run: bool) -> dict:
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split

    if not images_dir.exists():
        logger.warning(f"  Split '{split}' not found at {images_dir} — skipping")
        return {}

    image_paths = sorted(images_dir.glob("*.png"))
    total = len(image_paths)
    removed_dark = removed_no_label = removed_all_tiny = pruned_labels = kept = 0

    for img_path in image_paths:
        label_path = labels_dir / img_path.with_suffix(".txt").name

        if is_dark_tile(img_path, min_brightness):
            removed_dark += 1
            if not dry_run:
                img_path.unlink(missing_ok=True)
                label_path.unlink(missing_ok=True)
            continue

        if not label_path.exists():
            removed_no_label += 1
            if not dry_run:
                img_path.unlink(missing_ok=True)
            continue

        lines = parse_label_file(label_path)
        if not lines:
            removed_no_label += 1
            if not dry_run:
                img_path.unlink(missing_ok=True)
                label_path.unlink(missing_ok=True)
            continue

        large_lines = [ln for ln in lines if polygon_bbox_px(ln) >= min_obj_px]
        if not large_lines:
            removed_all_tiny += 1
            if not dry_run:
                img_path.unlink(missing_ok=True)
                label_path.unlink(missing_ok=True)
            continue

        if len(large_lines) < len(lines):
            pruned_labels += 1
            if not dry_run:
                label_path.write_text("\n".join(large_lines))
        kept += 1

    return {"split": split, "total": total, "removed_dark": removed_dark,
            "removed_no_label": removed_no_label, "removed_all_tiny": removed_all_tiny,
            "pruned_labels": pruned_labels, "kept": kept}


def print_stats(stats_list: list[dict], dry_run: bool) -> None:
    mode = "DRY RUN — no files changed" if dry_run else "Files removed/rewritten"
    print(f"\n{'='*60}\n  Duke dataset cleaning  ({mode})\n{'='*60}")
    print(f"{'Split':<8} {'Total':>7} {'Dark':>6} {'NoLabel':>8} {'AllTiny':>8} {'PrunedLbl':>10} {'Kept':>7}")
    print(f"{'-'*60}")
    totals = {k: 0 for k in ("total", "removed_dark", "removed_no_label", "removed_all_tiny", "pruned_labels", "kept")}
    for s in stats_list:
        if not s:
            continue
        print(f"{s['split']:<8} {s['total']:>7} {s['removed_dark']:>6} {s['removed_no_label']:>8} "
              f"{s['removed_all_tiny']:>8} {s['pruned_labels']:>10} {s['kept']:>7}")
        for k in totals:
            totals[k] += s.get(k, 0)
    print(f"{'-'*60}")
    print(f"{'TOTAL':<8} {totals['total']:>7} {totals['removed_dark']:>6} {totals['removed_no_label']:>8} "
          f"{totals['removed_all_tiny']:>8} {totals['pruned_labels']:>10} {totals['kept']:>7}")
    removed = totals["total"] - totals["kept"]
    print(f"\nRemoved {removed} / {totals['total']} tiles ({100*removed/max(1, totals['total']):.1f}%)\n{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/yolo/duke"))
    parser.add_argument("--min-brightness", type=float, default=15.0)
    parser.add_argument("--min-obj-px", type=float, default=16.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    logger.info(f"Cleaning {dataset_dir} (min_brightness={args.min_brightness}, min_obj_px={args.min_obj_px}, dry_run={args.dry_run})")
    stats_list = [process_split(dataset_dir, split, args.min_brightness, args.min_obj_px, args.dry_run) for split in SPLITS]
    print_stats(stats_list, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
