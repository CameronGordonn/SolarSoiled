"""Draw YOLO polygon labels onto tile PNGs and save annotated copies for visual inspection."""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import yaml

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
COLORS = [(255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 200, 0), (0, 200, 255)]
ALPHA = 80  # polygon fill opacity (0–255)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=20, help="Number of tiles to sample")
    parser.add_argument("--out-dir", type=str, default="outputs/label_viz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="Process all tiles (ignores --n)")
    return parser.parse_args()


def load_data_yaml(data_path: Path) -> Path:
    with open(data_path) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    return root


def parse_label(label_path: Path, img_w: int, img_h: int) -> list[list[tuple]]:
    """Return list of polygon point lists in pixel coords."""
    polygons = []
    if not label_path.exists():
        return polygons
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            coords = [float(p) for p in parts[1:]]
        except ValueError:
            continue
        xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
        ys = [coords[i] * img_h for i in range(1, len(coords), 2)]
        polygons.append(list(zip(xs, ys)))
    return polygons


def annotate_tile(img_path: Path, label_path: Path) -> tuple[Image.Image, int]:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    polygons = parse_label(label_path, w, h)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for i, poly in enumerate(polygons):
        color = COLORS[i % len(COLORS)]
        fill = color + (ALPHA,)
        outline = color + (220,)
        if len(poly) >= 3:
            draw.polygon(poly, fill=fill, outline=outline)

    img_rgba = img.convert("RGBA")
    combined = Image.alpha_composite(img_rgba, overlay).convert("RGB")
    return combined, len(polygons)


def main():
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = load_data_yaml(data_path)
    images_dir = dataset_root / "images" / args.split
    labels_dir = dataset_root / "labels" / args.split

    if not images_dir.exists():
        print(f"ERROR: images dir not found: {images_dir}")
        return

    image_files = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS])
    if not image_files:
        print(f"No images found in {images_dir}")
        return

    if not args.all:
        random.seed(args.seed)
        image_files = random.sample(image_files, min(args.n, len(image_files)))

    print(f"Annotating {len(image_files)} tiles from {args.split} → {out_dir}/")
    empty_count = 0

    for img_path in sorted(image_files):
        label_path = labels_dir / img_path.with_suffix(".txt").name
        annotated, n_obj = annotate_tile(img_path, label_path)
        out_path = out_dir / img_path.name
        annotated.save(out_path)
        if n_obj == 0:
            empty_count += 1
        print(f"  {img_path.name}: {n_obj} object(s)")

    print(f"\nSaved {len(image_files)} annotated tiles to {out_dir}/")
    print(f"Empty tiles (no labels): {empty_count}/{len(image_files)}")
    print("\nOpen the output PNGs and verify: polygon outlines sit on actual solar array pixels.")
    print("Displaced or stretched polygons = normalization issue still present.")


if __name__ == "__main__":
    main()
