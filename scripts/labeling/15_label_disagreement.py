#!/usr/bin/env python3
"""Test-time disagreement analyzer for YOLO segmentation labels.

Runs the current model on a split, compares per-pixel predicted masks against
hand-drawn label polygons, and buckets each tile so we can spot tiles where
the model 'sees' arrays the human missed (confident_FP) or vice versa (confident_FN).

Outputs:
  outputs/eval/label_disagreement_<split>.csv
  outputs/label_viz/disagreement/<tile>.png  (top-N most-suspicious tiles)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.tile_metadata import TileIndex
from src.utils.train_utils import resolve_weights, select_device
from src.utils.overlay_render import (
    parse_label_polys,
    polys_to_mask,
    mask_iou,
    render_overlay,
)

try:
    from ultralytics import YOLO
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("Run: pip install -r requirements.txt") from exc


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

# Bucketing thresholds (single source of truth)
HIGH_CONF = 0.50
AGREE_IOU = 0.50
MIN_OVERLAP_IOU = 0.30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=str, default=None, help="Path to YOLO weights (.pt)")
    p.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "yolo" / "naip" / "data.yaml")
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--conf", type=float, default=0.15, help="Inference confidence floor")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--top-n-overlays", type=int, default=30,
                   help="Render this many disagreement-overlay PNGs (highest-scored tiles)")
    p.add_argument("--esri-inset", action="store_true",
                   help="Attach Esri high-res reference inset (requires internet, license is adjudication-only)")
    p.add_argument("--tile-index", type=Path, default=REPO_ROOT / "data" / "interim" / "tile_index.json")
    p.add_argument("--output-csv", type=Path,
                   default=REPO_ROOT / "outputs" / "eval" / "label_disagreement.csv")
    p.add_argument("--overlay-dir", type=Path,
                   default=REPO_ROOT / "outputs" / "label_viz" / "disagreement")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N tiles (for smoke testing)")
    return p.parse_args()


def load_data_root(data_yaml: Path) -> Path:
    with data_yaml.open() as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root


def bucket_tile(label_mask: np.ndarray, pred_masks: List[np.ndarray], pred_confs: List[float],
                label_polys: List[np.ndarray]) -> Tuple[str, dict]:
    """Decide a single bucket per tile based on per-polygon comparisons.

    Returns (bucket, stats_dict).
    """
    union_pred = (np.any(np.stack(pred_masks), axis=0) if pred_masks
                  else np.zeros_like(label_mask, dtype=bool))

    overall_iou = mask_iou(label_mask, union_pred)
    n_label_polys = len(label_polys)
    n_pred = len(pred_masks)

    # confident_FP: model has at least one mask with conf > HIGH_CONF that doesn't intersect any label
    label_present = label_mask.any()
    confident_fps = 0
    for m, c in zip(pred_masks, pred_confs):
        if c < HIGH_CONF:
            continue
        if not np.logical_and(m, label_mask).any():
            confident_fps += 1

    # confident_FN: a label polygon has no overlapping prediction at IoU > MIN_OVERLAP_IOU
    label_polys_uncovered = 0
    for lp in label_polys:
        lp_mask = polys_to_mask([lp], label_mask.shape[0], label_mask.shape[1])
        best = max((mask_iou(lp_mask, pm) for pm in pred_masks), default=0.0)
        if best < MIN_OVERLAP_IOU:
            label_polys_uncovered += 1

    if overall_iou >= AGREE_IOU and confident_fps == 0 and label_polys_uncovered == 0:
        bucket = "agree"
    elif confident_fps > 0 and label_polys_uncovered == 0 and not label_present:
        bucket = "confident_FP"
    elif label_polys_uncovered > 0 and confident_fps == 0:
        bucket = "confident_FN"
    elif confident_fps > 0 and label_polys_uncovered > 0:
        bucket = "both_directions"
    elif confident_fps > 0 and label_present:
        bucket = "confident_FP"  # extra detection on a tile that also has labeled arrays
    else:
        bucket = "noisy"

    score = confident_fps + label_polys_uncovered  # higher = more interesting to review

    return bucket, {
        "overall_iou": round(overall_iou, 3),
        "n_label_polys": n_label_polys,
        "n_pred": n_pred,
        "confident_fps": confident_fps,
        "label_polys_uncovered": label_polys_uncovered,
        "max_pred_conf": round(max(pred_confs), 3) if pred_confs else 0.0,
        "disagreement_score": score,
    }


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def main() -> int:
    args = parse_args()
    weights_path = resolve_weights(args.weights, REPO_ROOT)

    data_root = load_data_root(args.data)
    images_dir = data_root / "images" / args.split
    labels_dir = data_root / "labels" / args.split
    if not images_dir.exists():
        print(f"ERROR: images dir not found: {images_dir}", file=sys.stderr)
        return 1

    args.output_csv = Path(args.output_csv).resolve()
    args.overlay_dir = Path(args.overlay_dir).resolve()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.overlay_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if args.limit:
        image_files = image_files[: args.limit]
    if not image_files:
        print(f"No images in {images_dir}", file=sys.stderr)
        return 1

    print(f"Loading {weights_path}")
    device = select_device()
    model = YOLO(str(weights_path), task="segment")

    print(f"Running disagreement analysis: {len(image_files)} {args.split} tiles, conf>={args.conf}")

    tile_index = TileIndex(tile_index_path=args.tile_index) if args.tile_index.exists() else None

    rows: list[dict] = []
    overlay_payloads: list[tuple] = []  # (score, img_path, label_polys, pred_masks, pred_confs, stats, bucket, source)

    for idx, img_path in enumerate(image_files, 1):
        result = model.predict(
            source=str(img_path), imgsz=args.imgsz, conf=args.conf,
            device=device, verbose=False,
        )[0]

        img_w, img_h = Image.open(img_path).size
        label_polys = parse_label_polys(labels_dir / (img_path.stem + ".txt"), img_w, img_h)
        label_mask = polys_to_mask(label_polys, img_h, img_w)

        pred_masks: List[np.ndarray] = []
        pred_confs: List[float] = []
        if result.masks is not None and result.masks.data is not None and len(result.masks.data) > 0:
            mask_tensor = result.masks.data.cpu().numpy()  # (N, H', W'), bool/float
            confs = result.boxes.conf.cpu().numpy().tolist() if result.boxes is not None else [0.0] * len(mask_tensor)
            for m, c in zip(mask_tensor, confs):
                m_bool = m > 0.5 if m.dtype != bool else m
                if m_bool.shape != (img_h, img_w):
                    m_img = Image.fromarray((m_bool.astype(np.uint8) * 255), mode="L").resize((img_w, img_h), Image.NEAREST)
                    m_bool = np.array(m_img) > 127
                pred_masks.append(m_bool)
                pred_confs.append(float(c))

        bucket, stats = bucket_tile(label_mask, pred_masks, pred_confs, label_polys)

        canonical = img_path.name
        source = None
        if tile_index is not None:
            canonical_lookup, entry = tile_index.lookup_by_stripped_name(img_path.name)
            if entry is not None:
                canonical = canonical_lookup
                source = entry.get("source")

        row = {
            "tile_id": img_path.name,
            "canonical_tile": canonical,
            "source_ortho": source or "",
            "bucket": bucket,
            **stats,
        }
        rows.append(row)
        overlay_payloads.append(
            (stats["disagreement_score"], img_path, label_polys, pred_masks, pred_confs, stats, bucket, entry if tile_index else None)
        )

        if idx % 20 == 0 or idx == len(image_files):
            print(f"  [{idx}/{len(image_files)}] last bucket={bucket} score={stats['disagreement_score']}")

    # Write CSV
    fieldnames = list(rows[0].keys())
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows → {_rel(args.output_csv)}")

    # Bucket summary
    from collections import Counter
    bucket_counts = Counter(r["bucket"] for r in rows)
    print("\nBucket summary:")
    for b, c in sorted(bucket_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {b:<18} {c:>4}  ({100 * c / len(rows):.1f}%)")

    # Top-N overlays
    overlay_payloads.sort(key=lambda x: x[0], reverse=True)
    n_render = min(args.top_n_overlays, len(overlay_payloads))
    print(f"\nRendering top-{n_render} disagreement overlays → {_rel(args.overlay_dir)}/")

    esri_fn = None
    if args.esri_inset:
        from src.utils.esri_imagery import fetch_with_retry
        from rasterio.transform import Affine

        def esri_for_tile(entry):
            if not entry:
                return None
            bounds = entry.get("bounds")
            if bounds:
                bb = (bounds["minx"], bounds["miny"], bounds["maxx"], bounds["maxy"])
            else:
                t = Affine(*entry["transform"])
                minx, maxy = t.c, t.f
                maxx = minx + entry["width"] * t.a
                miny = maxy + entry["height"] * t.e
                bb = (minx, miny, maxx, maxy)
            return fetch_with_retry(bb, size=512, bbox_crs=entry.get("crs", "EPSG:3857"))

        esri_fn = esri_for_tile

    for score, img_path, lps, pms, pcs, stats, bucket, entry in overlay_payloads[:n_render]:
        esri_path = esri_fn(entry) if esri_fn else None
        canvas = render_overlay(img_path, lps, pms, pcs, stats, bucket, esri_path)
        canvas.save(args.overlay_dir / img_path.name)

    print(f"Saved {n_render} overlays.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
