#!/usr/bin/env python3
"""Compute detection confusion counts (TP/FP/FN) for SAHI label predictions.

Matches predicted label files against ground-truth label files by tile token
and reports TP/FP/FN counts per image.

Note: `scripts/05c_per_detection_rca.py` is the richer tool for new work — it
owns its own SAHI inference loop, preserves per-polygon confidence, and emits
one row per detection rather than per-image aggregates. Use this script only
when you already have pre-exported YOLO label predictions from a previous run.

Usage:
  python scripts/compute_sahi_confusion_matrix.py \\
      --pred-dir runs/segment/<run>/labels \\
      --gt-dir data/yolo/naip/labels/val \\
      --out-path outputs/eval/<run>/confusion_matrix.json
"""
import argparse
import json
import os
import re
import sys
from glob import glob
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.det_match import match_predictions

IMG_SIZE = 640
IOU_THRESH = 0.5


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", required=True,
                   help="Directory of YOLO-format prediction .txt files")
    p.add_argument("--gt-dir", required=True,
                   help="Directory of YOLO-format ground-truth .txt files")
    p.add_argument("--out-path", required=True,
                   help="Output JSON path for the confusion matrix results")
    p.add_argument("--iou-thresh", type=float, default=IOU_THRESH,
                   help="IoU threshold for TP matching (default: 0.5)")
    p.add_argument("--img-size", type=int, default=IMG_SIZE,
                   help="Image size used to scale normalized coords (default: 640)")
    return p.parse_args(argv)


def read_polygons_from_yolo_seg(path: str, img_size: int) -> list[tuple]:
    """Return list of bboxes (xmin,ymin,xmax,ymax) from polygon coords in YOLO seg TXT."""
    bboxes = []
    try:
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                coords = list(map(float, parts[1:]))
                xs = [x * img_size for x in coords[0::2]]
                ys = [y * img_size for y in coords[1::2]]
                if not xs or not ys:
                    continue
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                if xmax > xmin and ymax > ymin:
                    bboxes.append((xmin, ymin, xmax, ymax))
    except FileNotFoundError:
        return []
    return bboxes


def main(argv=None) -> None:
    args = parse_args(argv)
    out_path = args.out_path
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    pred_files = sorted(glob(os.path.join(args.pred_dir, "*.txt")))
    gt_files = sorted(glob(os.path.join(args.gt_dir, "**", "*.txt"), recursive=True))

    def token_map(files: list[str]) -> dict[str, str]:
        m = {}
        for f in files:
            match = re.search(r"(tile_\d+)", os.path.basename(f))
            if match:
                m[match.group(1)] = f
        return m

    pred_map = token_map(pred_files)
    gt_map = token_map(gt_files)

    tokens_pred = set(pred_map.keys())
    tokens_gt = set(gt_map.keys())
    common = sorted(tokens_pred & tokens_gt)

    total_tp = total_fp = total_fn = 0
    per_image: dict[str, dict] = {}

    for token in common:
        preds = read_polygons_from_yolo_seg(pred_map[token], args.img_size)
        gts = read_polygons_from_yolo_seg(gt_map[token], args.img_size)
        match = match_predictions(preds, gts, iou_thresh=args.iou_thresh)
        tp, fp, fn = len(match.tp), len(match.fp), len(match.fn)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_image[token] = {"tp": tp, "fp": fp, "fn": fn,
                            "n_pred": len(preds), "n_gt": len(gts)}

    # Unmatched preds → FP; unmatched GTs → FN
    for token in tokens_pred - tokens_gt:
        total_fp += len(read_polygons_from_yolo_seg(pred_map[token], args.img_size))
    for token in tokens_gt - tokens_pred:
        total_fn += len(read_polygons_from_yolo_seg(gt_map[token], args.img_size))

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    out = {
        "pred_dir": args.pred_dir,
        "gt_dir": args.gt_dir,
        "iou_thresh": args.iou_thresh,
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": precision, "recall": recall, "f1": f1,
        "n_pred_images": len(tokens_pred),
        "n_gt_images": len(tokens_gt),
        "n_common_images": len(common),
        "per_image": per_image,
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({"tp": total_tp, "fp": total_fp, "fn": total_fn,
                      "precision": precision, "recall": recall, "f1": f1}, indent=2))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
