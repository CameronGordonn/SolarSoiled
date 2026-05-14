#!/usr/bin/env python3
"""Compute detection confusion counts (TP/FP/FN) for SAHI predictions.

Saves results to `outputs/eval/sahi_baseline_train7/confusion_matrix.json`.

Usage: python scripts/compute_sahi_confusion_matrix.py
"""
import os
import sys
import json
from glob import glob
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.det_match import match_predictions

# Use bbox IoU (no shapely dependency)
IMG_SIZE = 640
IOU_THRESH = 0.5
PRED_DIR = "runs/segment/sahi_baseline_train7/labels"
GT_DIR = "data/yolo/naip/labels/val"
OUT_PATH = "outputs/eval/sahi_baseline_train7/confusion_matrix.json"


def read_polygons_from_yolo_seg(path):
    """Return list of bboxes (xmin,ymin,xmax,ymax) extracted from polygon coords in YOLO seg TXT.
    Uses normalized coords scaled by IMG_SIZE.
    """
    bboxes = []
    try:
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                coords = list(map(float, parts[1:]))
                xs = coords[0::2]
                ys = coords[1::2]
                if not xs or not ys:
                    continue
                xs = [x * IMG_SIZE for x in xs]
                ys = [y * IMG_SIZE for y in ys]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                # sanity
                if xmax > xmin and ymax > ymin:
                    bboxes.append((xmin, ymin, xmax, ymax))
    except FileNotFoundError:
        return []
    return bboxes


def find_gt_file_for_pred(pred_name, gt_files):
    # Extract 'tile_XXXXX' token from pred_name and match GT files starting with it
    import re
    m = re.search(r'(tile_\d+)', pred_name)
    if m:
        token = m.group(1)
        for g in gt_files:
            if os.path.basename(g).startswith(token):
                return g
    # fallback: try any gt file whose basename (without ext) appears in pred_name
    for g in gt_files:
        if os.path.basename(g).split('.')[0] in pred_name:
            return g
    return None


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    pred_files = sorted(glob(os.path.join(PRED_DIR, "*.txt")))
    # search GT across train/val/test
    gt_files = sorted(glob(os.path.join("data/yolo/naip/labels", "**", "*.txt"), recursive=True))

    # map token -> file
    import re
    pred_map = {}
    for p in pred_files:
        m = re.search(r'(tile_\d+)', os.path.basename(p))
        if m:
            pred_map[m.group(1)] = p
    gt_map = {}
    for g in gt_files:
        m = re.search(r'(tile_\d+)', os.path.basename(g))
        if m:
            gt_map[m.group(1)] = g

    tokens_pred = set(pred_map.keys())
    tokens_gt = set(gt_map.keys())
    common = sorted(list(tokens_pred & tokens_gt))

    total_tp = total_fp = total_fn = 0
    per_image = {}

    # compute for intersection only
    for token in common:
        p = pred_map[token]
        g = gt_map[token]
        preds = read_polygons_from_yolo_seg(p)
        gts = read_polygons_from_yolo_seg(g)

        match = match_predictions(preds, gts, iou_thresh=IOU_THRESH)
        tp = len(match.tp)
        fp = len(match.fp)
        fn = len(match.fn)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        per_image[token] = {"tp": tp, "fp": fp, "fn": fn, "n_pred": len(preds), "n_gt": len(gts)}

    # unmatched preds (no GT) -> count as FP
    unmatched_preds = tokens_pred - tokens_gt
    unmatched_fp = 0
    for token in unmatched_preds:
        preds = read_polygons_from_yolo_seg(pred_map[token])
        unmatched_fp += len(preds)

    # unmatched gts (no pred) -> count as FN
    unmatched_gts = tokens_gt - tokens_pred
    unmatched_fn = 0
    for token in unmatched_gts:
        gts = read_polygons_from_yolo_seg(gt_map[token])
        unmatched_fn += len(gts)

    total_fp += unmatched_fp
    total_fn += unmatched_fn

    summary_extra = {
        "n_pred_images": len(tokens_pred),
        "n_gt_images": len(tokens_gt),
        "n_common_images": len(common),
        "unmatched_pred_images": sorted(list(unmatched_preds))[:20],
        "unmatched_gt_images": sorted(list(unmatched_gts))[:20],
        "unmatched_fp": unmatched_fp,
        "unmatched_fn": unmatched_fn,
    }

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    out = {
        "weights": "models/sahi_baseline_train7.pt",
        "iou_thresh": IOU_THRESH,
        "conf_thresh": 0.1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_image": per_image,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({"tp": total_tp, "fp": total_fp, "fn": total_fn, "precision": precision, "recall": recall, "f1": f1}, indent=2))


if __name__ == "__main__":
    main()
