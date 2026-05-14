#!/usr/bin/env python3
"""Per-detection root-cause analysis (RCA) harness — Tyler's TP / FN / FP framing.

The load-bearing piece of the diagnose-first plan. For every detection-or-miss
on the chosen split, emit one CSV row classifying it as TP, FP, or FN with
enough metadata (size, position, density, confidence) to find the failure-mode
pattern. Owns its own SAHI inference loop because `04_infer_yolov8_seg.py`
discards the per-polygon confidence (`pred.score.value`) when it serializes to
.txt — and that confidence is what makes "show me the most-confident FPs"
possible.

Outputs:
  outputs/eval/<run-name>/per_detection.csv
  outputs/eval/<run-name>/failure_modes.json   (when --summarize is set)
  outputs/eval/<run-name>/manifest.json

Cross-check: aggregating per_detection.csv by (tile_id, class) reproduces
`compute_sahi_confusion_matrix.py`'s per-image TP/FP/FN counts. If it doesn't,
the matcher has drifted and the rest of Phase 1 is unreliable.

Usage:
  # Full SAHI RCA on NAIP val+test
  python scripts/05c_per_detection_rca.py \\
      --weights models/sahi_baseline_train7.pt \\
      --data data/yolo/naip/data.yaml --splits val test \\
      --sahi --conf 0.05 --run-name sahi_baseline_train7

  # Aggregate failure-mode patterns from an existing CSV
  python scripts/05c_per_detection_rca.py --summarize \\
      --csv outputs/eval/sahi_baseline_train7/per_detection.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.solarsoiled.manifest import write_manifest
from src.utils.det_match import match_predictions
from src.utils.train_utils import resolve_weights, select_device

CSV_FIELDS = [
    "tile_id", "domain", "split", "class", "iou", "conf",
    "pred_area_px", "gt_area_px",
    "pred_centroid_x", "pred_centroid_y",
    "gt_centroid_x", "gt_centroid_y",
    "distance_to_image_edge_px",
    "num_other_panels_in_tile",
    "pred_aspect", "gt_aspect",
    "weights_run",
]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "yolo" / "naip" / "data.yaml")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--sahi", action="store_true",
                   help="Use SAHI sliced inference (preserves small-array recall)")
    p.add_argument("--standard-pred", action="store_true",
                   help="Also run full-tile inference and merge with slice results (SAHI combined mode)")
    p.add_argument("--slice", type=int, default=640, help="SAHI slice size")
    p.add_argument("--overlap", type=float, default=0.2, help="SAHI overlap ratio")
    p.add_argument("--conf", type=float, default=0.05,
                   help="Inference confidence floor (low so 05d can re-threshold)")
    p.add_argument("--iou", type=float, default=0.5, help="Matching IoU threshold")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--run-name", type=str, default=None,
                   help="Output subdirectory under outputs/eval/. Defaults to weights stem.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N tiles per split (smoke testing)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override output directory. Default: outputs/eval/<run-name>/")
    p.add_argument("--summarize", action="store_true",
                   help="Skip inference; aggregate failure modes from --csv")
    p.add_argument("--csv", type=Path, default=None,
                   help="Path to existing per_detection.csv (with --summarize)")
    return p.parse_args(argv)


def load_data_root(data_yaml: Path) -> Path:
    with data_yaml.open() as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root


def domain_from_data_yaml(data_yaml: Path) -> str:
    parent = data_yaml.parent.name.lower()
    if "duke" in parent:
        return "duke"
    if "naip" in parent or parent == "yolo":
        return "naip"
    return parent


def parse_label_polys_norm(label_path: Path) -> list[np.ndarray]:
    """Return list of (N,2) polygons in normalized [0,1] coords."""
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            coords = np.array([float(p) for p in parts[1:]], dtype=np.float64)
        except ValueError:
            continue
        xs = coords[0::2]
        ys = coords[1::2]
        if len(xs) < 3:
            continue
        out.append(np.column_stack([xs, ys]))
    return out


def poly_to_bbox_centroid_area_aspect(poly_xy: np.ndarray) -> tuple:
    xs, ys = poly_xy[:, 0], poly_xy[:, 1]
    bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    cx = float(xs.mean())
    cy = float(ys.mean())
    area = 0.5 * abs(float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    aspect = (max(bbox_w, bbox_h) / min(bbox_w, bbox_h)) if min(bbox_w, bbox_h) > 0 else float("nan")
    return bbox, cx, cy, area, aspect


def edge_distance(cx: float, cy: float, w: int, h: int) -> float:
    return float(min(cx, cy, w - cx, h - cy))


def run_sahi_on_tile(detection_model, img_path: Path, slice_size: int, overlap: float,
                     match_iou: float, perform_standard_pred: bool = False):
    """Returns list of dicts with bbox / centroid / area / aspect / conf in pixel coords."""
    from sahi.predict import get_sliced_prediction

    img_w, img_h = Image.open(img_path).size
    result = get_sliced_prediction(
        str(img_path), detection_model,
        slice_height=slice_size, slice_width=slice_size,
        overlap_height_ratio=overlap, overlap_width_ratio=overlap,
        postprocess_type="GREEDYNMM", postprocess_match_metric="IOS",
        postprocess_match_threshold=match_iou,
        perform_standard_pred=perform_standard_pred,
        verbose=0,
    )
    out = []
    for pred in result.object_prediction_list:
        conf = float(pred.score.value)
        # Polygon path (preferred — gives true area/centroid)
        if pred.mask is not None and pred.mask.segmentation:
            seg = pred.mask.segmentation[0]
            if len(seg) >= 6:
                xs = np.array(seg[0::2], dtype=np.float64)
                ys = np.array(seg[1::2], dtype=np.float64)
                poly = np.column_stack([xs, ys])
                bbox, cx, cy, area, aspect = poly_to_bbox_centroid_area_aspect(poly)
                out.append({
                    "conf": conf,
                    "bbox": bbox,
                    "cx": cx, "cy": cy,
                    "area_px": area,
                    "aspect": aspect,
                })
                continue
        # Bbox fallback (no mask)
        b = pred.bbox.to_xyxy()
        bbox = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        out.append({
            "conf": conf, "bbox": bbox, "cx": cx, "cy": cy,
            "area_px": float(bw * bh),
            "aspect": (max(bw, bh) / min(bw, bh)) if min(bw, bh) > 0 else float("nan"),
        })
    # Sort by descending conf so the matcher claims the best preds first
    out.sort(key=lambda d: -d["conf"])
    return out, img_w, img_h


def run_standard_on_tile(model, img_path: Path, imgsz: int, conf: float, device: str):
    """Standard ultralytics inference path (no SAHI)."""
    result = model.predict(source=str(img_path), imgsz=imgsz, conf=conf,
                           device=device, verbose=False)[0]
    img_w, img_h = Image.open(img_path).size
    out = []
    if result.masks is not None and result.masks.data is not None and len(result.masks.data) > 0:
        confs = (result.boxes.conf.cpu().numpy().tolist()
                 if result.boxes is not None
                 else [0.0] * len(result.masks.data))
        # ultralytics .masks.xy: list of (N,2) arrays in pixel coords (image scale)
        for poly, c in zip(result.masks.xy, confs):
            if len(poly) < 3:
                continue
            poly = np.asarray(poly, dtype=np.float64)
            bbox, cx, cy, area, aspect = poly_to_bbox_centroid_area_aspect(poly)
            out.append({
                "conf": float(c), "bbox": bbox, "cx": cx, "cy": cy,
                "area_px": area, "aspect": aspect,
            })
    out.sort(key=lambda d: -d["conf"])
    return out, img_w, img_h


def gt_records(label_path: Path, img_w: int, img_h: int):
    out = []
    for poly_norm in parse_label_polys_norm(label_path):
        poly_px = poly_norm * np.array([img_w, img_h])
        bbox, cx, cy, area, aspect = poly_to_bbox_centroid_area_aspect(poly_px)
        out.append({"bbox": bbox, "cx": cx, "cy": cy, "area_px": area, "aspect": aspect})
    return out


def detection_rows(tile_id: str, domain: str, split: str, weights_run: str,
                   preds: list[dict], gts: list[dict],
                   img_w: int, img_h: int, iou_thresh: float) -> list[dict]:
    """Build TP/FP/FN rows for a single tile using the shared matcher."""
    pred_bboxes = [p["bbox"] for p in preds]
    gt_bboxes = [g["bbox"] for g in gts]
    match = match_predictions(pred_bboxes, gt_bboxes, iou_thresh=iou_thresh)
    rows: list[dict] = []
    n_panels = len(gts)

    for p_idx, g_idx, iou_val in match.tp:
        p, g = preds[p_idx], gts[g_idx]
        rows.append({
            "tile_id": tile_id, "domain": domain, "split": split,
            "class": "tp", "iou": round(iou_val, 4),
            "conf": round(p["conf"], 4),
            "pred_area_px": round(p["area_px"], 2),
            "gt_area_px": round(g["area_px"], 2),
            "pred_centroid_x": round(p["cx"], 2),
            "pred_centroid_y": round(p["cy"], 2),
            "gt_centroid_x": round(g["cx"], 2),
            "gt_centroid_y": round(g["cy"], 2),
            "distance_to_image_edge_px": round(edge_distance(g["cx"], g["cy"], img_w, img_h), 2),
            "num_other_panels_in_tile": max(0, n_panels - 1),
            "pred_aspect": _round(p["aspect"]),
            "gt_aspect": _round(g["aspect"]),
            "weights_run": weights_run,
        })

    for p_idx in match.fp:
        p = preds[p_idx]
        rows.append({
            "tile_id": tile_id, "domain": domain, "split": split,
            "class": "fp", "iou": "",
            "conf": round(p["conf"], 4),
            "pred_area_px": round(p["area_px"], 2),
            "gt_area_px": "",
            "pred_centroid_x": round(p["cx"], 2),
            "pred_centroid_y": round(p["cy"], 2),
            "gt_centroid_x": "", "gt_centroid_y": "",
            "distance_to_image_edge_px": round(edge_distance(p["cx"], p["cy"], img_w, img_h), 2),
            "num_other_panels_in_tile": n_panels,
            "pred_aspect": _round(p["aspect"]),
            "gt_aspect": "",
            "weights_run": weights_run,
        })

    for g_idx in match.fn:
        g = gts[g_idx]
        rows.append({
            "tile_id": tile_id, "domain": domain, "split": split,
            "class": "fn", "iou": "",
            "conf": "",
            "pred_area_px": "",
            "gt_area_px": round(g["area_px"], 2),
            "pred_centroid_x": "", "pred_centroid_y": "",
            "gt_centroid_x": round(g["cx"], 2),
            "gt_centroid_y": round(g["cy"], 2),
            "distance_to_image_edge_px": round(edge_distance(g["cx"], g["cy"], img_w, img_h), 2),
            "num_other_panels_in_tile": max(0, n_panels - 1),
            "pred_aspect": "",
            "gt_aspect": _round(g["aspect"]),
            "weights_run": weights_run,
        })
    return rows


def _round(v: float) -> float | str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return ""
    return round(v, 3)


def summarize(csv_path: Path, out_json: Path) -> dict:
    """Aggregate per_detection.csv into a failure-mode summary.

    Buckets by panel size (small <800 px², medium <4000, large >=4000), edge
    distance (near <40 px, mid <120, far), and density (alone, 2-3, ≥4).
    Reports FN rate per panel-size bucket and FP rate per edge bucket — those
    are the two patterns the meeting flagged as most likely.
    """
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        out_json.write_text(json.dumps({"error": "empty CSV"}) + "\n")
        return {}

    def num(x: str, default=float("nan")) -> float:
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    summary: dict[str, Any] = {"n_rows": len(rows)}

    # Panel-size buckets — use whichever area the row has (gt for tp/fn, pred for fp)
    def size_bucket(row) -> str:
        a = num(row.get("gt_area_px") or row.get("pred_area_px") or "nan")
        if math.isnan(a):
            return "unknown"
        if a < 800:
            return "small_lt800"
        if a < 4000:
            return "medium_800_4000"
        return "large_ge4000"

    def edge_bucket(row) -> str:
        d = num(row.get("distance_to_image_edge_px"))
        if math.isnan(d):
            return "unknown"
        if d < 40:
            return "near_lt40"
        if d < 120:
            return "mid_40_120"
        return "far_ge120"

    def density_bucket(row) -> str:
        n = num(row.get("num_other_panels_in_tile"))
        if math.isnan(n):
            return "unknown"
        if n == 0:
            return "alone"
        if n < 4:
            return "few_1_3"
        return "crowded_ge4"

    overall = {"tp": 0, "fp": 0, "fn": 0}
    by_size: dict[str, dict] = {}
    by_edge: dict[str, dict] = {}
    by_density: dict[str, dict] = {}
    for r in rows:
        cls = r["class"]
        if cls in overall:
            overall[cls] += 1
        for d, key in ((by_size, size_bucket(r)), (by_edge, edge_bucket(r)),
                       (by_density, density_bucket(r))):
            d.setdefault(key, {"tp": 0, "fp": 0, "fn": 0})
            if cls in d[key]:
                d[key][cls] += 1

    def fn_rate(d: dict) -> float:
        denom = d["tp"] + d["fn"]
        return d["fn"] / denom if denom else float("nan")

    def precision(d: dict) -> float:
        denom = d["tp"] + d["fp"]
        return d["tp"] / denom if denom else float("nan")

    summary["overall"] = {
        **overall,
        "precision": precision(overall),
        "recall": (overall["tp"] / (overall["tp"] + overall["fn"])) if (overall["tp"] + overall["fn"]) else float("nan"),
    }
    summary["by_panel_size"] = {
        k: {**v, "fn_rate": fn_rate(v), "precision": precision(v)} for k, v in by_size.items()
    }
    summary["by_edge_distance"] = {
        k: {**v, "fn_rate": fn_rate(v), "precision": precision(v)} for k, v in by_edge.items()
    }
    summary["by_density"] = {
        k: {**v, "fn_rate": fn_rate(v), "precision": precision(v)} for k, v in by_density.items()
    }

    # Surface the loudest pattern: FN-rate ratio between size buckets
    sizes = summary["by_panel_size"]
    if "small_lt800" in sizes and "large_ge4000" in sizes:
        s = sizes["small_lt800"]["fn_rate"]
        l = sizes["large_ge4000"]["fn_rate"]
        if not math.isnan(s) and not math.isnan(l) and l > 0:
            summary["headline"] = (
                f"FN rate is {s/l:.2f}× higher for arrays under 800 px² "
                f"({s:.2%} vs {l:.2%}) — model is missing small panels."
            )

    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.summarize:
        if args.csv is None:
            print("--summarize requires --csv", file=sys.stderr)
            return 1
        out_json = args.csv.parent / "failure_modes.json"
        summary = summarize(args.csv, out_json)
        if "headline" in summary:
            print(summary["headline"])
        print(f"Wrote {out_json}")
        return 0

    weights_path = resolve_weights(args.weights, REPO_ROOT)
    weights_run = (weights_path.parent.parent.name
                   if weights_path.parent.name == "weights"
                   else weights_path.stem)
    run_name = args.run_name or weights_run
    out_dir = args.out_dir or (REPO_ROOT / "outputs" / "eval" / run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_detection.csv"

    data_root = load_data_root(args.data)
    domain = domain_from_data_yaml(args.data)
    device = select_device()

    detection_model = None
    yolo_model = None
    if args.sahi:
        try:
            from sahi import AutoDetectionModel
        except ImportError as exc:
            raise ImportError("Run: pip install sahi") from exc
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=str(weights_path),
            confidence_threshold=args.conf, device=device,
        )
    else:
        from ultralytics import YOLO
        yolo_model = YOLO(str(weights_path), task="segment")

    rows: list[dict] = []
    n_tiles_processed = 0
    for split in args.splits:
        img_dir = data_root / "images" / split
        lbl_dir = data_root / "labels" / split
        if not img_dir.exists():
            print(f"  skip split {split}: {img_dir} missing", file=sys.stderr)
            continue
        tiles = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if args.limit:
            tiles = tiles[: args.limit]
        print(f"[{split}] {len(tiles)} tiles, weights={weights_path.name}, sahi={args.sahi}")
        for idx, img_path in enumerate(tiles, 1):
            if args.sahi:
                preds, img_w, img_h = run_sahi_on_tile(
                    detection_model, img_path, args.slice, args.overlap, args.iou,
                    perform_standard_pred=args.standard_pred)
            else:
                preds, img_w, img_h = run_standard_on_tile(
                    yolo_model, img_path, args.imgsz, args.conf, device)
            gts = gt_records(lbl_dir / (img_path.stem + ".txt"), img_w, img_h)
            rows.extend(detection_rows(
                tile_id=img_path.name, domain=domain, split=split,
                weights_run=weights_run, preds=preds, gts=gts,
                img_w=img_w, img_h=img_h, iou_thresh=args.iou,
            ))
            n_tiles_processed += 1
            if idx % 20 == 0 or idx == len(tiles):
                print(f"  [{idx}/{len(tiles)}]")

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows → {csv_path}")

    # Quick aggregate for the manifest
    counts = {"tp": 0, "fp": 0, "fn": 0}
    for r in rows:
        if r["class"] in counts:
            counts[r["class"]] += 1

    write_manifest(
        out_dir,
        stage="eval",
        model_version=f"rca-{weights_run}",
        model_weights=weights_path,
        inputs=[str(args.data)],
        metrics={
            "n_tiles": n_tiles_processed,
            "n_rows": len(rows),
            "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"],
            "precision": counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) else 0.0,
            "recall": counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) else 0.0,
            "iou_thresh": args.iou,
            "conf_floor": args.conf,
        },
        known_limitations=[
            "Greedy bbox-IoU matching at IoU=0.5 — small misalignments may flip TP↔FP at the boundary",
            "When SAHI is enabled, postprocess_match_metric=IOS means re-thresholding requires re-running SAHI (see 05d)",
        ],
        extra={"sahi": bool(args.sahi), "splits": args.splits},
    )
    print(f"Counts: tp={counts['tp']} fp={counts['fp']} fn={counts['fn']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
