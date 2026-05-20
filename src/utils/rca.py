"""Core RCA (root-cause analysis) logic for per-detection TP/FP/FN inspection.

Extracted from scripts/05c_per_detection_rca.py so that both 05c (CLI wrapper)
and 05d (SAHI threshold sweep) can import cleanly without the importlib hack
that digit-prefixed script names require.

Public API
----------
run_rca_pass(...)    -> Path   runs inference, writes per_detection.csv
summarize_rca(...)   -> dict   aggregates failure modes → failure_modes.json
render_standard_buckets(...)   renders confident_fp / worst_small_fn / large_fp
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.det_match import match_predictions

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


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

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


def _round(v: float) -> float | str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return ""
    return round(v, 3)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_sahi_on_tile(
    detection_model,
    img_path: Path,
    slice_size: int,
    overlap: float,
    match_iou: float,
    perform_standard_pred: bool = False,
):
    """Return (predictions, img_w, img_h) with pixel-coord bbox/centroid/area/aspect/conf."""
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
        if pred.mask is not None and pred.mask.segmentation:
            seg = pred.mask.segmentation[0]
            if len(seg) >= 6:
                xs = np.array(seg[0::2], dtype=np.float64)
                ys = np.array(seg[1::2], dtype=np.float64)
                poly = np.column_stack([xs, ys])
                bbox, cx, cy, area, aspect = poly_to_bbox_centroid_area_aspect(poly)
                out.append({"conf": conf, "bbox": bbox, "cx": cx, "cy": cy,
                            "area_px": area, "aspect": aspect})
                continue
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
        for poly, c in zip(result.masks.xy, confs):
            if len(poly) < 3:
                continue
            poly = np.asarray(poly, dtype=np.float64)
            bbox, cx, cy, area, aspect = poly_to_bbox_centroid_area_aspect(poly)
            out.append({"conf": float(c), "bbox": bbox, "cx": cx, "cy": cy,
                        "area_px": area, "aspect": aspect})
    out.sort(key=lambda d: -d["conf"])
    return out, img_w, img_h


def gt_records(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    out = []
    for poly_norm in parse_label_polys_norm(label_path):
        poly_px = poly_norm * np.array([img_w, img_h])
        bbox, cx, cy, area, aspect = poly_to_bbox_centroid_area_aspect(poly_px)
        out.append({"bbox": bbox, "cx": cx, "cy": cy, "area_px": area, "aspect": aspect})
    return out


def detection_rows(
    tile_id: str, domain: str, split: str, weights_run: str,
    preds: list[dict], gts: list[dict],
    img_w: int, img_h: int, iou_thresh: float,
) -> list[dict]:
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
            "pred_area_px": round(p["area_px"], 2), "gt_area_px": round(g["area_px"], 2),
            "pred_centroid_x": round(p["cx"], 2), "pred_centroid_y": round(p["cy"], 2),
            "gt_centroid_x": round(g["cx"], 2), "gt_centroid_y": round(g["cy"], 2),
            "distance_to_image_edge_px": round(edge_distance(g["cx"], g["cy"], img_w, img_h), 2),
            "num_other_panels_in_tile": max(0, n_panels - 1),
            "pred_aspect": _round(p["aspect"]), "gt_aspect": _round(g["aspect"]),
            "weights_run": weights_run,
        })

    for p_idx in match.fp:
        p = preds[p_idx]
        rows.append({
            "tile_id": tile_id, "domain": domain, "split": split,
            "class": "fp", "iou": "",
            "conf": round(p["conf"], 4),
            "pred_area_px": round(p["area_px"], 2), "gt_area_px": "",
            "pred_centroid_x": round(p["cx"], 2), "pred_centroid_y": round(p["cy"], 2),
            "gt_centroid_x": "", "gt_centroid_y": "",
            "distance_to_image_edge_px": round(edge_distance(p["cx"], p["cy"], img_w, img_h), 2),
            "num_other_panels_in_tile": n_panels,
            "pred_aspect": _round(p["aspect"]), "gt_aspect": "",
            "weights_run": weights_run,
        })

    for g_idx in match.fn:
        g = gts[g_idx]
        rows.append({
            "tile_id": tile_id, "domain": domain, "split": split,
            "class": "fn", "iou": "", "conf": "",
            "pred_area_px": "", "gt_area_px": round(g["area_px"], 2),
            "pred_centroid_x": "", "pred_centroid_y": "",
            "gt_centroid_x": round(g["cx"], 2), "gt_centroid_y": round(g["cy"], 2),
            "distance_to_image_edge_px": round(edge_distance(g["cx"], g["cy"], img_w, img_h), 2),
            "num_other_panels_in_tile": max(0, n_panels - 1),
            "pred_aspect": "", "gt_aspect": _round(g["aspect"]),
            "weights_run": weights_run,
        })
    return rows


# ---------------------------------------------------------------------------
# High-level inference pass
# ---------------------------------------------------------------------------

def run_rca_pass(
    weights_path: Path,
    data_yaml: Path,
    splits: list[str],
    out_dir: Path,
    *,
    sahi: bool = True,
    conf: float = 0.05,
    iou: float = 0.5,
    slice_size: int = 640,
    overlap: float = 0.2,
    imgsz: int = 640,
    limit: int | None = None,
    standard_pred: bool = False,
) -> Path:
    """Run inference + matching for all splits, write per_detection.csv, return its path."""
    from src.utils.train_utils import select_device

    weights_run = (
        weights_path.parent.parent.name
        if weights_path.parent.name == "weights"
        else weights_path.stem
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_detection.csv"

    data_root = load_data_root(data_yaml)
    domain = domain_from_data_yaml(data_yaml)
    device = select_device()

    detection_model = None
    yolo_model = None
    if sahi:
        try:
            from sahi import AutoDetectionModel
        except ImportError as exc:
            raise ImportError("Run: pip install sahi") from exc
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=str(weights_path),
            confidence_threshold=conf, device=device,
        )
    else:
        from ultralytics import YOLO
        yolo_model = YOLO(str(weights_path), task="segment")

    rows: list[dict] = []
    for split in splits:
        img_dir = data_root / "images" / split
        lbl_dir = data_root / "labels" / split
        if not img_dir.exists():
            print(f"  skip split {split}: {img_dir} missing", file=sys.stderr)
            continue
        tiles = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if limit:
            tiles = tiles[:limit]
        print(f"[{split}] {len(tiles)} tiles  weights={weights_path.name}  sahi={sahi}")
        for idx, img_path in enumerate(tiles, 1):
            if sahi:
                preds, img_w, img_h = run_sahi_on_tile(
                    detection_model, img_path, slice_size, overlap, iou,
                    perform_standard_pred=standard_pred)
            else:
                preds, img_w, img_h = run_standard_on_tile(yolo_model, img_path, imgsz, conf, device)
            gts = gt_records(lbl_dir / (img_path.stem + ".txt"), img_w, img_h)
            rows.extend(detection_rows(
                tile_id=img_path.name, domain=domain, split=split,
                weights_run=weights_run, preds=preds, gts=gts,
                img_w=img_w, img_h=img_h, iou_thresh=iou,
            ))
            if idx % 20 == 0 or idx == len(tiles):
                print(f"  [{idx}/{len(tiles)}]")

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows → {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Failure-mode summary
# ---------------------------------------------------------------------------

def summarize_rca(csv_path: Path, out_json: Path) -> dict:
    """Aggregate per_detection.csv into a failure-mode summary.

    Buckets by panel size (small <800 px², medium <4000, large >=4000), edge
    distance (near <40 px, mid <120, far), and density (alone, 2-3, ≥4).
    Reports FN rate per panel-size bucket and FP rate per edge bucket.
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
        "recall": (overall["tp"] / (overall["tp"] + overall["fn"])
                   if (overall["tp"] + overall["fn"]) else float("nan")),
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

    sizes = summary["by_panel_size"]
    if "small_lt800" in sizes and "large_ge4000" in sizes:
        s = sizes["small_lt800"]["fn_rate"]
        lg = sizes["large_ge4000"]["fn_rate"]
        if not math.isnan(s) and not math.isnan(lg) and lg > 0:
            summary["headline"] = (
                f"FN rate is {s/lg:.2f}× higher for arrays under 800 px² "
                f"({s:.2%} vs {lg:.2%}) — model is missing small panels."
            )

    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# ---------------------------------------------------------------------------
# Bucket overlay rendering
# ---------------------------------------------------------------------------

def render_standard_buckets(
    csv_path: Path,
    data_yaml: Path,
    top: int = 20,
) -> None:
    """Render confident_fp / worst_small_fn / large_fp overlay PNGs.

    Invoked by 05c --render-buckets. Output lands in the same directory
    structure as a manual 18_bucket_overlays.py run.
    """
    import importlib.util
    _18_path = REPO_ROOT / "scripts" / "labeling" / "18_bucket_overlays.py"
    if not _18_path.exists():
        print(f"[render_standard_buckets] 18_bucket_overlays.py not found at {_18_path}",
              file=sys.stderr)
        return
    spec = importlib.util.spec_from_file_location("bucket_overlays_18", str(_18_path))
    if spec is None or spec.loader is None:
        return
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for bucket in ("confident_fp", "worst_small_fn", "large_fp"):
        print(f"  rendering bucket: {bucket}")
        argv = [
            "--csv", str(csv_path),
            "--bucket", bucket,
            "--top", str(top),
            "--data", str(data_yaml),
        ]
        try:
            mod.main(argv)
        except SystemExit:
            pass
        except Exception as exc:
            print(f"  [warn] bucket {bucket} failed: {exc}", file=sys.stderr)
