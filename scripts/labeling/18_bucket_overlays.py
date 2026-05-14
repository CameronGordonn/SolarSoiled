#!/usr/bin/env python3
"""Render top-N tile overlays per failure-mode bucket.

Tyler's "show me the worst small-array misses" / "show me the most-confident
false positives" task. Reads `per_detection.csv` from `05c`, picks tiles in a
named or expression-based bucket, and renders a label-vs-prediction overlay
PNG per tile using the shared helpers in `src/utils/overlay_render.py`.

Bucket presets:
  worst_small_fn    class=fn AND gt_area_px<800             (top by gt_area_px asc)
  edge_fn           class=fn AND distance_to_image_edge_px<40 (top by distance asc)
  confident_fp      class=fp                                  (top by conf desc)
  large_fp          class=fp AND pred_area_px>=4000           (top by pred_area_px desc)
  worst_iou_tp      class=tp                                  (top by iou asc)

Or pass a free-form expression with --bucket-expr, e.g.:
  --bucket-expr "class=fp AND conf>0.5"

Output: outputs/label_viz/<run-name>/<bucket>/<tile>.png + manifest.json.

This script does NOT re-run the model — it consumes the CSV from 05c. So one
SAHI inference pass produces the per_detection.csv, and 18 can be invoked
many times with different bucket queries against the same CSV.

Usage:
  python scripts/labeling/18_bucket_overlays.py \\
      --csv outputs/eval/sahi_baseline_train7/per_detection.csv \\
      --bucket worst_small_fn --top 20
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.solarsoiled.manifest import write_manifest
from src.utils.overlay_render import (
    parse_label_polys,
    polys_to_mask,
    render_overlay,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


PRESETS: dict[str, dict] = {
    "worst_small_fn": {
        "filter": lambda r: r["class"] == "fn" and _num(r.get("gt_area_px")) < 800,
        "rank_key": lambda r: _num(r.get("gt_area_px"), default=float("inf")),
        "rank_desc": False,
        "description": "False negatives on the smallest panels (gt_area < 800 px²)",
    },
    "edge_fn": {
        "filter": lambda r: r["class"] == "fn" and _num(r.get("distance_to_image_edge_px")) < 40,
        "rank_key": lambda r: _num(r.get("distance_to_image_edge_px"), default=float("inf")),
        "rank_desc": False,
        "description": "Missed panels near the image edge",
    },
    "confident_fp": {
        "filter": lambda r: r["class"] == "fp",
        "rank_key": lambda r: _num(r.get("conf"), default=0.0),
        "rank_desc": True,
        "description": "Highest-confidence false positives",
    },
    "large_fp": {
        "filter": lambda r: r["class"] == "fp" and _num(r.get("pred_area_px")) >= 4000,
        "rank_key": lambda r: _num(r.get("pred_area_px"), default=0.0),
        "rank_desc": True,
        "description": "Largest false positives (the model claims a big panel where there isn't one)",
    },
    "worst_iou_tp": {
        "filter": lambda r: r["class"] == "tp",
        "rank_key": lambda r: _num(r.get("iou"), default=1.0),
        "rank_desc": False,
        "description": "True positives with the worst IoU (loose match — likely partial detections)",
    },
}


def _num(s, default=float("nan")) -> float:
    try:
        return float(s) if s not in ("", None) else default
    except (TypeError, ValueError):
        return default


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True, help="per_detection.csv from 05c")
    p.add_argument("--bucket", type=str, default=None,
                   choices=sorted(PRESETS.keys()),
                   help="Named bucket preset")
    p.add_argument("--bucket-expr", type=str, default=None,
                   help="Free-form filter, e.g. 'class=fp AND conf>0.5'")
    p.add_argument("--rank-by", type=str, default=None,
                   help="Column to rank by when using --bucket-expr (default: conf)")
    p.add_argument("--rank-desc", action="store_true",
                   help="Sort descending (only used with --bucket-expr)")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--data", type=Path,
                   default=REPO_ROOT / "data" / "yolo" / "naip" / "data.yaml",
                   help="data.yaml — used to find image+label paths")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override output dir. Default: outputs/label_viz/<run-name>/<bucket>/")
    return p.parse_args(argv)


def load_data_root(data_yaml: Path) -> Path:
    with data_yaml.open() as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root


def find_image(data_root: Path, tile_id: str) -> tuple[Path | None, str | None]:
    for split in ("train", "val", "test"):
        candidate = data_root / "images" / split / tile_id
        if candidate.exists():
            return candidate, split
    return None, None


def parse_expression(expr: str) -> Callable[[dict], bool]:
    """Parse 'class=fp AND conf>0.5' into a row-predicate.

    Supports `=`, `!=`, `<`, `<=`, `>`, `>=` chained with `AND`. No `OR` — keep
    the grammar tiny; users can run the script twice if they need a disjunction.
    """
    parts = [p.strip() for p in re.split(r"\bAND\b", expr, flags=re.IGNORECASE)]
    clauses: list[Callable[[dict], bool]] = []
    pat = re.compile(r"^\s*(\w+)\s*(=|!=|<=|>=|<|>)\s*(.+?)\s*$")
    for part in parts:
        m = pat.match(part)
        if not m:
            raise ValueError(f"Cannot parse clause: {part!r}")
        col, op, raw = m.group(1), m.group(2), m.group(3).strip().strip("'\"")
        try:
            num_val: float | None = float(raw)
        except ValueError:
            num_val = None
        if op == "=":
            clauses.append(lambda r, c=col, v=raw: str(r.get(c, "")).strip() == v)
        elif op == "!=":
            clauses.append(lambda r, c=col, v=raw: str(r.get(c, "")).strip() != v)
        else:
            if num_val is None:
                raise ValueError(f"Operator {op} requires a numeric RHS, got {raw!r}")
            cmp = {
                "<":  lambda a, b: a < b,
                "<=": lambda a, b: a <= b,
                ">":  lambda a, b: a > b,
                ">=": lambda a, b: a >= b,
            }[op]
            clauses.append(lambda r, c=col, v=num_val, f=cmp: not _is_nan(_num(r.get(c))) and f(_num(r.get(c)), v))

    def pred(row: dict) -> bool:
        return all(c(row) for c in clauses)
    return pred


def _is_nan(x: float) -> bool:
    return isinstance(x, float) and x != x


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.bucket and not args.bucket_expr:
        print("Provide --bucket <preset> or --bucket-expr <expr>", file=sys.stderr)
        return 1
    if args.bucket and args.bucket_expr:
        print("Use --bucket OR --bucket-expr, not both", file=sys.stderr)
        return 1

    if args.bucket:
        bucket_name = args.bucket
        preset = PRESETS[args.bucket]
        filter_fn = preset["filter"]
        rank_key = preset["rank_key"]
        rank_desc = preset["rank_desc"]
        description = preset["description"]
    else:
        bucket_name = "expr_" + re.sub(r"\W+", "_", args.bucket_expr)[:40]
        filter_fn = parse_expression(args.bucket_expr)
        rank_col = args.rank_by or "conf"
        rank_key = lambda r, c=rank_col: _num(r.get(c), default=0.0)
        rank_desc = args.rank_desc
        description = f"Custom: {args.bucket_expr}"

    rows = list(csv.DictReader(args.csv.open()))
    if not rows:
        print(f"Empty CSV: {args.csv}", file=sys.stderr)
        return 1

    selected = [r for r in rows if filter_fn(r)]
    selected.sort(key=rank_key, reverse=rank_desc)
    selected = selected[: args.top]
    if not selected:
        print(f"No rows matched bucket {bucket_name!r}.", file=sys.stderr)
        return 1

    # Group selected detections by tile so each tile renders once with all its
    # in-bucket detections highlighted in the header.
    by_tile: dict[str, list[dict]] = defaultdict(list)
    for r in selected:
        by_tile[r["tile_id"]].append(r)

    run_name = args.csv.parent.name
    out_dir = args.out_dir or (REPO_ROOT / "outputs" / "label_viz" / run_name / bucket_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = load_data_root(args.data)

    rendered: list[str] = []
    skipped: list[str] = []

    # Load model lazily only if we need pred masks (we do — overlay shows pred
    # masks). For 18 we re-run inference on just the chosen tiles to keep the
    # CSV small. This is the trade: the CSV stays compact (no mask blobs) but
    # 18 needs a model handle.
    yolo_model = None

    for tile_id, tile_rows in by_tile.items():
        img_path, split = find_image(data_root, tile_id)
        if img_path is None:
            skipped.append(tile_id)
            continue
        img_w, img_h = Image.open(img_path).size
        label_path = data_root / "labels" / split / (img_path.stem + ".txt")
        label_polys = parse_label_polys(label_path, img_w, img_h)
        label_mask = polys_to_mask(label_polys, img_h, img_w)

        # Re-run the model just on this tile to recover prediction masks.
        # Light-weight: we already know which tiles to render (top-N).
        if yolo_model is None:
            try:
                from ultralytics import YOLO
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError("Run: pip install -r requirements.txt") from exc
            from src.utils.train_utils import resolve_weights, select_device
            # Recover weights path from manifest beside the CSV
            manifest_path = args.csv.parent / "manifest.json"
            weights_path: Path | None = None
            if manifest_path.exists():
                import json as _json
                m = _json.loads(manifest_path.read_text())
                wp = m.get("model_weights_path")
                if wp:
                    weights_path = Path(wp)
            if weights_path is None or not weights_path.exists():
                # Fall back to a registry-resolvable name
                weights_path = resolve_weights("production", REPO_ROOT)
            device = select_device()
            yolo_model = YOLO(str(weights_path), task="segment")
            print(f"Loaded {weights_path} on {device}")

        result = yolo_model.predict(source=str(img_path), imgsz=640, conf=0.05,
                                    device="cpu", verbose=False)[0]
        pred_masks: list[np.ndarray] = []
        pred_confs: list[float] = []
        if result.masks is not None and result.masks.data is not None and len(result.masks.data) > 0:
            mask_tensor = result.masks.data.cpu().numpy()
            confs = (result.boxes.conf.cpu().numpy().tolist()
                     if result.boxes is not None
                     else [0.0] * len(mask_tensor))
            for m, c in zip(mask_tensor, confs):
                m_bool = m > 0.5 if m.dtype != bool else m
                if m_bool.shape != (img_h, img_w):
                    pm_img = Image.fromarray((m_bool.astype(np.uint8) * 255), mode="L").resize(
                        (img_w, img_h), Image.NEAREST)
                    m_bool = np.array(pm_img) > 127
                pred_masks.append(m_bool)
                pred_confs.append(float(c))

        # Build a stats dict so the header strip says something useful for this bucket
        n_in_bucket = len(tile_rows)
        sample = tile_rows[0]
        stats = {
            "n_label_polys": len(label_polys),
            "n_pred": len(pred_masks),
            "overall_iou": _num(sample.get("iou"), default=0.0) if sample["class"] == "tp" else 0.0,
            "confident_fps": sum(1 for r in tile_rows if r["class"] == "fp"),
            "label_polys_uncovered": sum(1 for r in tile_rows if r["class"] == "fn"),
            "max_pred_conf": max((_num(r.get("conf"), default=0.0) for r in tile_rows), default=0.0),
        }
        bucket_label = f"{bucket_name} ({n_in_bucket} in-tile)"

        canvas = render_overlay(
            img_path=img_path,
            label_polys=label_polys,
            pred_masks=pred_masks,
            pred_confs=pred_confs,
            stats=stats,
            bucket=bucket_label,
            esri_inset_path=None,
        )
        canvas.save(out_dir / img_path.name)
        rendered.append(img_path.name)

    print(f"\nBucket: {bucket_name}  ({description})")
    print(f"Rendered {len(rendered)} tiles → {out_dir}")
    if skipped:
        print(f"Skipped {len(skipped)} tiles (image not found): "
              f"{', '.join(skipped[:5])}{'…' if len(skipped) > 5 else ''}")

    write_manifest(
        out_dir,
        stage="eval",
        model_version=f"bucket-{bucket_name}",
        inputs=[str(args.csv)],
        metrics={
            "n_rendered": len(rendered),
            "n_selected": len(selected),
            "n_total_rows": len(rows),
        },
        known_limitations=[
            "Tile-image lookup walks train/val/test splits — a tile present in multiple splits would resolve to whichever is first",
            "Re-runs inference on the chosen tiles (one pass each) so the renderer can show prediction masks; CSV alone has no mask data",
        ],
        extra={"bucket": bucket_name, "description": description},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
