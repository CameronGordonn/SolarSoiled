#!/usr/bin/env python3
"""Sweep confidence/IoU thresholds and rank segmentation metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from ultralytics import YOLO

from src.utils.train_utils import compute_f1, resolve_project_dir, select_device


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Threshold sweep for YOLO segmentation")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/yolo/naip/data.yaml")
    parser.add_argument("--config", type=str, default="configs/yolo/thresholds.yaml")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Write outputs under outputs/eval/<run-name>/ instead of flat")
    parser.add_argument("--out-csv", type=str, default=None,
                        help="Override output CSV path (default: outputs/eval[/<run-name>]/threshold_sweep_results.csv)")
    parser.add_argument("--summary-json", type=str, default=None,
                        help="Override best-result JSON path")
    parser.add_argument("--project", type=str, default="runs/segment")
    parser.add_argument("--name", type=str, default="threshold_sweep")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]

    weights = Path(args.weights).expanduser().resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    data_yaml = Path(args.data).expanduser().resolve()
    if not data_yaml.exists():
        data_yaml = (repo_root / args.data).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {args.data}")

    config_yaml = Path(args.config).expanduser().resolve()
    if not config_yaml.exists():
        config_yaml = (repo_root / args.config).resolve()
    if not config_yaml.exists():
        raise FileNotFoundError(f"Threshold config not found: {args.config}")

    eval_dir = (repo_root / "outputs" / "eval" / args.run_name
                if args.run_name
                else repo_root / "outputs" / "eval")
    out_csv = (Path(args.out_csv).expanduser().resolve() if args.out_csv
               else eval_dir / "threshold_sweep_results.csv")
    summary_json = (Path(args.summary_json).expanduser().resolve() if args.summary_json
                    else eval_dir / "threshold_sweep_best.json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    project_dir = resolve_project_dir(args.project, repo_root)

    with config_yaml.open(encoding="utf-8") as f:
        cfg = (yaml.safe_load(f) or {}).get("threshold_sweep", {})
    conf_values = [float(v) for v in cfg.get("confidence", [0.25])]
    iou_values = [float(v) for v in cfg.get("iou", [0.7])]
    split = str(cfg.get("split", "val"))
    imgsz = int(cfg.get("imgsz", 640))
    target_metric = str(cfg.get("target_metric", "f1"))

    device = select_device()
    model = YOLO(str(weights), task="segment")
    rows: List[Dict[str, float]] = []

    for conf in conf_values:
        for iou in iou_values:
            run_name = f"{args.name}_c{conf:.2f}_i{iou:.2f}".replace(".", "p")
            print(f"  conf={conf:.2f} iou={iou:.2f}")
            metrics = model.val(
                data=str(data_yaml), imgsz=imgsz, conf=conf, iou=iou,
                split=split, device=device, project=str(project_dir), name=run_name, verbose=False,
            )
            p, r = float(metrics.seg.mp), float(metrics.seg.mr)
            rows.append({
                "conf": conf, "iou": iou,
                "seg_map50": float(metrics.seg.map50),
                "seg_map50_95": float(metrics.seg.map),
                "seg_precision": p, "seg_recall": r, "seg_f1": compute_f1(p, r),
            })

    if not rows:
        raise RuntimeError("No threshold sweep rows produced")

    metric_key = {"f1": "seg_f1", "map50": "seg_map50", "map50_95": "seg_map50_95",
                  "precision": "seg_precision", "recall": "seg_recall"}.get(target_metric, "seg_f1")
    best = max(rows, key=lambda x: x[metric_key])

    fieldnames = ["conf", "iou", "seg_map50", "seg_map50_95", "seg_precision", "seg_recall", "seg_f1"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_json.write_text(json.dumps({
        "weights": str(weights), "data": str(data_yaml), "split": split,
        "imgsz": imgsz, "target_metric": target_metric, "best": best,
    }, indent=2), encoding="utf-8")

    print(f"Saved: {out_csv}")
    print(f"Best by {target_metric}: conf={best['conf']:.2f} iou={best['iou']:.2f} f1={best['seg_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
