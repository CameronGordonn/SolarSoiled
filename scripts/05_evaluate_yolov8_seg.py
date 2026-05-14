#!/usr/bin/env python3
"""Evaluate a trained YOLOv11 segmentation model and optionally save metrics as JSON."""

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO

from src.utils.train_utils import (
    compute_f1,
    resolve_data_yaml,
    resolve_project_dir,
    resolve_weights,
    select_device,
)
from solarsoiled.manifest import write_manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate YOLO segmentation model")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--project", type=str, default="runs/segment")
    parser.add_argument("--name", type=str, default="val")
    parser.add_argument("--metrics-json", type=str, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]

    weights_path = resolve_weights(args.weights, repo_root)
    data_yaml = resolve_data_yaml(args.data, repo_root)
    project_dir = resolve_project_dir(args.project, repo_root)
    device = select_device()

    model = YOLO(str(weights_path), task="segment")
    metrics = model.val(
        data=str(data_yaml), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        split=args.split, device=device, project=str(project_dir), name=args.name, verbose=True,
    )

    sp, sr = float(metrics.seg.mp), float(metrics.seg.mr)
    print(f"\nSegmentation — mAP50: {metrics.seg.map50:.4f}  mAP50-95: {metrics.seg.map:.4f}"
          f"  P: {sp:.4f}  R: {sr:.4f}  F1: {compute_f1(sp, sr):.4f}")
    print(f"Box          — mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")

    if args.metrics_json:
        out_path = Path(args.metrics_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "weights": str(weights_path), "data": str(data_yaml),
            "split": args.split, "imgsz": args.imgsz, "conf": args.conf, "iou": args.iou,
            "seg": {"map50": float(metrics.seg.map50), "map50_95": float(metrics.seg.map),
                    "precision": sp, "recall": sr, "f1": compute_f1(sp, sr)},
            "box": {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
                    "precision": float(metrics.box.mp), "recall": float(metrics.box.mr)},
        }, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}")

        seg_map50 = float(metrics.seg.map50)
        # If weights live under runs/segment/<run>/weights/, use <run>; else fall back to filename stem.
        run_tag = (
            weights_path.parent.parent.name
            if weights_path.parent.name == "weights"
            else weights_path.stem
        )
        write_manifest(
            out_path.parent,
            stage="eval",
            model_version=f"stage1-{run_tag}",
            model_weights=weights_path,
            inputs=[str(data_yaml)],
            beta=seg_map50 < 0.70,
            metrics={
                "box_map50": float(metrics.box.map50),
                "box_map50_95": float(metrics.box.map),
                "box_precision": float(metrics.box.mp),
                "box_recall": float(metrics.box.mr),
                "mask_map50": seg_map50,
                "mask_map50_95": float(metrics.seg.map),
                "mask_precision": sp,
                "mask_recall": sr,
                "mask_f1": compute_f1(sp, sr),
            },
            known_limitations=["NAIP Santa Cruz training distribution; 0.6m GSD"],
            extra={"split": args.split, "metrics_json": str(out_path)},
        )


if __name__ == "__main__":
    main()
