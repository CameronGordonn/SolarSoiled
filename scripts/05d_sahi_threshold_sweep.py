#!/usr/bin/env python3
"""SAHI-aware threshold sweep — calibrates conf/iou for the SAHI inference path.

`05b_eval_threshold_sweep.py` calibrates ultralytics `model.val()`, but our
production inference uses SAHI sliced prediction with `GREEDYNMM` + `IOS`
postprocessing — and those change which boxes survive at each confidence
threshold in ways that pandas-level re-thresholding cannot reproduce. So this
script invokes the full 05c SAHI pipeline once per (conf, iou) and tabulates
the resulting precision / recall / F1 / AP-at-IoU.

Slow but defensible. Cost is roughly `n_threshold_combos × time(05c)`. With
the default 6×3 grid that's 18× a single 05c pass — keep `--limit` small if
you're iterating.

Outputs:
  outputs/eval/<run-name>/sahi_threshold_sweep.csv
  outputs/eval/<run-name>/sahi_threshold_sweep_best.json
  outputs/eval/<run-name>/manifest.json (overwrites the 05c manifest's metrics)

Usage:
  python scripts/05d_sahi_threshold_sweep.py \\
      --weights models/sahi_baseline_train7.pt \\
      --config configs/yolo/thresholds_sahi.yaml \\
      --run-name sahi_baseline_train7
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.solarsoiled.manifest import write_manifest  # noqa: E402
from src.utils.rca import run_rca_pass  # noqa: E402
from src.utils.train_utils import resolve_weights  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "yolo" / "naip" / "data.yaml")
    p.add_argument("--config", type=Path,
                   default=REPO_ROOT / "configs" / "yolo" / "thresholds_sahi.yaml")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N tiles per split (smoke testing)")
    return p.parse_args(argv)


def metrics_from_csv(csv_path: Path) -> dict:
    counts = {"tp": 0, "fp": 0, "fn": 0}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            cls = row["class"]
            if cls in counts:
                counts[cls] += 1
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def main(argv=None) -> int:
    args = parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())["threshold_sweep_sahi"]
    confs = list(cfg.get("confidence", [0.10, 0.20, 0.30]))
    ious = list(cfg.get("iou", [0.50]))
    splits = list(cfg.get("splits", ["val"]))
    slice_size = int(cfg.get("slice_size", 640))
    slice_overlap = float(cfg.get("slice_overlap", 0.2))
    target_metric = cfg.get("target_metric", "f1")
    standard_pred = bool(cfg.get("perform_standard_pred", False))

    weights_path = resolve_weights(args.weights, REPO_ROOT)
    weights_run = (weights_path.parent.parent.name
                   if weights_path.parent.name == "weights"
                   else weights_path.stem)
    run_name = args.run_name or weights_run
    out_dir = REPO_ROOT / "outputs" / "eval" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_csv = out_dir / "sahi_threshold_sweep.csv"
    best_json = out_dir / "sahi_threshold_sweep_best.json"

    sweep_rows: list[dict] = []
    print(f"SAHI threshold sweep: {len(confs)}×{len(ious)} = {len(confs)*len(ious)} combos "
          f"on splits={splits} (this is slow — each combo runs SAHI on every tile)")

    for conf in confs:
        for iou in ious:
            with tempfile.TemporaryDirectory() as td:
                tmp_dir = Path(td)
                print(f"  [conf={conf:.2f} iou={iou:.2f}]", flush=True)
                csv_path = run_rca_pass(
                    weights_path=weights_path,
                    data_yaml=args.data,
                    splits=splits,
                    out_dir=tmp_dir,
                    sahi=True,
                    conf=conf,
                    iou=iou,
                    slice_size=slice_size,
                    overlap=slice_overlap,
                    limit=args.limit,
                    standard_pred=standard_pred,
                )
                metrics = metrics_from_csv(csv_path)
            sweep_rows.append({"conf": conf, "iou": iou, **metrics})
            print(f"    → P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                  f"F1={metrics['f1']:.3f}")

    # Write sweep CSV
    with sweep_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)
    print(f"\nWrote {sweep_csv}")

    # Best-by-target
    best = max(sweep_rows, key=lambda r: r.get(target_metric, 0.0))
    best_json.write_text(json.dumps({
        "target_metric": target_metric,
        "weights": str(weights_path),
        "config": str(args.config),
        **best,
    }, indent=2) + "\n")
    print(f"Best ({target_metric}): conf={best['conf']:.2f} iou={best['iou']:.2f} "
          f"P={best['precision']:.3f} R={best['recall']:.3f} F1={best['f1']:.3f}")

    write_manifest(
        out_dir,
        stage="eval",
        model_version=f"threshold-sweep-sahi-{weights_run}",
        model_weights=weights_path,
        inputs=[str(args.data), str(args.config)],
        metrics={
            "n_combos": len(sweep_rows),
            f"best_{target_metric}": float(best.get(target_metric, 0.0)),
            "best_conf": float(best["conf"]),
            "best_iou": float(best["iou"]),
            "best_precision": float(best["precision"]),
            "best_recall": float(best["recall"]),
        },
        known_limitations=[
            "Each (conf, iou) re-runs SAHI from scratch — slow but required because GREEDYNMM+IOS postprocess is non-monotonic in conf",
            "Single-class only (AP-at-IoU is degenerate for nc=1; the metric reported here is sweep precision/recall/F1)",
        ],
        extra={
            "slice_size": slice_size,
            "slice_overlap": slice_overlap,
            "splits": splits,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
