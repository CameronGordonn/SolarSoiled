#!/usr/bin/env python3
"""Per-detection root-cause analysis (RCA) harness.

For every detection-or-miss on the chosen split, emit one CSV row classifying
it as TP, FP, or FN with size, position, density, and confidence metadata.
Owns its own SAHI inference loop because production inference uses SAHI sliced
prediction — discarding per-polygon confidence would make confidence-ranked
failure-mode queries impossible.

Outputs:
  outputs/eval/<run-name>/per_detection.csv
  outputs/eval/<run-name>/failure_modes.json   (when --summarize is set)
  outputs/eval/<run-name>/manifest.json

Cross-check: aggregating per_detection.csv by (tile_id, class) reproduces
`compute_sahi_confusion_matrix.py`'s per-image TP/FP/FN counts. If it doesn't,
the matcher has drifted and the rest of the eval pipeline is unreliable.

Usage:
  # Full SAHI RCA on NAIP val+test
  python scripts/05c_per_detection_rca.py \\
      --weights runs/segment/<run>/weights/best.pt \\
      --data data/yolo/naip/data.yaml --splits val test \\
      --sahi --conf 0.05 --run-name <run-name>

  # Auto-render the three standard bucket overlays after the RCA pass
  python scripts/05c_per_detection_rca.py \\
      --weights <path> --sahi --conf 0.05 --run-name <name> --render-buckets

  # Aggregate failure-mode patterns from an existing CSV
  python scripts/05c_per_detection_rca.py --summarize \\
      --csv outputs/eval/<run-name>/per_detection.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.solarsoiled.manifest import write_manifest
from src.utils.rca import render_standard_buckets, run_rca_pass, summarize_rca
from src.utils.train_utils import resolve_weights


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
    p.add_argument("--render-buckets", action="store_true",
                   help="After writing per_detection.csv, auto-render "
                        "confident_fp / worst_small_fn / large_fp overlays")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.summarize:
        if args.csv is None:
            print("--summarize requires --csv", file=sys.stderr)
            return 1
        out_json = args.csv.parent / "failure_modes.json"
        summary = summarize_rca(args.csv, out_json)
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

    csv_path = run_rca_pass(
        weights_path=weights_path,
        data_yaml=args.data,
        splits=args.splits,
        out_dir=out_dir,
        sahi=args.sahi,
        conf=args.conf,
        iou=args.iou,
        slice_size=args.slice,
        overlap=args.overlap,
        imgsz=args.imgsz,
        limit=args.limit,
        standard_pred=args.standard_pred,
    )

    counts = {"tp": 0, "fp": 0, "fn": 0}
    seen_tiles: set[str] = set()
    for r in csv.DictReader(csv_path.open()):
        if r["class"] in counts:
            counts[r["class"]] += 1
        seen_tiles.add(r["tile_id"])

    write_manifest(
        out_dir,
        stage="eval",
        model_version=f"rca-{weights_run}",
        model_weights=weights_path,
        inputs=[str(args.data)],
        metrics={
            "n_tiles": len(seen_tiles),
            "n_rows": sum(counts.values()),
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

    if args.render_buckets:
        print("Rendering standard bucket overlays...")
        render_standard_buckets(csv_path, args.data)

    return 0


if __name__ == "__main__":
    sys.exit(main())
