#!/usr/bin/env bash
# Phase 1 diagnostic battery — Cameron runs this on his machine (has SAHI weights + torch).
# Writes everything to outputs/eval/sahi_baseline_train7/ and outputs/label_viz/.
#
# *** Reframed after 2026-05-04 finding ***: NAIP labels were drawn at 60cm
# source resolution, so small panels are systematically under-labeled. The
# "FP on alone tile" pattern is therefore most likely a *label gap*, not a
# model hallucination. These overlays double as a label-quality audit — every
# rendered confident_fp tile is a candidate for re-labeling.
#
# Order:
#   1. Bucket overlays — fast (each script re-runs inference on top-N tiles only,
#      not the whole split). ~1-2 min per bucket.
#   2. Threshold sweep — slow (~1-2 hours). Run after the overlays.
#
# After this finishes, Cameron triages the confident_fp overlays into:
#   (a) real label gaps → re-label before Josh trains R0
#   (b) genuine hallucinations (rare?) → note for post-retrain analysis
# Hard-negative augmentation is OFF the table — we'd be teaching the model to
# ignore real panels.

set -euo pipefail
cd "$(dirname "$0")/.."

WEIGHTS="models/sahi_baseline_train7.pt"
CSV="outputs/eval/sahi_baseline_train7/per_detection.csv"
RUN="sahi_baseline_train7"

[ -f "$WEIGHTS" ] || { echo "Missing $WEIGHTS"; exit 1; }
[ -f "$CSV" ]      || { echo "Missing $CSV — run 05c first"; exit 1; }

echo "=== 1/6 confident_fp overlays (high-conf hallucinations) ==="
python3 scripts/labeling/18_bucket_overlays.py --csv "$CSV" --bucket confident_fp --top 20

echo "=== 2/6 alone-tile FPs (the empty-tile hallucinations specifically) ==="
python3 scripts/labeling/18_bucket_overlays.py --csv "$CSV" \
    --bucket-expr "class=fp AND num_other_panels_in_tile=0" --top 20 \
    --rank-by conf --rank-desc

echo "=== 3/6 large_fp (the model claims a big panel where there isn't one) ==="
python3 scripts/labeling/18_bucket_overlays.py --csv "$CSV" --bucket large_fp --top 20

echo "=== 4/6 edge_fn (panels missed near image edges) ==="
python3 scripts/labeling/18_bucket_overlays.py --csv "$CSV" --bucket edge_fn --top 20

echo "=== 5/6 worst_iou_tp (TPs with the loosest match — partial detections) ==="
python3 scripts/labeling/18_bucket_overlays.py --csv "$CSV" --bucket worst_iou_tp --top 20

echo "=== 6/6 SAHI threshold sweep (slow — re-runs SAHI per (conf, iou) combo) ==="
echo "    18 combos × ~1 SAHI pass each. Background this if you want."
python3 scripts/05d_sahi_threshold_sweep.py \
    --weights "$WEIGHTS" \
    --config configs/yolo/thresholds_sahi.yaml \
    --run-name "$RUN"

echo
echo "Done. Inspect:"
echo "  outputs/label_viz/$RUN/expr_class_fp_AND_num_other_panels_in_tile_0/  ← MOST IMPORTANT"
echo "  outputs/label_viz/$RUN/confident_fp/"
echo "  outputs/label_viz/$RUN/large_fp/"
echo "  outputs/eval/$RUN/sahi_threshold_sweep.csv"
echo "  outputs/eval/$RUN/sahi_threshold_sweep_best.json"
