# CLAUDE.md — solar-soiling-ml

YOLOv11 polygon segmentation pipeline for detecting solar arrays in NAIP aerial imagery, plus an XGBoost soiling-risk model that scores each detected array.
**Stage 1 target:** SAHI F1 ≥ 0.55 (beta gate), ≥ 0.65 (GA gate). `model.val()` mAP50 is NOT the production metric for SAHI-based models — use `05d_sahi_threshold_sweep.py`. **Current production:** R2-cameron-20260509 (SAHI F1=0.396 on current val labels, conf=0.40). Train relabeling in progress — val/test were relabeled (multiple rounds) but train has not been yet.
**Stage 2:** per-array soiling risk (XGBoost on weather + location + structural features). The active default uses the NREL panel-label CSV (`nrel_panel`) with spatial CV at 10 km cluster bins; Kimber physics-proxy labels remain available as a bootstrap fallback. **Honest baseline:** 0.63 spatial-CV AUC / 0.66 holdout-2022 AUC (leakage-free, sparse-feature-gated, run_k_patched_holdout2022). Below the 0.70 acceptance gate — ships as beta. See @docs/SOILING_STAGE2_GUIDE.md.

---

## Core Principles

1. **Data > Model** — label quality beats architecture complexity; audit before training
2. **Config-driven** — hyperparameters live in YAML, not scripts; commit config with results
3. **Geospatial first** — `tile_index.json` is sacred; CRS + affine must survive every round-trip
4. **Traceability** — every run produces unique weights in `runs/segment/<name>/`; use `best.pt` not `last.pt`
5. **Hardware flexibility** — presets: `laptop` (batch=1, imgsz=512), `small` (batch=8, imgsz=640), `medium` (batch=4, imgsz=768)
6. **API stability** — ultralytics>=11.0; metrics via `results[0].mp` / `results[0].mr` (v11 API)

---

## Key Commands

### Partner-facing CLI (Tier 1 — `solarsoiled`)

```bash
# End-to-end on one AOI (tile → detect → score → recommend)
solarsoiled run \
  --aoi "minx,miny,maxx,maxy"                  \  # or a GeoJSON polygon path
  --weights production                         \  # registered name from models/registry.yaml
  --soiling-model runs/soiling/run_latest/model.ubj \
  --last-cleaned 2026-01-01                    \
  --partner-id smoketest

# Re-run only score + recommend on cached upstream artifacts
solarsoiled run --aoi <…> --weights <…> --soiling-model <…> \
  --last-cleaned 2026-01-01 --partner-id smoketest \
  --skip-tile --skip-detect

# Each stage as a subcommand (tile / detect / score / recommend / eval)
solarsoiled detect --aoi <…> --weights <…> --partner-id smoketest

# Build a self-contained HTML quality report from an existing eval run
# (consumes per_detection.csv + sahi_threshold_sweep.csv + failure_modes.json + overlay PNGs)
solarsoiled eval --weights <name-or-path> --report \
  --report-dir outputs/eval/<run-name> \
  --report-out outputs/eval/<run-name>/report.html
```

`--weights` accepts a registered model name (`production`, `latest`, `stage1-v0.5-baseline`, `yolo11s-base`, …) from `models/registry.yaml`, an alias from the same file, or a filesystem path to a `.pt` (passthrough mode → `model_version="ad-hoc:<sha12>"`). Add a new training cut by appending an entry under `models:` and bumping `aliases.latest`. Outputs land under `outputs/aoi/<partner_id>/{aoi.geojson, tiles/, detect/, arrays.geojson, features/, risk.geojson, recommendations.json, manifest.json}`. Every artifact dir carries a `manifest.json` with model_version + inputs_hash + beta flag.

The 14 numbered scripts in `scripts/` remain the canonical research surface — the CLI imports their `main(argv=…)` functions as library calls and does not replace them.

### Research scripts (the existing flow)

```bash
# Audit labels
python scripts/01_audit_dataset.py --config configs/yolo/dataset_audit.yaml

# Train (single run)
python scripts/03_train_yolov8_seg.py --model models/yolo11s-seg.pt --epochs 50

# Train (experiment matrix)
python scripts/03b_train_experiment_matrix.py --config configs/yolo/experiments.yaml

# Threshold sweep — model.val() path
python scripts/05b_eval_threshold_sweep.py --weights runs/segment/<run_name>/weights/best.pt

# Threshold sweep — SAHI inference path (required for production calibration; slower)
python scripts/05d_sahi_threshold_sweep.py --weights runs/segment/<run_name>/weights/best.pt \
    --config configs/yolo/thresholds_sahi.yaml --run-name <run_name>

# Per-detection RCA — TP/FN/FP rows with confidence + size + edge + density metadata
python scripts/05c_per_detection_rca.py --weights <weights.pt> \
    --data data/yolo/naip/data.yaml --splits val test --sahi --conf 0.05 --run-name <run_name>
python scripts/05c_per_detection_rca.py --summarize \
    --csv outputs/eval/<run_name>/per_detection.csv

# Bucketed overlay rendering for label-gap audit (consumes per_detection.csv)
python scripts/labeling/18_bucket_overlays.py \
    --csv outputs/eval/<run_name>/per_detection.csv --bucket confident_fp --top 20

# NAIP/Duke domain equivalence baseline (no inference required)
python scripts/01b_compare_naip_duke_distributions.py

# Per-step Duke ramp eval (drives 05 + 05c, appends ramp_curve.csv, prints HALT on regression)
python scripts/05e_ramp_eval.py --run R0 --weights runs/segment/<run>/weights/best.pt

# Infer + export GeoJSON
python scripts/04_infer_yolov8_seg.py
python scripts/06_export_polygons_geojson.py

# Stage 2 soiling risk (production pipeline)
PYTHONPATH=. python scripts/12_ingest_nrel_soiling_map.py                             # NREL JSON → summary + annual-panel CSVs (one-time)
PYTHONPATH=. python scripts/13_build_static_features.py                               # elevation + WorldCover + OSM distances (one-time, ~15 min)
PYTHONPATH=. python scripts/10_train_soiling_model.py --run-name run_latest           # NREL panel labels + Kimber feature + static features + CI-weighted
PYTHONPATH=. python scripts/10_train_soiling_model.py --run-name run_holdout2022 --holdout-year 2022   # temporal validation
PYTHONPATH=. python scripts/14_compare_soiling_runs.py                                # metrics comparison table
PYTHONPATH=. python scripts/09_build_soiling_features.py --arrays outputs/array_features.geo.parquet
PYTHONPATH=. python scripts/11_predict_soiling_risk.py --model runs/soiling/<run_name>/model.ubj  # auto-applies calibrator.joblib
```

---

## Key Paths

| What | Where |
|------|-------|
| NAIP images | `data/yolo/naip/images/{train,val,test}/` |
| YOLO labels | `data/yolo/naip/labels/{train,val,test}/` |
| Geospatial metadata | `data/interim/tile_index.json` |
| Best weights | `runs/segment/<run_name>/weights/best.pt` |
| Experiment configs | `configs/yolo/experiments.yaml` (prod), `experiments_laptop.yaml` (CPU) |
| Soiling module | `src/soiling/`, configs in `configs/soiling/`, scripts 09–14 |
| NREL labels | `data/external/nrel_soiling_map.csv` (summary), `data/external/nrel_soiling_map_annual.csv` (panel) — both gitignored |
| Static features | `data/external/static_features.csv` (per-station elevation + WorldCover + OSM distances) — gitignored |
| Soiling outputs | `outputs/soiling/training_matrix.parquet`, `runs/soiling/<run>/{model.ubj,feature_names.json,feature_medians.json,metrics.json,calibrator.joblib}`, `outputs/soiling_risk.geojson` |
| GeoJSON output | `outputs/solar_arrays.geojson` |
| Eval results | `outputs/eval/experiment_results.csv` |

---

## Dataset Status (as of 2026-05-13)

- Train: 174 images, ~247 arrays (NOT yet relabeled — train relabeling in progress)
- Val: 37 images, ~212 objects (relabeled multiple rounds) | Test: 38 images, ~52 objects (relabeled)
- Format: YOLOv11 polygon segmentation (normalized 0-1 coords, class=0)
- Recommended thresholds (SAHI production path): conf=0.40, iou=0.50

---

## Stage 1 — Diagnose-First + Fresh-Init R0 (Active, week of 2026-05-04)

After Tyler's meeting (2026-05-04), the active workstream pivoted from "joint training to lift past 56%" to **diagnose-first** then iterative R0 retrains on relabeled NAIP, warm-starting each iteration from `models/sahi_baseline_train7.pt` (preserves the small-panel detection prior our 60 cm hand labels can't teach). Reasons: (1) joint v1 + v2 over-prediction failures (latest joint run: mAP50 0.128 / R 0.76 / P <0.1 — model hallucinated panels everywhere); (2) NAIP labels were drawn at 60cm source resolution and miss small panels — the "65 FPs on 20 alone-tiles at conf=0.05" pattern is a label gap, not a model failure; (3) Duke uses per-panel labels vs NAIP whole-array (KS=0.895 on area_m²), so joint training tries to bridge incompatible label conventions.

Active runbook: [@docs/PHASE1_HANDOFF.md](docs/PHASE1_HANDOFF.md). Active config: `configs/yolo/experiments_joint_v2_ramp.yaml` (R0–R5; R0 = NAIP-only warm-start from SAHI baseline, R1+ = ramp Duke after R0 lands). RCA harness: `scripts/05c_per_detection_rca.py` + `scripts/labeling/18_bucket_overlays.py`. Project-level status: @docs/Q2_PLAN.md.

---

## Quarter direction

@docs/Q2_PLAN.md @docs/PRODUCT_VISION.md

---

## Notes

- Model naming: Ultralytics uses `yolo11*` (no `v`). Configs point to `models/yolo11s-seg.pt` and `models/yolo11m-seg.pt`.
- Base weights (`.pt` files) are gitignored — download manually to `models/` on each machine.
- Reference docs (not auto-pulled): `docs/NAIP_ROBOFLOW_WORKFLOW.md` (pipeline reference), `docs/RTX3060_SETUP_GUIDE.md` (GPU env). Read on demand.
- Ultralytics YOLOv11 docs: https://docs.ultralytics.com/
