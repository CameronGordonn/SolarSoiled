# SolarSoiled

End-to-end geospatial ML system for detecting rooftop solar arrays in public aerial imagery and scoring per-array soiling risk. **Two-stage architecture**: YOLOv11 polygon segmentation on NAIP imagery (Stage 1) → XGBoost risk model trained on weather reanalysis, air quality, land use, and structural features (Stage 2). Designed county-agnostic; runs on any NAIP-covered AOI.

**Why it matters**: Solar panel soiling (dust, pollen, aerosols) reduces energy output by 1–10% in temperate climates and up to 25–30% in arid regions. Optimizing cleaning schedules across a county-scale install base requires knowing *which* arrays are highest-risk and *when* — from public data, without site visits. This system automates that from raw GeoTIFF to actionable cleaning recommendations.

**Current metrics** — honest, methodology-explained in [Results](#results):
- Stage 1: **SAHI F1 = 0.396** (conf=0.40, iou=0.50, NAIP Santa Cruz val). Active retrain in progress; beta gate ≥ 0.55.
- Stage 2: **0.63 spatial-CV AUC / 0.66 temporal holdout AUC** (255 NREL stations, ~15 yr panel labels). GA gate ≥ 0.70.

Every output carries `model_version`, `beta`, and `known_limitations` — quality metadata is in the output contract, not a footnote.

---

## Architecture

```
NAIP GeoTIFF (0.6m GSD)
       │
       ▼
  Tiling + CRS preservation          640×640 PNG chips; affine + CRS logged in tile_index.json
       │
       ▼
  Roboflow annotation pipeline        polygon segmentation labels (whole-array convention)
       │
       ▼
  YOLOv11 segmentation training       YAML-driven experiment matrix; SGD warm-start from
  scripts/03_train_yolov8_seg.py      SAHI-calibrated checkpoint
       │
       ▼
  SAHI sliced inference               overlapping tile inference, greedy NMS with IoS
  → threshold sweep                   full SAHI loop per (conf, iou) combo — not re-thresholded mAP50
  scripts/05d_sahi_threshold_sweep.py
       │
       ▼
  Per-detection RCA                   one row per TP/FP/FN with size, density, edge, confidence;
  scripts/05c_per_detection_rca.py    failure-mode buckets → targeted Roboflow relabeling
       │
       ▼
  GeoJSON polygon export              georeferenced array footprints, CRS round-tripped end-to-end
       │
       ▼
  Feature engineering                 ERA5 weather · CAMS PM2.5/PM10 · ESA WorldCover ·
  scripts/09_build_soiling_features.py  OSM proximity · Kimber IWSR physics prior (as feature, not label)
       │
       ▼
  XGBoost soiling-risk model          10km spatial GroupKFold · isotonic calibration ·
  scripts/10_train_soiling_model.py   --holdout-year temporal validation
       │
       ▼
  solarsoiled CLI                     tile / detect / score / recommend / run / eval subcommands
  Per-AOI outputs + manifest.json     model_version + inputs_hash + beta flag on every artifact
```

---

## Key Engineering Choices

**Production metric is SAHI F1, not model.val() mAP50.** Production inference uses SAHI (Slicing Aided Hyper Inference) with greedy NMS and IoS (intersection over smaller area). This is non-monotonic in confidence threshold — you cannot re-threshold a mAP50 sweep to estimate SAHI F1. `scripts/05d_sahi_threshold_sweep.py` runs the full SAHI inference loop per (conf, iou) combination. `model.val()` mAP50 is tracked only as a fast regression signal between sweeps.

**10km spatial GroupKFold to prevent geographic leakage.** Soiling rate is spatially autocorrelated — neighboring weather stations share climate signal. Random CV leaks across spatial neighbors and inflates AUC. We cluster all 255 NREL stations into 10km bins via KMeans and hold out whole bins. The gap between spatial-CV AUC (0.63) and random-CV AUC (~0.74) quantifies the leakage that naive splitting would mask.

**Warm-start from SAHI-calibrated checkpoint, not COCO weights.** R0 retraining warm-starts from the existing SAHI baseline rather than COCO pretrained weights. The baseline learned to detect small arrays at 0.6m GSD — a prior that hand labels at source resolution can't reliably teach from scratch (small arrays are frequently under-labeled). Starting from COCO discards this prior; warm-starting preserves it while labels improve iteratively.

**Kimber IWSR physics prior as a feature, not a label source.** The Kimber 2007 Incident Weighted Soiling Rate model gives a physics-derived soiling estimate per station. Rather than using Kimber rates as training labels (which would cap model accuracy at the physics model's error floor), we include the Kimber-derived rate as one input feature. XGBoost can learn to up-weight this prior where NREL station density is sparse and discount it where empirical data is dense.

**Isotonic calibration for actionable risk scores.** Raw XGBoost predicted probabilities are miscalibrated for sparse geographic data — model confidence doesn't match empirical outcome rates. Isotonic regression (monotone, non-parametric) is fit on a held-out calibration fold post-training. Calibrated probabilities feed directly into the cleaning recommendation engine, where overconfidence would cause systematically early or late recommendations.

**Per-detection RCA harness for targeted label correction.** Instead of bulk-reviewing tiles, inference runs at low confidence (conf=0.05) and emits one row per TP/FP/FN with size, density, edge-proximity, and confidence metadata. Failure-mode buckets (alone-tile FPs, small FNs, high-confidence errors) drive targeted Roboflow relabeling batches. This approach diagnosed that 65 of 360 FPs were concentrated on 20 GT-empty tiles — likely real arrays the original 60cm labels missed, not model hallucinations — informing relabeling priority without wasted review cycles.

---

## Datasets

| Dataset | Scale | Source | Role |
|---|---|---|---|
| NAIP Santa Cruz | 249 tiles, ~360 labeled arrays, 0.6m GSD | USDA NAIP via Roboflow | Primary detection training/val/test domain |
| Duke / Bradbury | 601 source images, ~19,400 array polygons, 0.3m GSD | Duke Energy / Figshare | Small-array diversity for joint curriculum training |
| NREL soiling database | 255 stations, ~15 years panel-level soiling measurements | NREL public API | Stage 2 training labels |
| Open-Meteo ERA5 reanalysis | Historical weather per station (temp, humidity, wind, precip) | Open-Meteo OPeNDAP (1940–present) | Stage 2 weather features |
| CAMS global atmosphere | PM2.5, PM10 per station | Copernicus / MERRA-2 OPeNDAP (1980–present) | Stage 2 air quality features |
| ESA WorldCover 2021 | 10m land cover classification | ESA | Stage 2 land use features |
| OpenStreetMap | Road network, agricultural land boundaries | Overpass API | Stage 2 proximity features |

All external data fetches are disk-cached. Weather and air quality data streams via OPeNDAP — no bulk download required.

---

## Results

| Stage | Metric | Why this metric | Value |
|---|---|---|---|
| Stage 1 | **SAHI F1** | Production inference path; model.val() mAP50 consistently differs due to full-tile vs sliced inference | **0.396** at conf=0.40, iou=0.50 |
| Stage 1 | model.val() mAP50 (regression signal only) | Fast check between SAHI sweeps — not used as the production bar | 0.563 (SAHI baseline checkpoint) |
| Stage 2 | **Spatial-CV AUC** | 10km GroupKFold prevents geographic leakage; conservative generalization estimate | **0.63** |
| Stage 2 | **Temporal holdout AUC** | Held-out 2022 data; tests for temporal distribution shift | **0.66** |
| Stage 2 | (Random-CV AUC — shown for reference) | Illustrates the leakage magnitude of naive splits; not used in any production decision | ~0.74 |

**Gates**: Stage 1 beta ≥ 0.55 SAHI F1, GA ≥ 0.65. Stage 2 GA: both AUC ≥ 0.70.

---

## What's Built

**Stage 1 — Detection**

- YOLOv11 polygon segmentation training with YAML-driven experiment matrix (`scripts/03b_train_experiment_matrix.py`) — named experiment cuts from a single config, results auto-namespaced under `runs/segment/<name>/`
- SAHI threshold sweep (`scripts/05d_sahi_threshold_sweep.py`) — full SAHI inference per (conf, iou) combination; emits calibrated operating point and F1 curve
- Per-detection RCA harness (`scripts/05c_per_detection_rca.py`) — one row per TP/FP/FN with size, density, edge-proximity, confidence; `--summarize` produces `failure_modes.json`
- Failure-mode bucket overlays (`scripts/labeling/18_bucket_overlays.py`) — renders top-N PNGs per bucket (alone-tile FPs, small FNs, high-conf errors, or arbitrary `--bucket-expr`)
- Ramp eval helper (`scripts/05e_ramp_eval.py`) — per-curriculum-step eval, appends to `ramp_curve.csv`, prints HALT on NAIP regression exceeding threshold
- Domain equivalence baseline (`scripts/01b_compare_naip_duke_distributions.py`) — KS tests + distribution plots for NAIP vs Duke label conventions; used to validate dataset merging decisions

**Stage 2 — Risk Model**

- Feature engineering from five external sources with per-station alignment and disk caching (`scripts/09_build_soiling_features.py`, `scripts/13_build_static_features.py`)
- XGBoost training with spatial GroupKFold, isotonic calibration, `--holdout-year` temporal validation (`scripts/10_train_soiling_model.py`)
- Run comparison table (`scripts/14_compare_soiling_runs.py`) — side-by-side metrics across named training runs
- NREL validation (`scripts/15_validate_against_nrel.py`) — independent check against held-out station measurements

**Product / CLI**

- `solarsoiled` CLI — `tile / detect / score / recommend / run / eval` subcommands; `run --aoi <bbox-or-geojson>` chains all stages; per-AOI output namespace under `outputs/aoi/<partner_id>/`
- Model registry (`models/registry.yaml`) — resolves named aliases (`production`, `latest`, …) and ad-hoc `.pt` paths; `model_version` tagged on every output
- Manifest contract (`src/solarsoiled/manifest.py`) — every artifact-producing stage writes a sibling `manifest.json` with inputs hash, model SHA256, beta flag, known limitations
- HTML eval report (`solarsoiled eval --report`) — single-file report with PR curve, F1-colored sweep table, failure-mode tables, base64-embedded overlay PNGs; no inference re-run required
- Docker image — CPU and CUDA 12.1 GPU variants; `.cache/` mount for weather data persistence

---

## What's In Progress

- **Train-set relabeling + R0 retrain** — iterative Roboflow relabeling with per-detection RCA driving batches; val/test already relabeled; targeting SAHI F1 ≥ 0.55 beta gate
- **Duke curriculum ramp (R1+)** — joint NAIP + Duke training after R0 lands; curriculum in `configs/yolo/experiments_joint_v2_ramp.yaml` with hard regression stop-rule
- **Stage 2 AUC improvement** — feature ablation and label-source comparison toward the 0.70 spatial-CV gate
- **Beta API surface** — `/detect`, `/risk`, `/recommend` endpoints with quality metadata, auth, and per-scan metering

---

## Quick Start

### Docker (recommended)

```bash
# CPU build
docker build -t solarsoiled .

# GPU build (CUDA 12.1)
docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 -t solarsoiled:gpu .

# End-to-end on an AOI
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/.cache:/app/.cache \
  solarsoiled run \
    --aoi "-122.05,36.90,-121.85,37.05" \
    --weights production \
    --soiling-model soiling_production \
    --last-cleaned 2025-12-01 \
    --partner-id smoketest
```

Place `.pt` weight files in `models/` before running (not baked into the image). Outputs land in `outputs/aoi/<partner_id>/`. The `.cache/` mount (~440 MB after warm-up) persists weather data across runs.

### Local (dev)

```bash
cd setup/ && bash setup_conda.sh
conda activate solar-soiling
pip install -e .   # registers the solarsoiled CLI

# Full pipeline on one AOI
solarsoiled run \
  --aoi "minx,miny,maxx,maxy" \
  --weights models/sahi_baseline_train7.pt \
  --soiling-model runs/soiling/run_latest/model.ubj \
  --last-cleaned 2026-01-01 \
  --partner-id smoketest

# Re-run only score + recommend on cached upstream artifacts
solarsoiled run --aoi <…> --weights <…> --soiling-model <…> \
  --last-cleaned 2026-01-01 --partner-id smoketest \
  --skip-tile --skip-detect

# Research scripts are still supported standalone
python scripts/01_audit_dataset.py --config configs/yolo/dataset_audit.yaml
python scripts/03_train_yolov8_seg.py --model models/yolo11s-seg.pt --epochs 50
python scripts/05d_sahi_threshold_sweep.py --weights runs/segment/<run>/weights/best.pt
PYTHONPATH=. python scripts/10_train_soiling_model.py --run-name run_latest
```

Base YOLO weights (`yolo11s-seg.pt`, `yolo11m-seg.pt`) are gitignored — download to `models/` manually.

---

## Documentation

| Doc | Purpose |
|---|---|
| [docs/SOILING_STAGE2_GUIDE.md](docs/SOILING_STAGE2_GUIDE.md) | Stage 2 risk model — features, training, validation |
| [docs/NAIP_ROBOFLOW_WORKFLOW.md](docs/NAIP_ROBOFLOW_WORKFLOW.md) | Full tile → label → train → export reference |
| [docs/PRODUCT_VISION.md](docs/PRODUCT_VISION.md) | Strategy, beta/GA contract, customer-readiness arc |
| [docs/Q2_PLAN.md](docs/Q2_PLAN.md) | Current roadmap and workstream status |
| [docs/HYPERPARAM_PLAYBOOK.md](docs/HYPERPARAM_PLAYBOOK.md) | Training hyperparameter rationale |
| [docs/RTX3060_SETUP_GUIDE.md](docs/RTX3060_SETUP_GUIDE.md) | GPU environment setup |

---

## Stack

Python 3.11 · PyTorch 2.0+ · [YOLOv11 / ultralytics ≥ 11.0](https://docs.ultralytics.com/) · XGBoost · GDAL / rasterio · SAHI · Roboflow · Open-Meteo ERA5 · CAMS / MERRA-2 · ESA WorldCover · OpenStreetMap Overpass · Docker · scikit-learn (isotonic calibration, GroupKFold)
