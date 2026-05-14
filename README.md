# solar-soiling-ml

Geospatial ML for detecting rooftop solar arrays in NAIP aerial imagery and scoring per-array soiling risk. Two-stage pipeline: YOLOv11 polygon segmentation (Stage 1) → XGBoost soiling-risk model on weather + location + structural features (Stage 2). Designed county-agnostic.

**Current status (beta):**
- **Stage 1 detection** — 56% mAP50 on NAIP Santa Cruz (production-safe). Active goal 70%+ via Stage 1 joint Duke + NAIP training.
- **Stage 2 soiling risk** — 0.63 spatial-CV AUC / 0.66 holdout-2022 AUC on NREL panel labels. Below the 0.70 GA bar; ships as beta with explicit metric disclosure.

Both surfaces follow an honesty-by-default contract: every output carries `model_version`, `beta`, and `known_limitations`. See [docs/PRODUCT_VISION.md](docs/PRODUCT_VISION.md).

## Features

- **YOLOv11 polygon segmentation** on NAIP RGB (CLAHE tone mapping, geospatial metadata preserved end-to-end)
- **Roboflow-assisted labeling** with traceable round-trip via `roboflow_metadata.json` + `tile_index.json`
- **Environmental features** — Open-Meteo ERA5 weather + CAMS air quality, ESA WorldCover, OSM distance-to-highway / agriculture, Kimber 2007 IWSR physics-prior. All network calls disk-cached.
- **XGBoost soiling model** — NREL panel labels (255 stations × ~15 yrs), 10 km spatial GroupKFold, isotonic calibration, `--holdout-year` temporal validation

## Pipeline at a glance

```
NAIP GeoTIFF → tile → Roboflow → YOLO labels → audit → train → calibrate → infer → GeoJSON
                                                                          ↓
                                                         per-array features → XGBoost → soiling risk
```

Numbered scripts walk it: `02_*` tile/label, `03_*` train, `05_*` eval, `06_*` export, `09–14_*` Stage 2.

## Quick start

### Docker (partner-recommended)

```bash
# Build (CPU)
docker build -t solarsoiled .

# Build (GPU — CUDA 12.1)
docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 -t solarsoiled:gpu .

# Run end-to-end on an AOI
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

Place `.pt` weight files in `models/` before running — they are not baked into the image.
Outputs land in `outputs/aoi/<partner_id>/`. The `.cache/` mount (~440 MB after warm-up) avoids re-fetching weather data on repeat runs.

### Local (dev)

```bash
# Setup
cd setup/ && bash setup_conda.sh
conda activate solar-soiling
pip install -e .   # registers the `solarsoiled` CLI entry point

# End-to-end on one AOI (Tier 1 CLI — wraps the numbered scripts below)
solarsoiled run \
  --aoi "minx,miny,maxx,maxy" \
  --weights models/sahi_baseline_train7.pt \
  --soiling-model runs/soiling/run_latest/model.ubj \
  --last-cleaned 2026-01-01 \
  --partner-id smoketest

# Or each stage directly: solarsoiled tile / detect / score / recommend / eval

# Research-mode scripts (still supported and unchanged):
# Stage 1
python scripts/02_tile_naip_image.py
python scripts/02b_export_to_roboflow.py     # → label in Roboflow web UI
python scripts/02c_import_from_roboflow.py
python scripts/01_audit_dataset.py --rules configs/yolo/dataset_audit.yaml
python scripts/03_train_yolov8_seg.py --model models/yolo11s-seg.pt --epochs 50 --rtx3060-preset small
python scripts/05b_eval_threshold_sweep.py --weights runs/segment/<run>/weights/best.pt
python scripts/04_infer_yolov8_seg.py
python scripts/06_export_polygons_geojson.py

# Stage 2 (see docs/SOILING_STAGE2_GUIDE.md)
PYTHONPATH=. python scripts/12_ingest_nrel_soiling_map.py
PYTHONPATH=. python scripts/13_build_static_features.py
PYTHONPATH=. python scripts/10_train_soiling_model.py --run-name run_latest
```

Base YOLO weights (`yolo11s-seg.pt`) are gitignored — download to `models/` manually.

## Roadmap

- ✅ NAIP labeling, Stage 1 baseline (56% mAP50), threshold calibration, experiment matrix tooling
- ✅ Stage 2 honest leakage-free baseline on NREL panel labels with isotonic calibration + temporal holdout
- 🔄 **Stage 1 detector retrain** — diagnose-first RCA + iterative NAIP relabel + R0 retrain to break past 56% mAP50 (see [docs/PHASE1_HANDOFF.md](docs/PHASE1_HANDOFF.md))
- ✅ **Customer-readiness Tier 0–1** — `manifest.json` output contract, `solarsoiled` CLI with `tile / detect / score / recommend / run / eval` subcommands, per-AOI output namespace
- 🔄 **Tier 2** — model registry (`models/registry.yaml`), hardened AOI primitive with CRS validation (see [docs/PRODUCT_VISION.md](docs/PRODUCT_VISION.md))
- 📋 Multi-county / multi-vintage NAIP coverage
- 📋 Beta API surface (`/detect`, `/risk`, `/recommend`) with quality metadata in every response

## Documentation

| Doc | Purpose |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Operational core — commands, paths, dataset status |
| [docs/PHASE1_HANDOFF.md](docs/PHASE1_HANDOFF.md) | Active Stage 1 retrain runbook (diagnose-first + R0) |
| [docs/SOILING_STAGE2_GUIDE.md](docs/SOILING_STAGE2_GUIDE.md) | Stage 2 partner guide |
| [docs/PRODUCT_VISION.md](docs/PRODUCT_VISION.md) | Strategy, beta contract, customer-readiness arc |
| [docs/Q2_PLAN.md](docs/Q2_PLAN.md) | Quarter roadmap |
| [docs/NAIP_ROBOFLOW_WORKFLOW.md](docs/NAIP_ROBOFLOW_WORKFLOW.md) | Full tile → label → train → export reference |
| [docs/RTX3060_SETUP_GUIDE.md](docs/RTX3060_SETUP_GUIDE.md) | GPU env + presets |

## Requirements

Python 3.11, PyTorch 2.0+ with CUDA (RTX 3060 12 GB tested), GDAL/rasterio, ultralytics ≥ 11.0, Roboflow API. Full list in `requirements.txt`.
