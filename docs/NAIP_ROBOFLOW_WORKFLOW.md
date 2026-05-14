# NAIP + Roboflow Workflow

End-to-end pipeline for tiling NAIP imagery, labeling in Roboflow, and training YOLOv11 segmentation. The full Stage 1 pipeline lives in numbered scripts (`scripts/02_*` → `scripts/07_*`); this guide explains the order and the geospatial contract.

```
NAIP GeoTIFF
  ↓ scripts/02_tile_naip_image.py
PNG tiles + tile_index.json
  ↓ scripts/02b_export_to_roboflow.py
roboflow_upload/
  ↓ [manual labeling in Roboflow web UI]
roboflow_download/
  ↓ scripts/02c_import_from_roboflow.py
data/yolo/naip/
  ↓ scripts/01_audit_dataset.py        (validate)
  ↓ scripts/03_train_yolov8_seg.py     (or 03b for matrix)
runs/segment/<run>/weights/best.pt
  ↓ scripts/05b_eval_threshold_sweep.py (calibrate)
  ↓ scripts/04_infer_yolov8_seg.py      (inference)
  ↓ scripts/06_export_polygons_geojson.py
outputs/solar_arrays.geojson
```

---

## Geospatial contract

`data/interim/tile_index.json` is the linchpin. It maps each tile filename to its CRS, affine transform, source GeoTIFF, and bounds. Without it, polygon outputs cannot be georeferenced. **Never delete it; never overwrite without reading first.** Every script that touches tiles must preserve it.

Sample entry:
```json
"tile_000_000.png": {
  "transform": [630000.0, 1.0, 0, 4180640.0, 0, -1.0],
  "crs": "EPSG:32610",
  "width": 640, "height": 640,
  "source": "naip_scene_001.tif",
  "bounds": {...}
}
```

---

## Prerequisites

1. NAIP GeoTIFFs in `data/raw/` (filenames must contain `NAIP`). 3-band RGB, any CRS.
2. Roboflow account ([roboflow.com](https://roboflow.com)) — API key from settings.
3. `pip install -r requirements.txt`. For the GPU env, see `docs/RTX3060_SETUP_GUIDE.md`.

---

## Step 1 — Tile

```bash
python scripts/02_tile_naip_image.py
```

Produces `data/tiles/tile_*.png` (640×640, CLAHE-toned) and `data/interim/tile_index.json`. If band order looks wrong (blue-shifted output), edit `detect_rgb_band_order()` in `src/utils/naip_preprocessing.py`.

Optional GeoAI sourcing if you don't have NAIP TIFs yet:
```bash
python scripts/02_tile_naip_image.py --download-aoi "bbox:-122.5,37.7,-122.3,37.9"
```

Optional preview: `python scripts/research/08c_visualize_tiles.py`.

---

## Step 2 — Export to Roboflow

```bash
python scripts/02b_export_to_roboflow.py
```

Produces `roboflow_upload/images/{train,val,test}/` (70/15/15 random split), `data.yaml`, and **`roboflow_metadata.json`** (preserves tile_index links across the Roboflow round-trip).

Upload via web UI (recommended first time): create a project named `solar_arrays_naip`, type **Instance Segmentation**, upload the `images/` directory.

API alternative:
```python
from src.utils.roboflow_api import RoboflowClient
client = RoboflowClient(api_key="...")
client.upload_dataset(project_name="solar_arrays_naip",
                     dataset_dir=Path("roboflow_upload"),
                     create_if_missing=True)
```

---

## Step 3 — Label in Roboflow

Use the **Polygon tool** (not bounding box). Class name `solar_array`. Include rooftop, ground-mounted, partial arrays. Aim for 50+ labeled images minimum.

When done, generate a dataset version → export format **YOLOv11 (Segmentation)**. Note the version number.

---

## Step 4 — Import labeled data

Manual download: in Roboflow, click Download → YOLOv11 (Segmentation) → ZIP → extract to `roboflow_download/`.

API alternative:
```python
client.download_dataset(project_name="solar_arrays_naip", version=1,
                       format_type="yolov8-seg", output_dir=Path("roboflow_download"))
```

Then import to YOLO layout:
```bash
python scripts/02c_import_from_roboflow.py
```

Produces `data/yolo/naip/{images,labels}/{train,val,test}/`, training `data.yaml`, and preserves `roboflow_metadata.json`.

Verify counts and label format:
```bash
for split in train val test; do
  echo "$split: $(ls data/yolo/naip/images/$split/ | wc -l)"
done
head data/yolo/naip/labels/train/$(ls data/yolo/naip/labels/train/ | head -1)
# Expected: "0 0.45 0.30 0.65 0.30 0.65 0.50 0.45 0.50"
```

YOLO polygon format: `<class> x1 y1 x2 y2 ... xn yn` with normalized 0–1 coords; class always 0.

---

## Step 5 — Audit

```bash
python scripts/01_audit_dataset.py --rules configs/yolo/dataset_audit.yaml
```

Verifies every image has a label, flags polygon-size outliers, prints per-split stats. Must pass before training.

---

## Step 6 — Train

Single experiment:
```bash
python scripts/03_train_yolov8_seg.py --model models/yolo11s-seg.pt --epochs 50
```

Matrix (preferred for non-trivial runs):
```bash
python scripts/03b_train_experiment_matrix.py --config configs/yolo/experiments.yaml
```

Stage 1 retrain (current Phase 1 active workstream) — see [docs/PHASE1_HANDOFF.md](PHASE1_HANDOFF.md). Joint Duke + NAIP training is paused; resumes at R1+ in the ramp curriculum after R0 lands.

Model sizes for this dataset (~250 tiles): `yolo11s-seg.pt` is the sweet spot. `yolo11n` for fast iteration / CPU; `yolo11m` if you have headroom and want a few more mAP points.

Outputs land in `runs/segment/<run_name>/`: `results.csv`, `weights/best.pt` (use this, not `last.pt`), training plots.

Calibrate confidence/IoU thresholds:
```bash
python scripts/05b_eval_threshold_sweep.py \
  --weights runs/segment/<run_name>/weights/best.pt \
  --test_dir data/yolo/naip/images/test/
```

Recommended starting thresholds for NAIP: `conf=0.40, iou=0.40`.

---

## Step 7 — Inference and geospatial export

```bash
python scripts/04_infer_yolov8_seg.py
python scripts/06_export_polygons_geojson.py
python scripts/07_extract_array_features.py
```

`06_export_polygons_geojson.py` is where `tile_index.json` becomes load-bearing — it converts normalized YOLO coords → pixel coords → world coords using the per-tile affine transform, then writes `outputs/solar_arrays.geojson`. `07_extract_array_features.py` then computes geometric features (area, perimeter, compactness) and spatial context, writing `outputs/array_features.{parquet,geo.parquet}`.

---

## Multi-iteration workflow

Each Roboflow version is immutable, so each labeling pass adds a new version. Re-running `02c_import_from_roboflow.py` against a newer version refreshes `data/yolo/naip/` in place. `tile_index.json` survives; new tiles get new entries.

```
Iteration 2:
  scripts/02_tile_naip_image.py       # new scene
  scripts/02b_export_to_roboflow.py   # new Roboflow version
  [label in Roboflow]
  scripts/02c_import_from_roboflow.py # refresh data/yolo/naip/
  scripts/03_train_yolov8_seg.py      # retrain on combined data
```

---

## Troubleshooting

**"No NAIP files found":** filenames must contain `NAIP`. Rename or edit the glob in `02_tile_naip_image.py`.

**Tile CRS mismatch:** all source NAIP TIFs need the same CRS. Check `tile_index.json`.

**Colors look wrong (blue-shifted):** band order is BGR. Edit `detect_rgb_band_order()` in `src/utils/naip_preprocessing.py`.

**Roboflow API download fails:** use the manual web UI download instead.

**Training is on CPU and slow:** check `python -c "import torch; print(torch.cuda.is_available())"`. If False, see `docs/RTX3060_SETUP_GUIDE.md`. CPU fallback configs live in `configs/yolo/experiments_laptop.yaml`.

**GeoJSON coords don't match expected bounds:** verify `tile_index.json` has correct CRS and that source NAIP TIFs have valid geospatial metadata.

---

## Key files

| File | Purpose |
|---|---|
| `data/interim/tile_index.json` | Geospatial metadata — **do not delete** |
| `scripts/02_tile_naip_image.py` | NAIP → tiles + tile_index |
| `scripts/02b_export_to_roboflow.py` | Tiles → Roboflow upload directory |
| `scripts/02c_import_from_roboflow.py` | Roboflow download → YOLO layout |
| `scripts/01_audit_dataset.py` | Dataset validation |
| `scripts/03_train_yolov8_seg.py`, `03b_train_experiment_matrix.py` | Train (single / matrix) |
| `scripts/05b_eval_threshold_sweep.py` | Threshold calibration |
| `scripts/04_infer_yolov8_seg.py`, `06_export_polygons_geojson.py`, `07_extract_array_features.py` | Inference + geospatial export |
| `configs/yolo/experiments.yaml`, `experiments_laptop.yaml` | Experiment matrices |
