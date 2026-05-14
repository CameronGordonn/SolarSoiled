# Post-Roboflow Import Runbook

You have a COCO Segmentation export from Roboflow after relabeling a batch of tiles. This doc walks you from that zip to a retrained R0 with eval artifacts.

---

## Before you start — check the export for split structure

This is the most important thing to verify before running anything. Unzip the Roboflow export somewhere temporary and check what's inside:

```
# If Roboflow preserved splits, you'll see:
your_export/
├── train/   ← _annotations.coco.json + images
├── valid/   ← _annotations.coco.json + images
└── test/    ← _annotations.coco.json + images

# If Roboflow collapsed everything to train, you'll see:
your_export/
└── train/   ← _annotations.coco.json + ALL 249 images
```

**Why this matters:** the existing `data/yolo/naip/` already has the correct 174/37/38 train/val/test split that all previous metrics are measured against. If you re-import with random re-splitting, the val/test sets change and metrics become non-comparable to the baseline.

**Based on what's already in the repo:** the previous `roboflow_upload/YOLOv8 (Segmentation)/` export had all 249 tiles flattened into `train/` — Roboflow did not preserve splits. The new COCO export may do the same.

---

## Path A — Export has train/valid/test splits (preferred)

Extract to `roboflow_download/` at the repo root (create it if it doesn't exist):

```
solar-soiling-ml/
└── roboflow_download/
    ├── train/   ← _annotations.coco.json + images
    ├── valid/   ← _annotations.coco.json + images
    └── test/    ← _annotations.coco.json + images
```

Then run the import:

```bash
PYTHONPATH=. python scripts/02c_import_from_roboflow.py
```

The script detects COCO format automatically, converts the JSON to YOLO `.txt` labels, and copies images into `data/yolo/naip/`. Splits are preserved.

Skip to **Step 2 — Validate** below.

---

## Path B — Export is train-only (all 249 tiles in train/)

Do **not** run `02c_import_from_roboflow.py` on a train-only export. It would auto-split 70/15/15 randomly, changing val/test composition and making metrics non-comparable to the R0 baseline.

Instead, selectively update only the labels for tiles you actually relabeled. The approach:

**Option B1 — Manual selective copy (safest):**

1. Convert the COCO JSON to YOLO `.txt` files using the converter in `02c`:
   ```bash
   python - <<'EOF'
   import sys, json
   from pathlib import Path
   sys.path.insert(0, ".")
   
   # Point at your unzipped export
   coco_json = Path("roboflow_download/train/_annotations.coco.json")
   dst_labels = Path("/tmp/updated_labels")
   
   from scripts.import_02c import coco_to_yolo_labels
   EOF
   ```
   
   Actually, just call the function directly:
   ```python
   # In a python shell from the repo root:
   import sys; sys.path.insert(0, ".")
   from pathlib import Path
   import importlib.util
   spec = importlib.util.spec_from_file_location("m", "scripts/02c_import_from_roboflow.py")
   m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
   
   m.coco_to_yolo_labels(
       Path("roboflow_download/train/_annotations.coco.json"),
       Path("/tmp/updated_labels")
   )
   ```

2. For each tile you relabeled in Roboflow, copy its updated `.txt` from `/tmp/updated_labels/` into the correct split directory in `data/yolo/naip/labels/{train,val,test}/`. Use the existing split assignment — do not move tiles between splits.

3. Tiles you did not relabel: leave their `.txt` files untouched.

**Option B2 — Ask Cameron to re-export with splits:**

In Roboflow, go to the dataset version → Generate → make sure "Split" is set (not "All train"). Re-export as COCO Segmentation. This is cleaner and avoids manual file juggling.

---

## Step 2 — Validate the imported labels

Run these after either import path, before any training:

```bash
# Coordinate range, empty labels, object size distribution
PYTHONPATH=. python scripts/labeling/validate_labels.py \
    --data data/yolo/naip/data.yaml \
    --splits train val test

# Confirm tile filenames still match tile_index.json
PYTHONPATH=. python scripts/labeling/16_vintage_audit.py \
    --data-root data/yolo/naip \
    --tile-index data/interim/tile_index.json
```

**What to look for:**

- `validate_labels.py`: zero coordinate violations (any value outside 0–1 means bad COCO conversion). Zero should be zero — stop if not.
- `validate_labels.py`: empty label count should be similar to before or slightly lower (relabeling adds polygons, not removes them). A spike means Roboflow deleted annotations.
- `16_vintage_audit.py`: all tiles should show as matched. Unmatched tiles mean the Roboflow hash changed on re-export — tile_index lookups used by the eval harness will break silently if you proceed.

**If unmatched tiles appear:** Roboflow appends `_png.rf.<HASH>` on upload; the hash can change between exports. The import script's `strip_roboflow_suffix()` handles this at lookup time, so training won't break, but the vintage audit will flag it. Ping Cameron if this happens — it may need a tile_index patch.

---

## Step 3 — Audit object counts

```bash
PYTHONPATH=. python scripts/01_audit_dataset.py \
    --config configs/yolo/dataset_audit.yaml
```

Compare against the previous run. Object count in train should be equal to or higher than before (you added annotations). If it's lower, Roboflow deleted a polygon somewhere — check the version history for affected tiles before proceeding.

---

## Step 4 — Retrain R0

```bash
python scripts/03b_train_experiment_matrix.py \
    --config configs/yolo/experiments_joint_v2_ramp.yaml \
    --experiment R0 \
    --data data/yolo/naip/data.yaml
```

**Before running, verify the R0 entry in `configs/yolo/experiments_joint_v2_ramp.yaml` has these settings — they must not have drifted:**

```yaml
optimizer: SGD          # NEVER auto — silently overrides lr0 to AdamW(0.002)
lr0: 0.001
auto_augment: null
erasing: 0.0
translate: 0.0
mosaic: 0.5
copy_paste: 0.0
model: models/sahi_baseline_train7.pt   # warm-start checkpoint
data: data/yolo/naip/data.yaml          # NAIP only for R0
```

**Failure modes during training:**

| Symptom | Cause | Fix |
|---|---|---|
| Box mAP50 collapses epoch 1 (e.g. 0.696 → ~0.004) | `optimizer: auto` applied AdamW(0.002), destroying warm-start | Restore `optimizer: SGD`, restart |
| Starburst tiles visible in train batch previews | `auto_augment` or `erasing` reverted to defaults | Set both explicitly in YAML |
| Training uses `joint_v2/data.yaml` | YAML `data:` field wrong | Add `--data data/yolo/naip/data.yaml` override |

---

## Step 5 — Evaluate and record

```bash
python scripts/05e_ramp_eval.py \
    --run R0 \
    --weights runs/segment/R0_<timestamp>/weights/best.pt
```

This appends one row to `outputs/eval/ramp_curve.csv`. Update the Results table in [docs/PHASE1_HANDOFF.md](PHASE1_HANDOFF.md).

**Acceptance bar:** NAIP test mAP50 ≥ 0.55. Below 0.45 = harness drift, stop.  
**Watch precision separately** — the over-prediction failure mode (mAP50 0.128 / precision <0.1) is undersold by mAP50 alone.

---

## Step 6 — Full eval pipeline

```bash
solarsoiled eval \
    --weights runs/segment/R0_<timestamp>/weights/best.pt \
    --full \
    --run-name R0_<timestamp>
```

Runs 05c RCA → summarize → 05d SAHI sweep → overlay rendering → HTML report in one shot. Takes 20–40 min. Report lands at `outputs/eval/R0_<timestamp>/report.html`.

---

## Before retraining — verify your previous run config

If your last R0 run regressed (mAP50 < 0.45, or precision collapsed while recall went up), check the training config before retraining. Run this from your repo root and share the output with Cameron:

```bash
cat runs/segment/<your_R0_run>/args.yaml | grep -E "optimizer|lr0|model|data|auto_augment|erasing|translate"
```

Expected output for a clean R0:
```
optimizer: SGD
lr0: 0.001
model: models/sahi_baseline_train7.pt
data: .../data/yolo/naip/data.yaml
auto_augment: null
erasing: 0.0
translate: 0.0
```

Any deviation from the above is the likely cause. Common failures:

| What you see | What happened | Fix |
|---|---|---|
| `optimizer: Adam` or `optimizer: auto` | Ultralytics overrode to AdamW(0.002), destroying warm-start in epoch 1 | Restore `optimizer: SGD` in the YAML |
| `auto_augment: randaugment` | Default wasn't overridden — causes starburst artifacts on small NAIP arrays | Set `auto_augment: null` explicitly |
| `erasing: 0.4` | Default wasn't overridden | Set `erasing: 0.0` |
| `translate: 0.1` | Default wasn't overridden — 10% shift is too large for 60cm arrays | Set `translate: 0.0` |
| `model: yolo11s-seg.pt` or COCO path | `sahi_baseline_train7.pt` wasn't found or config pointed elsewhere | Confirm `models/sahi_baseline_train7.pt` exists on your machine; check YAML `model:` field |
| `data: .../joint_v2/data.yaml` | Wrong dataset — R0 must be NAIP-only | Add `--data data/yolo/naip/data.yaml` override |

The precision-collapse + recall-up pattern (e.g. P: 0.703→0.305, R: 0.433→0.495) is specifically the `optimizer: auto` signature — same failure mode as the joint run in May.

---

## Checklist

```
[ ] Checked export zip for split structure (train/valid/test vs train-only)
[ ] Used Path A (02c import) if splits present, Path B (selective copy) if train-only
[ ] validate_labels: zero coordinate violations
[ ] validate_labels: empty label count not spiked vs. previous run
[ ] 16_vintage_audit: all tiles matched
[ ] 01_audit_dataset: object counts per split equal or higher than before
[ ] experiments_joint_v2_ramp.yaml R0 entry: optimizer=SGD, data=naip, model=sahi_baseline
[ ] ramp_eval: mAP50 ≥ 0.55, precision not collapsed
[ ] Results table in PHASE1_HANDOFF.md updated
[ ] solarsoiled eval --full run, report.html reviewed
```

---