# Phase 1 Handoff — Stage 1 Detector Retrain

## TL;DR — Iterative Relabel + Retrain Approach

**Goal:** Use the existing SAHI baseline as a label-gap finder, fix labels in Roboflow, retrain R0 against patched labels, repeat until SAHI F1 clears the 0.55 beta gate (measured by `05d_sahi_threshold_sweep.py` on the val split).

**Production metric is SAHI F1, not mAP50.** `model.val()` mAP50 feeds the model full tiles — the mode never used in production. The calibrated SAHI F1 from `05d` is the correct number to track. Current best: R2-cameron-20260509 at SAHI F1=0.396. Beta gate: 0.55. GA gate: 0.65.

**Steps:**
1. **Relabel train set** (highest priority — val/test are already relabeled). Train labels are still sparse; the model trains on bad signal and is scored against good labels. Relabel in Roboflow using the same workflow as the val/test rounds.
2. **Train R0** — warm-start `yolo11s-seg` from `models/sahi_baseline_train7.pt` on Santa Cruz NAIP train/val only (NAIP-only, no Duke yet). The SAHI baseline already finds small panels our 60 cm hand labels miss; we want to keep that prior while we patch the labels.
3. **Audit labels** — run R0 with `--sahi` on val/test, generate overlays showing where labels likely have gaps (model finds panels we never labeled).
4. **Relabel iteratively** — fix labels in batches based on model's detections, retrain R0 after each batch.
5. **Ramp to Duke** — only after R0 SAHI F1 ≥ 0.45 on val, proceed to R1 (joint NAIP + Duke), warm-starting from R0's best.pt.

---

## Quick Start — The workflow

**Step 0 (one-time): Verify tile names match Roboflow**

Before you start relabeling, confirm the tile filenames are consistent:

```bash
PYTHONPATH=. python3 scripts/labeling/16_vintage_audit.py \
    --data-root data/yolo/naip \
    --tile-index data/interim/tile_index.json
```

This should show all 249 tiles matched with no renames. If all green, proceed. If there are mismatches, the overlay filenames won't match Roboflow — ping Cameron.

**Step 1: Train R0 on Santa Cruz NAIP tiles (warm-start from SAHI baseline)**

```bash
# R0 warm-starts from models/sahi_baseline_train7.pt per the YAML config
# (fresh-COCO would discard the small-panel prior we need)
python3 scripts/03b_train_experiment_matrix.py \
    --config configs/yolo/experiments_joint_v2_ramp.yaml \
    --experiment R0 \
    --data data/yolo/naip/data.yaml
```

The run will be saved as `runs/segment/R0_<timestamp>/`. This is your baseline.

**Step 2: Generate overlays to audit labels**

Once R0 finishes training, run inference to find label gaps:

```bash
# Run RCA on the fresh R0 model. --sahi matches our production inference path,
# so the conf/iou calibration from 05d transfers directly.
python3 scripts/05c_per_detection_rca.py \
    --weights runs/segment/R0_<timestamp>/weights/best.pt \
    --data data/yolo/naip/data.yaml \
    --splits val test \
    --sahi --conf 0.05 --iou 0.50 \
    --run-name R0_<timestamp>

# Render overlay PNGs grouped by failure mode
python3 scripts/labeling/18_bucket_overlays.py \
    --csv outputs/eval/R0_<timestamp>/per_detection.csv \
    --bucket-expr "class=fp AND num_other_panels_in_tile=0" \
    --top 20 --rank-by conf --rank-desc

python3 scripts/labeling/18_bucket_overlays.py \
    --csv outputs/eval/R0_<timestamp>/per_detection.csv \
    --bucket confident_fp --top 20

python3 scripts/labeling/18_bucket_overlays.py \
    --csv outputs/eval/R0_<timestamp>/per_detection.csv \
    --bucket worst_small_fn --top 20
```

(Replace `R0_<timestamp>` with your actual run folder name from `runs/segment/`)

Then open `outputs/label_viz/R0_<timestamp>/` in your file browser.

**Step 3: Relabel in batches & retrain**

For each batch:

1. Review PNGs (red outline = label, cyan = model prediction). Is the cyan a real panel?
   - Yes → add tile filename to Roboflow relabel list
   - No → skip (genuine FP, no action)
2. Relabel the batch in Roboflow.
3. Retrain R0 with the updated labels:
   ```bash
   python3 scripts/03b_train_experiment_matrix.py \
       --config configs/yolo/experiments_joint_v2_ramp.yaml \
       --experiment R0 \
       --data data/yolo/naip/data.yaml
   ```
4. Generate new overlays and check if metrics improve (use 05c + 18_bucket_overlays again).
5. **Record result**: Find `outputs/eval/ramp_curve.csv`. Copy the last row into Results table below.

**Step 4: When R0 is solid (SAHI F1 ≥ 0.45 on NAIP val), proceed to Duke ramp**

Once you're satisfied with R0:

```bash
# Build joint lists at 20:1 NAIP:Duke ratio
python3 scripts/02g_build_joint_v2_lists.py --naip-repeat 580

# Train R1 — IMPORTANT: edit configs/yolo/experiments_joint_v2_ramp.yaml first
# so R1's `model:` field points at your converged R0's best.pt (NOT
# sahi_baseline_train7.pt, which would discard the relabeled-NAIP improvements).
python3 scripts/03b_train_experiment_matrix.py \
    --config configs/yolo/experiments_joint_v2_ramp.yaml \
    --experiment R1
```

**Results table** (record each R0 retrain iteration):

| batch | run_name | step | naip_test_map50 | naip_test_precision | naip_test_recall | status | tiles_relabeled | notes |
|---|---|---|---|---|---|---|---|---|
| B01_warm | R0_<date> | R0 warm-start (SAHI) | ? | ? | ? | in-progress | 0 | baseline against current labels |
| B02 | R0_<date> | R0 retrain | ? | ? | ? | queued | ~20 | after batch 1 relabel |
| B03 | R0_<date> | R0 retrain | ? | ? | ? | queued | ~20 | after batch 2 relabel |

**Key point:** Your weights and training sessions are already saved in `runs/segment/`. Populate the Results table after each iteration so you have a permanent record.

---

## What changed 

- **det_match + overlay_render shared modules** (`src/utils/`) — single matcher used by both `compute_sahi_confusion_matrix.py` and the new RCA harness. Verified byte-identical against the pre-refactor confusion-matrix output.
- **`05c_per_detection_rca.py`** — owns its own SAHI loop (preserves per-polygon confidence that 04 throws away). Emits `outputs/eval/<run>/per_detection.csv` with one row per TP/FP/FN, plus a `--summarize` aggregator that produces `failure_modes.json`.
- **`labeling/18_bucket_overlays.py`** — renders top-N overlays per failure-mode bucket (worst_small_fn, confident_fp, large_fp, edge_fn, worst_iou_tp, or arbitrary `--bucket-expr "class=fp AND num_other_panels_in_tile=0"`).
- **`05d_sahi_threshold_sweep.py`** — SAHI-aware sweep (does not collapse into pandas re-thresholding; GREEDYNMM+IOS is non-monotonic in conf so we re-run SAHI per combo).
- **`01b_compare_naip_duke_distributions.py`** — domain equivalence baseline. Full run says NAIP median panel = 24.3 m² vs Duke 1.7 m² (KS=0.895 on area_m²). Labels differ by convention (NAIP = whole-array, Duke = per-panel). This explains why Joint v1 collapsed both domains and why your overnight run learned Duke's per-panel prior.
- **`05e_ramp_eval.py` + `configs/yolo/experiments_joint_v2_ramp.yaml`** — per-step ramp eval helper + R0–R5 curriculum config. Each step warm-starts independently from `sahi_baseline_train7.pt`. Halt rule: if NAIP test mAP50 drops by >0.07 vs 0.563 baseline, prior step's weights become the production candidate. The mAP50 delta is used as a fast regression signal; the definitive production metric is SAHI F1 from `05d`.

---

## What the RCA actually shows

Run on `sahi_baseline_train7.pt` against NAIP val+test (75 tiles, conf floor 0.05):

```
overall:  TP=125  FP=360  FN=101    P=0.26  R=0.55
```

**Density bucket is the loudest signal:**

| density | tp | fp | fn | precision |
|---|---:|---:|---:|---:|
| `alone` (no GT panels) | 1 | 65 | 3 | **1.5%** |
| `few_1_3`              | 39 | 100 | 22 | 28% |
| `crowded_ge4`          | 85 | 195 | 76 | 30% |

**The 20 GT-empty tiles host 65 of the 360 total FPs.** Long tail: `tile_000150` has 13 detections at max conf 0.500. That tile is almost certainly populated with real panels that the 0.6 m resolution made hard to label by hand. The model is finding them; our labels weren't.

**Conf distribution of alone-tile FPs:**

| conf range | alone-FPs | TPs |
|---|---:|---:|
| [0.05,0.10) | 27 | 5 |
| [0.10,0.15) | 15 | 10 |
| [0.15,0.20) | 6 | 13 |
| [0.20,0.30) | 12 | 21 |
| [0.30,0.50) | 4 | 37 |
| [0.50,1.00) | 1 | 39 |

Most alone-FPs are at conf <0.20; most TPs are at conf >0.20. Threshold lift alone (no retrain, no relabel) cleans up most of this — but we should also fix the labels because the precision number is misleading our own training signal.

---

## Your three jobs

### 1. Label audit (parallel with 2)

The new overlays are sitting in `outputs/label_viz/sahi_baseline_train7/` after Cameron runs `scripts/run_phase1_diagnostics.sh`. Specifically:

- `expr_class_fp_AND_num_other_panels_in_tile_0/` — the empty-tile hits, ranked by conf desc. Highest priority.
- `confident_fp/` — top-N FPs across the entire eval set, ranked by conf.
- `large_fp/` — FPs where `pred_area >= 4000 px²`. Bigger than typical NAIP individual polygons; either really big arrays or really-wrong predictions.
- `worst_small_fn/` — small panels we missed.

For each tile in `expr_class_fp_AND_num_other_panels_in_tile_0/`:

1. Open the PNG. Red = our label (none, since these are GT-empty), cyan = model prediction.
2. Decide: is the cyan a real panel? (Use the Esri high-res inset if needed — `15_label_disagreement.py --esri-inset` will fetch fresher imagery for adjudication.)
3. If yes → add to a re-label task list. We'll batch fix in Roboflow.
4. If no → log as a genuine FP for post-retrain investigation.

You can also run `scripts/labeling/15_label_disagreement.py --weights models/sahi_baseline_train7.pt --split test --esri-inset` for a more polished version of the same task with side-by-side high-res reference. That tool has been pre-existing; the new `18` is a per-detection lens on top of it.

**Output:** a list of tile filenames to re-label (or a Roboflow project link). Keep it under ~25 tiles for the Wednesday deadline; we can do another pass after.

### Batch queue template

Use one batch at a time and keep the queue in this exact shape so nothing gets lost between review, Roboflow, and retraining.

| batch | tiles | likely issue | action | status | retrain step | notes |
|---|---|---|---|---|---|---|
| B01 | 15-25 tiles | label gap / genuine FP | relabel / skip | queued | R0 | keep under 25 tiles |
| B02 | 15-25 tiles | label gap / genuine FP | relabel / skip | queued | R0 | same rules |
| B03 | 15-25 tiles | label gap / genuine FP | relabel / skip | queued | R1+ | only after R0 reproduces |

Batch rules:

1. Prioritize `expr_class_fp_AND_num_other_panels_in_tile_0/` first, then `confident_fp/`, then `worst_small_fn/`.
2. Put every tile in exactly one batch, then stop and relabel before adding more.
3. Mark each tile `confirmed_panel`, `true_fp`, or `needs_second_pass` so you do not re-review the same evidence.
4. Only move to the next Duke step after the current step reproduces cleanly and the batch notes are closed out.

### Per-batch loop

Repeat this loop for each batch:

```bash
# 1) Build or update the tile list for the batch.
# 2) Relabel in Roboflow.
# 3) Rebuild lists if Duke composition changed.
python scripts/02g_build_joint_v2_lists.py --naip-repeat 580

# 4) Retrain the current step.
python scripts/03b_train_experiment_matrix.py \
    --config configs/yolo/experiments_joint_v2_ramp.yaml \
    --experiment R0

# 5) Evaluate and record the result.
python scripts/05e_ramp_eval.py \
    --run R0 \
    --weights runs/segment/<your-run-name>/weights/best.pt
```

If the batch is NAIP-only cleanup, keep it on R0. If it introduces Duke, only advance to R1 after the R0 checkpoint is still healthy.

### 2. Train R0 (NAIP-only reproduction)

This step exists to confirm the retraining harness is sane on patched labels before we touch Duke. After training, run `05d_sahi_threshold_sweep.py` to get the SAHI F1. Target ≥ 0.45 SAHI F1 on NAIP val before advancing to Duke. If `model.val()` mAP50 is very low (e.g., 0.20) but SAHI F1 is improving, that's expected — the model is calibrated for sliced inference. If SAHI F1 is also near zero, the harness has drifted.

```bash
# Step A: run the matrix runner on the R0 entry only.
python scripts/03b_train_experiment_matrix.py \
    --config configs/yolo/experiments_joint_v2_ramp.yaml \
    --experiment R0 \
    --data data/yolo/naip/data.yaml
# Step B: confirm the R0 entry still stays NAIP-only.
# If you need to smoke-test the full ramp later, drop --experiment and let the
# per-entry data fields in the ramp config drive the split.
# R0 uses data/yolo/naip/data.yaml; R1+ use data/yolo/joint_v2/data.yaml.

# Step C: per-step eval (writes to outputs/eval/ramp_curve.csv)
python scripts/05e_ramp_eval.py \
    --run R0 \
    --weights runs/segment/<your-run-name>/weights/best.pt
```

Expected: After `05d` sweep, SAHI F1 ≥ 0.40 on NAIP val. `model.val()` mAP50 will be lower than 0.563 (SAHI-calibrated models score low on full-tile eval) — do not use mAP50 as the gate. If SAHI F1 is near zero, **stop** — the harness has drifted.

### 3. Train R1 — only after R0 lands clean

```bash
# Build the joint training lists at 20:1 NAIP:Duke ratio
python scripts/02g_build_joint_v2_lists.py --naip-repeat 580

python scripts/03b_train_experiment_matrix.py \
    --config configs/yolo/experiments_joint_v2_ramp.yaml \
    --experiment R1
python scripts/05e_ramp_eval.py --run R1 --weights runs/segment/<run>/weights/best.pt
```

`05e_ramp_eval.py` will print **HALT** if NAIP regression Δ > 0.07 and the previous step's weights become the production candidate. If R1 halts, R2+ does not run.

**Watch precision separately, not just mAP50.** Your overnight run hit mAP50 0.128 with precision <0.1 — that's the over-prediction failure that mAP50 alone undersells. The `ramp_curve.csv` tracks `naip_test_precision` as a separate column for exactly this reason.

---

## Hyperparameters — copy verbatim

The R-series in `configs/yolo/experiments_joint_v2_ramp.yaml` mirrors `experiments_joint_v2.yaml`. Critical pieces that **must** survive any edit:

```yaml
optimizer: SGD          # NEVER auto — silently overrides lr0 to AdamW(0.002)
lr0: 0.001              # damped to preserve warm-start
auto_augment: null      # default randaugment caused starburst artifacts on 160px Duke
erasing: 0.0            # default 0.4 also caused starbursts
translate: 0.0          # 0.1 = 16px shift on 160px Duke (small-array-magnitude)
mosaic: 0.5             # damped from default 1.0
copy_paste: 0.0         # damped from default ~0.3
```

**Strategy note (settled 2026-05-04):** R0 **warm-starts from `models/sahi_baseline_train7.pt`** rather than fresh-COCO. The SAHI baseline already learned to detect small panels at 60 cm resolution — that's the prior we want to preserve while we patch the labels in Roboflow. Fresh COCO init would throw that prior away, and patched labels at 60 cm still won't teach it back from scratch. The "fresh-init" framing in the TL;DR refers to *fresh per relabel iteration* (each R0 retrain re-warmstarts from sahi_baseline against the latest patched labels), not fresh-from-COCO. R1+ (Duke ramp) should warm-start from R0's converged best.pt — but the R1–R5 entries in `experiments_joint_v2_ramp.yaml` still point at `sahi_baseline_train7.pt`; update those when ramp resumes.

---

## Calibration — SAHI baseline already swept (2026-05-04)

Cameron's 05d sweep on `sahi_baseline_train7.pt` finished. Calibrated SAHI operating point:

| | conf | iou | F1 | P | R | tp | fp | fn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Best** | **0.30** | **0.50** | **0.536** | **0.703** | **0.433** | 52 | 22 | 68 |
| Runner-up | 0.25 | 0.50 | 0.533 | 0.622 | 0.467 | 56 | 34 | 64 |
| High-recall | 0.10 | 0.50 | 0.459 | 0.40 | 0.54 | — | — | — |

**Context:** These numbers were measured against the **old (sparse) val labels**. Val has since been relabeled (multiple rounds, now ~212 GT objects vs 61 before). F1 numbers will differ on current labels — re-run 05d on any weights you want to compare rather than reading this table directly.

**Current production (R2-cameron-20260509) on current labels:** SAHI F1=0.396 at conf=0.40, iou=0.50. That's the baseline to beat with train relabeling + R0 retrain.

**Use `conf=0.05, iou=0.50` as the 05c conf floor** (low so 05d can re-threshold). After R0 lands, re-run 05d on R0's weights (`scripts/05d_sahi_threshold_sweep.py --weights runs/segment/R0_<timestamp>/weights/best.pt --run-name R0_<timestamp>`) — the calibrated conf will shift from 0.30/0.40 once trained on relabeled data.

Note: After train relabeling, precision should materially improve because the alone-tile label gaps will be closed — the same weights will show higher P with fewer false FPs.

---

## Don't do these

- **Hard-negative augmentation on alone tiles.** Those tiles are likely under-labeled, not negative.
- **Pre-train on Duke alone, fine-tune on NAIP.** Tried in the 0426 cycle, gave 0.028 transfer mAP50. Abandoned.
- **`optimizer: auto`.** Silently overrides `lr0` to AdamW(0.002), wrecks warm-start in epoch 1.
- **Increase Duke share before R0 reproduces baseline.** R1 (`--naip-repeat 580`) is the smallest reasonable Duke fraction; even that should wait.
- **Use `04_infer_yolov8_seg.py` for RCA** — its `.txt` output discards `pred.score.value`. Use `05c` instead.

---

## Files you'll touch

| | path |
|---|---|
| read | `outputs/eval/sahi_baseline_train7/per_detection.csv` |
| read | `outputs/eval/sahi_baseline_train7/failure_modes.json` |
| read | `outputs/label_viz/sahi_baseline_train7/<bucket>/*.png` |
| read | `configs/yolo/experiments_joint_v2_ramp.yaml` |
| read | `outputs/eval/domain_equivalence/report.md` |
| run | `scripts/03_train_yolov8_seg.py` (existing) |
| run | `scripts/05e_ramp_eval.py` (new — drives 05 and 05c, appends ramp_curve.csv) |
| write | `outputs/eval/ramp_curve.csv` (one row per ramp step) |
| write | `runs/segment/R0_<timestamp>/` and onward |

---

## If something goes wrong

| symptom | diagnosis | action |
|---|---|---|
| `optimizer: auto silently overriding to AdamW` warning at train start | YAML edit lost the explicit SGD line | Restore `optimizer: SGD` in the experiment entry |
| Box mAP50 collapses from 0.696 → 0.004 in epoch 1 | Warm-start being destroyed by AdamW(0.002) | Same — verify SGD + lr0=0.001 |
| RandAugment starburst tiles in train batches | `auto_augment: null` and `erasing: 0.0` not set explicitly | Set both in the YAML; defaults pass through |
| R0 lands below 0.45 mAP50 | Harness drift, possibly `joint_v2/data.yaml` instead of `naip/data.yaml` | Override `--data data/yolo/naip/data.yaml`; rerun |
| `05e_ramp_eval.py` prints HALT | NAIP regression Δ > 0.07 | Stop the ramp. Previous step's weights are the candidate. |

---

## Glossary

- **R0–R5**: Duke ramp curriculum steps (`configs/yolo/experiments_joint_v2_ramp.yaml`). R0 = NAIP-only; R5 = current 1:1 effective ratio.
- **alone tile**: NAIP tile with zero GT-labeled panels.
- **per_detection.csv**: one row per TP/FP/FN with size/edge/density/conf metadata; consumed by `18` for overlay rendering and `05d` for sweep metrics.
- **SAHI baseline checkpoint**: `models/sahi_baseline_train7.pt` (0.563 model.val() mAP50; SAHI F1=0.345 at conf=0.30 on old labels, 0.396 for R2-cameron-20260509 at conf=0.40 on current relabeled val).
- **Production metric**: SAHI F1 from `05d_sahi_threshold_sweep.py`. Beta gate ≥ 0.55. GA gate ≥ 0.65. `model.val()` mAP50 is used only as a fast regression signal in `05e`.

---

## Questions to ping Cameron about

1. Did `05d` finish and what's the calibrated conf?
2. Did the alone-tile FP overlays mostly come up as label gaps (expected) or genuine FPs?
3. Has the re-label batch landed in Roboflow yet?
