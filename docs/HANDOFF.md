# SolarSoiled — Project Handoff (May 2026)

For whoever picks this up next: Cameron, Josh, or a new contributor. This doc covers project state and next steps as of graduation (May 2026). It does **not** re-cover train relabeling — that runbook lives in [PHASE1_HANDOFF.md](PHASE1_HANDOFF.md). For quarter-level priorities see [Q2_PLAN.md](Q2_PLAN.md). For product direction see [PRODUCT_VISION.md](PRODUCT_VISION.md).

---

## 1. Project State

### Stage 1 — Panel Detection

| Checkpoint | NAIP test mAP50 | SAHI F1 (val) | SAHI conf | Status |
|---|---:|---:|---:|---|
| `stage1-v0.5-baseline` (`sahi_baseline_train7.pt`) | 0.563 | 0.345 | 0.15 | Superseded |
| `r0-cameron-20260510` | 0.346 | — | — | Regression — do not use |
| **`r2-cameron-20260509`** (production) | 0.263 | **0.396** | 0.40 | **Current production** |

Production metric is **SAHI F1** from `scripts/05d_sahi_threshold_sweep.py` — not `model.val()` mAP50. The production model runs sliced inference (SAHI); its non-SAHI mAP50 is low by design.

Gates: **beta** = SAHI F1 ≥ 0.55 · **GA** = SAHI F1 ≥ 0.65. Current: 0.396 (below beta gate).

Train relabeling is in progress (val/test relabeled; train not yet). R0 retrain + iterative relabel is the active path — see [PHASE1_HANDOFF.md](PHASE1_HANDOFF.md).

### Stage 2 — Soiling Risk Model

| Run | Spatial-CV AUC | Holdout-2022 AUC | Status |
|---|---:|---:|---|
| `run_k_patched_holdout2022` | 0.63 | 0.66 | **Current production (beta)** |

Gate: AUC ≥ 0.70 on both spatial-CV and year-holdout. Current: below gate. Ships beta.

Labels: NREL panel CSV (`nrel_panel`), 10 km spatial CV, isotonic calibrated, leakage-free. N ≈ 640 training rows.

### CLI / Product Surface

| Tier | Scope | Status |
|---|---|---|
| Tier 0 — output manifests + pyproject.toml | Every artifact-producing script writes `manifest.json` | `shipped` |
| Tier 1 — `solarsoiled` CLI | `tile / detect / score / recommend / run / eval` | `shipped` |
| Tier 2 — model registry + AOI primitive | Registry resolving `production`/`latest`; AOI WGS84 hardening | `in progress` |
| Tier 3 — partner UX polish | `eval --report` HTML | `shipped`; Dockerfile, CI contract test, partner example | `queued` |

Public repo (`SolarSoiled`) is live on GitHub (Apache 2.0). Landing page live at GitHub Pages. **Important:** `LICENSE` file not yet added to the public repo — add before any commercial activity.

---

## 2. Stage 1 Next Steps (no relabeling required)

These can run in order as soon as Josh's compute is available.

**Step 1 — R0 retrain on current labels**
```bash
python scripts/03b_train_experiment_matrix.py \
    --config configs/yolo/experiments_joint_v2_ramp.yaml \
    --experiment R0 \
    --data data/yolo/naip/data.yaml
```
Goal: confirm harness is healthy and establish SAHI F1 baseline on the current (relabeled) val set.

**Step 2 — SAHI threshold sweep on R0 weights**
```bash
python scripts/05d_sahi_threshold_sweep.py \
    --weights runs/segment/<R0_run>/weights/best.pt \
    --run-name R0_<timestamp>
```
Target ≥ 0.45 SAHI F1 to unlock the Duke ramp. If below, do another relabel batch (see PHASE1_HANDOFF.md) and retrain before advancing.

**Step 3 — Duke ramp (R1+), only after R0 ≥ 0.45**
```bash
python scripts/02g_build_joint_v2_lists.py --naip-repeat 580
python scripts/03b_train_experiment_matrix.py \
    --config configs/yolo/experiments_joint_v2_ramp.yaml \
    --experiment R1
python scripts/05e_ramp_eval.py --run R1 \
    --weights runs/segment/<R1_run>/weights/best.pt
```
`05e_ramp_eval.py` will print **HALT** if NAIP test mAP50 drops > 0.07 vs the 0.563 baseline — stop the ramp and treat the previous step's weights as the candidate.

**Step 4 — Update registry when any new weights beat `r2-cameron-20260509`**

Add an entry to [models/registry.yaml](../models/registry.yaml) and bump `aliases.production`. Include `sahi_f1_val` and `sahi_conf_val` fields so the CLI manifest is accurate.

---

## 3. Stage 2 Next Steps

**Clear the 0.70 AUC gate.** Current blockers:
- AQ features are sparse pre-2022 (PM columns dropped at training)
- N ≈ 640 training rows — limited by NREL station count

Options in rough priority order:
1. Add more NREL stations if available (check NREL's API for stations not yet ingested by `scripts/12_ingest_nrel_soiling_map.py`).
2. Engineer lagged weather features (rolling 30/90-day PM, precipitation deficit).
3. Tune XGBoost regularization (current run uses defaults; `max_depth`, `min_child_weight`, `colsample_bytree` are the highest-leverage knobs).
4. Re-run `scripts/10_train_soiling_model.py --holdout-year 2023` once 2023 NREL data is available.

After any new run, verify calibration still holds:
```bash
PYTHONPATH=. python scripts/14_compare_soiling_runs.py
```

---

## 4. Product / CLI Next Steps

**Immediate (no model work required):**
- Add `LICENSE` (Apache 2.0) to the `SolarSoiled` public repo — not yet present.
- Tier 2 remaining: named-scene resolution + AOI overlap detection in `src/solarsoiled/aoi.py`.

**After R0 lands:**
- Tier 3: Dockerfile (`docker run solarsoiled:latest run --aoi <bbox>`).
- Tier 3: CI contract test on fixture AOI — the skip-marked harness at `tests/test_smoke_run.py` is the building block; flips on once a `smoketest` registry entry + `SOLARSOILED_SMOKE_TILES` env var are present.
- Tier 3: `examples/partner_engagement/` worked example.

**Track C — Beta API (does not wait on Stage 1 GA):**
`/detect`, `/risk`, `/recommend`, `/health` endpoints with auth + metering. See [Q2_PLAN.md](Q2_PLAN.md) Track C. The Stage 1 → Stage 2 contract is stable enough to scaffold against today.

---

## 5. Commercialization Flag

**Before charging customers, resolve the Ultralytics license.**

Ultralytics YOLO is AGPL-3.0. Commercial use requires either:
- (a) Negotiate an Ultralytics startup/commercial license (contact via ultralytics.com), or
- (b) Swap the detection backbone to an Apache-2.0 alternative (RT-DETR or Detectron2).

The rest of the pipeline — CLI, Stage 2, manifest system — is backbone-agnostic. Only these scripts touch Ultralytics directly: `scripts/03_train_yolov8_seg.py`, `scripts/03b_train_experiment_matrix.py`, `scripts/04_infer_yolov8_seg.py`, `scripts/05b_eval_threshold_sweep.py`, `scripts/05c_per_detection_rca.py`, `scripts/05d_sahi_threshold_sweep.py`, `scripts/05e_ramp_eval.py`.

---

## 6. Key Contacts and Context

- **Cameron** — project lead, geospatial + ML architecture, graduating May 2026.
- **Josh** — Stage 1 training cuts; owns the Colab GPU compute for R-series runs.
- **Tyler** — oversight; redirected project to diagnose-first after 2026-05-04 meeting.

The strategic pivot (2026-05-04) away from joint training toward diagnose-first + iterative NAIP relabel is documented in [PHASE1_HANDOFF.md](PHASE1_HANDOFF.md). The rationale (NAIP whole-array vs Duke per-panel label convention mismatch, alone-tile FP pattern) is in [Q2_PLAN.md](Q2_PLAN.md) under "What we learned."

---

## 7. Where Things Live

| What | Where |
|---|---|
| Active Stage 1 runbook | [docs/PHASE1_HANDOFF.md](PHASE1_HANDOFF.md) |
| Quarter priorities | [docs/Q2_PLAN.md](Q2_PLAN.md) |
| Stage 2 guide | [docs/SOILING_STAGE2_GUIDE.md](SOILING_STAGE2_GUIDE.md) |
| Model registry | [models/registry.yaml](../models/registry.yaml) |
| CLI entrypoint | `src/solarsoiled/cli.py` |
| Eval harness | `scripts/05c_per_detection_rca.py`, `scripts/05d_sahi_threshold_sweep.py`, `scripts/05e_ramp_eval.py` |
| Per-detection RCA output | `outputs/eval/<run_name>/per_detection.csv` |
| Label viz overlays | `outputs/label_viz/<run_name>/` |
| Soiling outputs | `runs/soiling/<run>/`, `outputs/soiling/training_matrix.parquet` |
| Public repo | `../solar-soiling-ml-public/` |
