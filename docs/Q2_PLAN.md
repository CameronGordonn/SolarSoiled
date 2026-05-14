# Q2 Plan: Current Alignment

Canonical project-status doc. The active Stage 1 runbook lives in [docs/PHASE1_HANDOFF.md](PHASE1_HANDOFF.md) (diagnose-first + R0 retrain, week of 2026-05-04). Joint training is paused; v1/v2 forensic detail (29:1 ratio, optimizer=auto bug, RandAugment starburst) lives in commit `2424b4a` and the comments in `configs/yolo/experiments_joint_v2.yaml` for whenever ramp work resumes. Stage 2 in [docs/SOILING_STAGE2_GUIDE.md](SOILING_STAGE2_GUIDE.md). Other docs link here for live status; they do not restate it.

**Status flags used throughout (only these four):** `shipped` (delivered, in production use), `in progress` (active work), `queued` (planned, not started), `beta` (live but below GA quality bar).

## Project status

### Quarter phases — model + product milestones

| Phase | Goal | Done looks like | Status | Current |
|---|---|---|---|---|
| **Phase 1 — Panel detection** | Restore the detector without losing the NAIP target domain | NAIP test mAP50 ≥ 0.70, Duke test mAP50 > 0; threshold sweep saved; baseline-safe checkpoint preserved | `in progress`, ships `beta` | 0.563 NAIP-only baseline; ~0.65 with SAHI warm start (calibrated F1=0.536 at conf=0.30/iou=0.50); joint-training paused after over-prediction failure (mAP50 0.128 / R 0.76 / P <0.1, 2026-05-03). Active path: diagnose-first RCA harness landed; R0 retrain from SAHI warm-start + iterative NAIP relabel ([PHASE1_HANDOFF.md](PHASE1_HANDOFF.md)) |
| **Phase 2 — Soiling-risk model** | Honest leakage-free metrics on the patched pipeline | Spatial-CV AUC ≥ 0.70 **and** year-holdout AUC ≥ 0.70 with calibration retained | `in progress`, ships `beta` | 0.63 spatial-CV / 0.66 holdout-2022 (`run_k_patched_holdout2022`) |
| **Phase 3 — Product surface** | End-to-end AOI pipeline a partner can run | One CLI entrypoint; per-AOI namespace; manifest-versioned outputs; model registry; Stage1 → Stage2 contract test | `in progress` | CLI + manifests + registry `shipped` (Tiers 0–1, partial Tier 2); Tier 2 named-scene resolution + AOI overlap detection `in progress`; Tier 3 partial: `solarsoiled eval --report` HTML `shipped` (`src/solarsoiled/eval_report.py`); Dockerfile + partner example + Stage1 → Stage2 CI contract test `queued` behind fresh Phase 1 weights |

Phase 1 gates Phase 3 (product needs a trustworthy detector). Phase 2 runs in parallel; stays `beta` until both AUC gates clear. Phase 3 plumbing consumes versioned outputs and quality metadata rather than hides model limitations.

### Customer-readiness tiers — CLI / API surface

| Tier | Scope | Status |
|---|---|---|
| **Tier 0** — output manifests + dependency hygiene | `pyproject.toml` package; `manifest.json` from every artifact-producing script | `shipped` |
| **Tier 1** — `solarsoiled` CLI | `tile / detect / score / recommend / run / eval` subcommands; per-AOI output namespace | `shipped` |
| **Tier 2** — model registry + AOI primitive | `models/registry.yaml` resolving `production` / `latest` / aliases; AOI WGS84 + CRS + validity checks | `in progress` (registry + AOI hardening live; named-scene resolution + AOI-overlap detection `queued`) |
| **Tier 3** — partner UX polish | Dockerfile; `solarsoiled eval --report` HTML; partner example; Stage1 → Stage2 CI contract test | `in progress` (`eval --report` `shipped`; Dockerfile + partner example + CI contract test `queued`) |

### Parallel tracks — Track C does not wait on Track A

| Track | Scope | Ship criterion | Status |
|---|---|---|---|
| **Track A — model quality** | Stage 1 R0 retrain (SAHI-warm-start) + iterative NAIP relabel, Duke ramp held until R0 lands, Stage 2 year-holdout validation | Phase 1 + Phase 2 GA gates clear | `in progress` |
| **Track B — visibility surface** | `solarsoiled-landing/` acquisition page, waitlist, design-partner pitch | Live site with waitlist capture | `in progress` |
| **Track C — beta API** | `/detect`, `/risk`, `/recommend`, `/health` endpoints with quality metadata + auth + metering | Reachable beta endpoints behind an API key | `queued` |

The Stage 1 base weight today is the SAHI warm-start checkpoint (~0.65 NAIP test on `model.val()` at conf=0.10). After the 2026-05-03 joint over-prediction failure (mAP50 0.128 / R 0.76 / P <0.1) and Tyler's 2026-05-04 meeting, joint training is paused. The active path is now: (1) diagnose-first RCA on the SAHI baseline (landed: per-detection CSV + bucket overlays exposed 65 alone-tile FPs concentrated on 20 tiles, mostly likely under-labeled); (2) Josh trains R0 from scratch on Santa Cruz NAIP and iteratively relabels in Roboflow; (3) Duke ramp (R1+) only after R0 reproduces ≥0.55 baseline on patched labels.

## Phase 1 Summary

### What we are doing

Stage 1 is YOLOv11 instance segmentation for solar arrays. The contract is simple: preserve CRS and affine metadata end to end so polygons can be exported back to world coordinates.

The pre-train-then-fine-tune plan failed (Duke-only pre-training transferred at 0.028 mAP50). Joint training failed twice: v1 collapsed both domains (NAIP 0.275 / Duke 0.010), and the 2026-05-03 cut hit mAP50 0.128 / R 0.76 / P <0.1 — over-prediction, the model hallucinating panels everywhere. Tyler's 2026-05-04 meeting redirected to **diagnose-first**: classify every TP/FN/FP on the existing SAHI baseline, find the failure-mode pattern, then retrain.

The diagnostic surfaced two compounding issues. (1) NAIP labels were drawn at 60 cm source resolution and miss small panels — the alone-tile FPs (65 of 360 total FPs concentrated on 20 GT-empty tiles, max 13 detections at conf 0.5 on `tile_000150`) are most likely real arrays the model found but our labels never recorded. (2) NAIP and Duke encode arrays differently — NAIP labels whole-array polygons (median 24 m²), Duke labels per-panel (median 1.7 m², KS=0.895 on area_m²). Joint training tries to bridge two label conventions. Active strategy: retrain R0 on Santa Cruz NAIP warm-started from `sahi_baseline_train7.pt` (preserves the small-panel prior 60 cm hand labels can't teach), iterate label fixes in Roboflow (the model surfaces label gaps via `05c` + `18`), only then ramp Duke (`02g_build_joint_v2_lists.py --naip-repeat <N>`) with hard regression stop-rule (`05e_ramp_eval.py` halts if NAIP test mAP50 drops by >0.07 vs 0.563).

### Datasets in scope

| Dataset | What it is | Why it matters | Status |
|---|---|---|---|
| NAIP Santa Cruz | 249 tiles, 174 train / 37 val / 38 test, about 360 arrays, 0.6 m GSD, YOLO polygons | This is the target domain and the current baseline source | `in progress` |
| Duke / Bradbury 160 px | 601 source images, 19,433 arrays, 2014-2015 NAIP vintage, GeoJSON converted to YOLO polygons, tiled to 160 px | Adds small-array signal that NAIP lacks | `in progress` (post-vintage-fix re-download) |
| NAIP San Jose | Older notes mention a separate ~100-tile NAIP set | Could help with small-array recall, but I have not confirmed the canonical files in the workspace | `queued` (canonical files not confirmed) |
| Connecticut Solar PV | 87 tiles, 1,611 arrays, 30 cm semantic masks | Useful diversity, lower priority than the current joint run | `queued` |
| BDAPPV | About 13k French aerial installations | Robustness data, later phase only | `queued` |

Active today in the YOLO path: `data/yolo/naip`, `data/yolo/duke_160`, and the generated `data/yolo/joint_v2` lists.

### Pipeline

1. NAIP tiles are created as 640x640 PNGs by `scripts/02_tile_naip_image.py`, recorded in `data/interim/tile_index.json`, and labeled via Roboflow back into `data/yolo/naip/`.
2. Duke imagery is downloaded by `scripts/02e_download_duke_dataset.py` using the 2014-2015 vintage filter, converted by `scripts/02d_convert_duke_dataset.py` into 160 px YOLO chips, and cleaned by `scripts/02f_clean_duke_dataset.py`.
3. `scripts/01_audit_dataset.py`, `scripts/labeling/validate_labels.py`, and the visual inspection gates catch label/geometry problems before any training.
4. `scripts/02g_build_joint_v2_lists.py` creates the current mix: Duke once, NAIP repeated 29x, with NAIP-only validation so early stopping follows the target domain.
5. `scripts/03_train_yolov8_seg.py` trains from YAML configs, `scripts/05b_eval_threshold_sweep.py` calibrates NAIP thresholds, `scripts/04_infer_yolov8_seg.py` runs inference, and `scripts/06_export_polygons_geojson.py` exports georeferenced polygons.

### Results so far

| Run | Setup | Result | Meaning |
|---|---|---|---|
| Baseline | NAIP Santa Cruz only | 0.563 test mAP50, precision about 0.53, recall about 0.56 | Production-safe Stage 1 baseline |
| SAHI warm start | Baseline weights with overlapping SAHI inference | about 65% on test | Current base weight for the oversampled Duke run |
| SAHI calibration | Best baseline weights with threshold sweep | 0.687 calibrated val mAP50 at conf 0.10 / iou 0.50, F1 0.653 | Shows the baseline can improve without retraining |
| Duke pre-train | Old Duke-only 320 px path | 0.028 transferred mAP50 | Abandoned |
| NAIP fine-tune after Duke pre-train | Fine-tune on NAIP after Duke-only pre-train | 0.424 | Worse than baseline |
| Joint v1 | Duke 160 px + NAIP, uniform sampling | NAIP 0.275, Duke 0.010 | Regression on both domains |
| Joint 2026-05-03 | Duke 160 px + NAIP, --naip-repeat 29, optimizer=auto bug fixed | NAIP test mAP50 0.128, P <0.1, R 0.76 | Over-prediction — model hallucinates panels everywhere |
| RCA on SAHI baseline | per_detection.csv on val+test at conf=0.05 | TP=125 FP=360 FN=101 (P=0.26 R=0.55); 65 of 360 FPs on 20 GT-empty tiles | Most alone-tile FPs are likely under-labeled real panels, not hallucinations |

### What we learned

Joint v1 failed because Duke dominated gradients. Joint v2 with the optimizer bug fixed still over-predicted because the per-panel Duke label convention pulled the prior toward dense detections, and NAIP's whole-array labels don't supply the negative signal at panel scale. Adding hard negatives is off the table — those "negatives" likely contain real panels the labels missed. The path forward is R0 retrain on patched NAIP labels (warm-started from the SAHI baseline so the small-panel detection prior survives), then ramp Duke only once R0 reproduces baseline. Until R0 lands, the 0.563 NAIP-only checkpoint remains production-safe; calibrated SAHI operating point is `conf=0.30, iou=0.50` (F1=0.536, P=0.703, R=0.433 from the 2026-05-04 sweep).

## Quarter dependencies

- Phase 1 gates Phase 3 because product outputs need a trustworthy detector.
- Phase 2 can proceed once Phase 1 outputs are frozen enough to build feature matrices, but it should remain beta until it clears GA.
- Phase 3 should consume versioned outputs and model-quality metadata instead of hiding current limitations.
