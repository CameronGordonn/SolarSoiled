# Quarter Plan Overview

One-page elevator summary. **Live status table is in [docs/Q2_PLAN.md](Q2_PLAN.md) — read that for current numbers.**

**Goal:** an end-to-end pipeline from panel detection on NAIP imagery to array-level soiling-risk flagging, with honest quality metadata so design partners can integrate while the model improves.

**Sequencing:** the three quarter phases overlap. Phase 2 already runs in `beta` against Phase 1's current detector, and Phase 3 plumbing (manifests, CLI, registry) is in flight. GA on each phase is gated on its own quality bar, not on calendar weeks. This mirrors the Track A / B / C structure in [PRODUCT_VISION.md](PRODUCT_VISION.md).

| Phase | One-line description | GA gate |
|---|---|---|
| **Phase 1 — Panel detection** | YOLOv11 segmentation of solar arrays in NAIP (Santa Cruz baseline + Duke 160 px small-array signal). Active path: diagnose-first RCA + iterative NAIP relabel + R0 retrain (warm-started from the SAHI baseline). Runbook: [PHASE1_HANDOFF.md](PHASE1_HANDOFF.md). | NAIP test mAP50 ≥ 0.70 with non-zero Duke test mAP50; threshold sweep saved; baseline checkpoint preserved. |
| **Phase 2 — Soiling risk** | Per-array soiling score from weather + air quality + location + structural features. XGBoost on NREL panel labels with isotonic calibration; spatial CV at 10 km cluster bins. Runbook: [SOILING_STAGE2_GUIDE.md](SOILING_STAGE2_GUIDE.md). | Spatial-CV AUC ≥ 0.70 **and** year-holdout AUC ≥ 0.70 with calibration retained. |
| **Phase 3 — Product surface** | One pipeline a partner can run on a fresh AOI: detect → score → recommend, with versioned outputs and one entrypoint (`solarsoiled run --aoi <geojson>`). | Partner runs the full pipeline and receives `{arrays.geojson, risk.geojson, recommendations.json, manifest.json}` with model versions, accuracy, beta flags. Stage1 → Stage2 contract test green in CI. |

For current numbers per phase, customer-readiness Tier status, and Track A/B/C status, see the project-status tables in [Q2_PLAN.md](Q2_PLAN.md). Status flags throughout the docs are: `shipped` / `in progress` / `queued` / `beta`.
