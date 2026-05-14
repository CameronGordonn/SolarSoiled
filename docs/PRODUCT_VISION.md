# SolarSoiled — Product Vision

North-star doc for where SolarSoiled is going, how the pieces fit together as a product, and which workstreams run in parallel. Complements the operational runbooks in `docs/`; does not duplicate them.

---

## Product thesis

SolarSoiled detects rooftop solar arrays from public aerial imagery, scores each array's soiling risk from environmental and structural context, and recommends when to clean. Revenue lives in the recommendation: detection is the funnel, risk is the hook, the cleaning window is what operators pay for.

---

## Current state

Live status lives in [docs/Q2_PLAN.md](Q2_PLAN.md) — see the "Project status" tables there. This table is a pointer to the surfaces; flag values mirror the canonical four (`shipped` / `in progress` / `queued` / `beta`).

| Surface | Pointer | Status |
|---|---|---|
| Stage 1 — detection | [docs/NAIP_ROBOFLOW_WORKFLOW.md](NAIP_ROBOFLOW_WORKFLOW.md), [docs/PHASE1_HANDOFF.md](PHASE1_HANDOFF.md) | `beta` — 56% mAP50 NAIP Santa Cruz; target 70%+ |
| Stage 2 — risk model | [docs/SOILING_STAGE2_GUIDE.md](SOILING_STAGE2_GUIDE.md) | `beta` — 0.63 spatial-CV / 0.66 holdout-2022 AUC (`run_k_patched_holdout2022`) |
| Customer-readiness CLI | `src/solarsoiled/cli.py`, [CLAUDE.md](../CLAUDE.md) | Tier 0 + Tier 1 `shipped`; Tier 2 `in progress`; Tier 3 `in progress` (`eval --report` `shipped`) |
| Landing page | `build_solarsoiled.py`, `solarsoiled-landing/` | `in progress` — live with beta-status section + design-partner program; not yet deployed |
| Operational runbook | [CLAUDE.md](../CLAUDE.md) | Source of truth for commands, paths, dataset status |

---

## Parallel tracks

Three workstreams run concurrently. **Track C does not wait on Track A** — we ship a beta API with honest quality metadata so design partners can integrate while the model improves.

| Track | Scope | Ship criterion | Rationale |
|---|---|---|---|
| **A — Model quality** | Stage 1 joint training (Duke 160px + NAIP), pseudo-labeling round, Stage 2 year-holdout validation | mAP50 ≥70% for Stage 1; year-holdout pass for Stage 2 | Sets the GA bar |
| **B — Visibility surface** | Rebuild `solarsoiled-landing/` into an acquisition page: waitlist, design-partner pitch, sample outputs, honest "beta" labeling | Live site with waitlist capture | Visibility and signal channel need to exist now |
| **C — Beta API** | Scaffold `/detect`, `/risk`, `/recommend`, `/health`; every response carries quality metadata; auth + metering stubs from day one | Reachable beta endpoints behind an API key | Design partners can integrate in parallel with model improvements |

### Beta/GA honesty model

- Every API response includes `{model_version, map50_current, beta: true | false, known_limitations: [...]}`.
- Landing page states current detection accuracy explicitly — a trust asset, not a liability.
- The `beta` flag flips to `false` when Track A clears its bars.
- Early users see real metrics, so their feedback is calibrated and their trust is earned.

---

## API surface (v0 beta contract)

Contract sketches, not final specs. Every response carries the metadata block described above.

### `POST /detect` — async job

- Input: `{aoi: bbox | scene_id, vintage: "2014-2015" | "latest"}`. Vintage is mandatory because Bradbury-derived training data is 2014–2015.
- Returns `202 Accepted` with a `job_id`.
- `GET /detect/jobs/{job_id}` returns status + result URL on completion.
- Result payload: GeoJSON of array polygons + metadata block.
- Async because county-scale detection is hours; sync is wrong by construction.

### `POST /risk` — sync

- Input: `{polygons: GeoJSON}` OR `{detect_job_id: str}` (avoid round-tripping large GeoJSON when we already have it).
- Returns per-array `{risk_score, model_version, features_version, scored_at, beta, known_limitations}`.
- Default returns the most recent daily batch score. `force_rescore=true` triggers a synchronous rescore (premium, metered).

### `POST /recommend` — sync, subscription-gated

- Input: polygons + `{last_cleaned: date, operator_constraints?: {...}}`.
- Returns `{window_start, window_end, expected_recovery_pct: [low, high], confidence, rule_fired, model_version, beta}`.
- v1 is rule-based (see next section). v2 is ML-based.

### `GET /health/live` and `GET /health/ready`

- `/health/live` — process is up.
- `/health/ready` — models and calibrators are loaded, external API keys (NREL, Open-Meteo) present, downstream dependencies reachable.

### Cross-cutting

- API-key auth from day one.
- Per-scan metering on `/detect` and `/risk`.
- `model_version` in every response (XGBoost + YOLO weights both tagged).

---

## Cleaning recommendation engine — staged

### v1 — rule-based

- Rule: `risk_score > T AND forecast_rain_7d < R_mm AND days_since_clean > D` → recommend window `[today+1, today+forecast_dry_stretch_end]`.
- Inputs available today: risk_score (Stage 2), Open-Meteo forecast (already integrated), `last_cleaned` from client.
- Output must include `confidence` (bucketed from risk_score) and `rule_fired` (which threshold dominated) — operators need to understand *why*.
- `expected_recovery_pct` is a **range**, not a point estimate. v1 doesn't have enough ground truth for a credible point.

### v2 — ML-based

- Optimize `expected_recovery_kWh − cleaning_cost` over a rolling calendar.
- Depends on the feedback loop below to train against.

---

## Track B — Landing page scope

- Hero with honest positioning: "detect + score + recommend, currently in beta".
- Waitlist / design-partner signup form. Email capture into Airtable, Sheets, or a minimal backend — whichever ships fastest.
- Sample outputs: one real GeoJSON render per county we've processed. Show the product, don't just describe it.
- "Current model quality" section stating the live mAP50 number.
- API beta signup CTA linking to the API-key request flow.
- `build_solarsoiled.py` stays for now. Migrate to Astro or a Next.js static export when form handling and analytics start to matter.

---

## Dashboard v1

Static Leaflet render of the pre-computed `outputs/soiling_risk.geojson` per county. AOI filter + per-array side panel showing features and the v1 cleaning recommendation. Live API-backed dashboard is v2 — wait until we have real user AOIs to justify the backend load.

---

## Data feedback loop

Operators cleaning panels and reporting post-clean output is the long-term moat.

- Schema: `{array_id, cleaned_at, pre_clean_kWh_7d, post_clean_kWh_7d, notes?}`.
- Submission endpoint is **free** — incentive to contribute.
- Each submission becomes a training row for the v2 recommendation model and a calibration check for Stage 2.

---

## Freshness policy

- Risk scored **daily** via a cron job over active AOIs.
- `/risk` returns the most recent batch score by default.
- `force_rescore=true` triggers a synchronous rescore, premium-tier, counts against metering quota.
- Every response includes `scored_at` so clients can judge staleness.

---

## Monetization — product-decision guidance

Not a commitment; shapes API and output design.

- `/detect` + `/risk`: per-scan metered. Free tier (e.g., N scans/month) for design partners.
- `/recommend`: subscription-gated — per-MW annual or per-site.
- `force_rescore`: premium add-on.
- Feedback submission: free, always.

---

## Customer-readiness arc (Tier 0–3)

The current code is research-shaped — 14 numbered scripts each with their own argparse and output convention. That's correct for "Cameron and Josh debug a model" and wrong for "we have a partner AOI on Friday and need full Stage 1 → Stage 2 in one command with versioned outputs and beta metadata." Below in priority order; tiers are largely independent and can run in parallel with model work.

- **Tier 0 — output manifest + dependency hygiene. `shipped`.** `pyproject.toml` registers the package; `src/solarsoiled/manifest.py` writes a sibling `manifest.json` from every artifact-producing script (02/04/05/06/09/10/11). Schema mirrors the v0 beta API response: `{schema_version, stage, model_version, model_weights_sha256, inputs_hash, generated_at, beta, metrics, known_limitations}`.
- **Tier 1 — `solarsoiled` CLI. `shipped`.** Typer entrypoint registered as a console script (`pip install -e .` exposes `solarsoiled`). Subcommands `tile / detect / score / recommend / run / eval`; `run --aoi <bbox-or-geojson>` chains all four. The 14 scripts stay and still work standalone; the CLI imports their `main(argv=…)` functions as library calls. Per-AOI namespace: `outputs/aoi/<partner_id>/{aoi.geojson, tiles/, detect/, arrays.geojson, features/, risk.geojson, recommendations.json, manifest.json}`. Recommend engine is rule-based v1 with injectable forecast for tests; covered by 13 unit tests across `tests/test_aoi.py` and `tests/test_recommend.py`. End-to-end smoke against a fresh Stage 1 checkpoint is `queued` for the next training cut.
- **Tier 2 — model registry + AOI primitive. `in progress`.** `models/registry.yaml` is the catalog; `src/solarsoiled/registry.py` resolves `--weights production`, `--weights stage1-v0.5-baseline`, etc. through it for `detect`, `run`, and `eval`. Filesystem paths still resolve as ad-hoc passthrough (`model_version="ad-hoc:<sha12>"`) so partners can point at an arbitrary `.pt` without first editing the registry. AOI primitive hardened with WGS84 lon/lat range checks, explicit non-WGS84 GeoJSON rejection, and shapely validity checks. Next: named-scene resolution and AOI overlap detection.
- **Tier 3 — partner UX polish. `in progress`.** `solarsoiled eval --report` **`shipped`** (`src/solarsoiled/eval_report.py`, 6 tests) — single-file HTML report (PR curve, F1-colored sweep table, failure-mode tables, base64-embedded overlay PNGs, per-tile worst-offenders, sibling `manifest.json`) produced from existing eval artifacts with no inference re-run. Invoke: `solarsoiled eval --weights <name> --report --report-dir outputs/eval/<run-name>`. Still queued: Dockerfile so a partner runs `docker run solarsoiled:latest run --aoi <bbox>`; `examples/partner_engagement/` worked example; Stage1 → Stage2 contract test on a fixture AOI in CI (the skip-marked harness at `tests/test_smoke_run.py` is the building block — flips on once a `smoketest` registry entry + `SOLARSOILED_SMOKE_TILES` env var are present).

---

## Non-functional principles

- County-agnostic by construction.
- Reproducible training (configs committed with results).
- Minimal external dependencies.
- Modular codebase (detection, features, modeling, API as separable concerns).
- **Honest by default** — beta flags and quality metadata in every response.
