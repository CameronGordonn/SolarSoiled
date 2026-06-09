# Public ⇄ Private sync & pre-publish safety gate

SolarSoiled lives in two repos:

| Repo | Role | Contains |
|---|---|---|
| **Private** (`solar-soiling-ml`) | Source of truth | Full ML pipeline, training data, model weights, internal runbooks, homeowner/parcel/outreach data, API keys |
| **Public** (`SolarSoiled`, this repo) | Public surface | Landing page, homeowner dashboard, curated README/architecture, a reference snapshot of the pipeline |

The public repo is **published to GitHub Pages**, so everything here is world-readable. Treat it that way.

## Pre-publish safety gate — run before pushing anything public

Nothing crosses private → public until it clears all of these:

- [ ] **No PII.** No real addresses, owner names, parcel data, emails, or phone numbers. (A homeowner-address file, `address_lookup.js`, leaked here once — removed and history-scrubbed 2026-06. Don't reintroduce per-array addresses; identify arrays by ID only.)
- [ ] **No secrets.** No API keys/tokens (`SOLARSOILED_API_KEY`, `NASA_EARTHDATA_TOKEN`, `LOB_API_KEY`, …). Only `*.example` placeholders. Confirm `.env`, `keys.yaml`, `.cache/`, `data/`, `runs/`, `outputs/`, `*.pt` stay gitignored.
- [ ] **No internal runbooks / live status docs.** CLAUDE.md, PHASE1_HANDOFF, Q2_PLAN, meeting notes stay private.
- [ ] **Metrics match private.** Headline numbers in `README.md` match the current private `CLAUDE.md` / `README.md` (Stage 1 box mAP50, Stage 2 spatial-CV / holdout AUC). Don't let them drift.
- [ ] **Client-side JS is clean.** `solarsoiled-landing/*.js` embeds no secret, no PII. The dashboard's `arrays_data.js` carries per-array lat/lon (inherent to a map) — acceptable, but do **not** pair it with street addresses.

## Quick checks

```bash
# PII / secrets sweep over tracked files
git ls-files | xargs grep -lEi 'api[_-]?key|secret|token|password|[0-9]{2,5} [A-Z][a-z]+ (Street|Avenue|Way|Drive|Court)' 2>/dev/null

# Confirm sensitive paths are NOT tracked
git ls-files | grep -E '(^|/)(\.env$|keys\.yaml|.*\.sqlite|.*\.pt$)' && echo "LEAK ⛔" || echo "clean ✓"
```

## Open follow-up

- **`arrays_data.js` lat/lon** is reverse-geocodable back to a home. Lower risk than explicit addresses, but consider coarsening/jittering coordinates or gating precise location behind the QR deep-link (`dashboard.html?id=<array_id>`) rather than embedding all arrays.
- **Backend URL / partner ID** are hardcoded in `dashboard.js`. Verify the API blocks cross-partner enumeration of `/results/{partner_id}/arrays` even without an API key.
