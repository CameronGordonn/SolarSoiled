# SolarSoiled — Partner Quickstart

This guide walks you through running SolarSoiled end-to-end on your first area of interest (AOI): detect solar arrays from public aerial imagery, score each array for soiling risk, and get a cleaning recommendation.

**Time to first result: ~15 minutes** (mostly waiting on NAIP tile download + inference)

---

## Prerequisites

- Docker installed ([docs.docker.com/get-docker](https://docs.docker.com/get-docker/))
- An API key from the SolarSoiled team (email solarsoil.app@gmail.com)
- Model weights — download link provided with your API key

---

## Step 1 — Set up

```bash
# Clone the repo (or just copy this examples/ folder)
git clone https://github.com/camerongordon/solar-soiling-ml
cd solar-soiling-ml

# Copy the env template and fill in your key
cp .env.example .env
# Edit .env: set SOLARSOILED_API_KEY=<your-key>

# Build the Docker image (~5 min first time, cached after)
docker build -t solarsoiled .
```

Place the model weights you received in the `models/` directory:

```
models/
  r2-cameron-20260509.pt       # detection weights (provided)
  registry.yaml                # already in repo — maps names to files
runs/
  soiling/
    run_optionb/
      model.ubj                # soiling risk model (provided)
      calibrator.joblib        # probability calibrator (provided)
```

---

## Step 2 — Run your first AOI

Pick a named scene or provide your own bbox:

```bash
# Named scene (Santa Cruz, CA)
docker run --rm \
  --env-file .env \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/.cache:/app/.cache \
  -v $(pwd)/data/external:/app/data/external \
  solarsoiled run \
    --aoi santa_cruz \
    --weights production \
    --soiling-model soiling_production \
    --last-cleaned 2025-12-01 \
    --partner-id my-first-run

# Or use your own bbox: --aoi "-122.05,36.90,-121.85,37.05"
# Available named scenes: santa_cruz, fresno, bakersfield, phoenix, tucson,
#   las_vegas, palm_springs, san_diego, los_angeles, sacramento, san_jose,
#   el_paso, austin, albuquerque
```

The pipeline runs four stages in sequence:

| Stage | What happens | Output |
|---|---|---|
| `tile` | Downloads NAIP imagery for your AOI and slices into 640×640 tiles | `outputs/aoi/<id>/tiles/` |
| `detect` | Runs YOLOv11 to find every solar array in the tiles | `outputs/aoi/<id>/arrays.geojson` |
| `score` | Scores each array's soiling risk (0–1) using weather + environment | `outputs/aoi/<id>/risk.geojson` |
| `recommend` | Computes cleaning recommendation based on risk + last-cleaned date | `outputs/aoi/<id>/recommendations.json` |

---

## Step 3 — View your results

```bash
# Open the interactive risk map in your browser
open outputs/aoi/my-first-run/risk_map.html

# Or inspect the raw GeoJSON
cat outputs/aoi/my-first-run/risk.geojson | python -m json.tool | head -60

# Per-array recommendations
cat outputs/aoi/my-first-run/array_recommendations.json
```

The risk map shows each detected array colored by soiling risk score (green → low, red → high). Click any array for its score, estimated soiling rate, and cleaning window.

---

## Step 4 — Re-run score + recommend only (skip re-detecting)

If you change the `--last-cleaned` date or want to re-score after a weather update, skip the slow tile/detect stages:

```bash
docker run --rm \
  --env-file .env \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/.cache:/app/.cache \
  solarsoiled run \
    --aoi santa_cruz \
    --weights production \
    --soiling-model soiling_production \
    --last-cleaned 2026-03-15 \
    --partner-id my-first-run \
    --skip-tile --skip-detect
```

---

## Step 5 — Submit cleaning feedback (closes the loop)

After you clean a site, submit the before/after energy output so it feeds back into the model:

```bash
curl -X POST http://localhost:8000/feedback \
  -H "X-API-Key: <your-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "array_id": 0,
    "partner_id": "my-first-run",
    "cleaned_at": "2026-04-01",
    "pre_clean_kwh_7d": 142.3,
    "post_clean_kwh_7d": 168.7,
    "notes": "Heavy dust buildup, near agricultural field"
  }'
```

Feedback is stored in `outputs/aoi/my-first-run/feedback.json` and is automatically picked up on the next model retrain, weighted 3× higher than NREL proxy labels.

---

## Running the API server (instead of CLI)

Start the API and submit jobs programmatically:

```bash
# Start the server
docker compose up api

# Submit a pipeline job
curl -X POST http://localhost:8000/jobs \
  -H "X-API-Key: <your-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "aoi": "santa_cruz",
    "weights": "production",
    "soiling_model": "soiling_production",
    "last_cleaned": "2025-12-01",
    "partner_id": "my-first-run"
  }'
# Returns: {"job_id": "...", "status": "queued"}

# Poll status
curl http://localhost:8000/jobs/<job_id> -H "X-API-Key: <your-key>"

# Stream live progress (SSE)
curl -N http://localhost:8000/jobs/<job_id>/events -H "X-API-Key: <your-key>"

# Fetch results when done
curl http://localhost:8000/results/my-first-run/recommendations -H "X-API-Key: <your-key>"
```

---

## Model quality (honest beta disclosure)

| Metric | Value | Gate |
|---|---|---|
| Detection (SAHI F1 on NAIP val) | 0.396 | ≥0.55 beta, ≥0.65 GA |
| Soiling risk — spatial CV AUC | 0.728 | ≥0.70 ✓ |
| Soiling risk — holdout 2022 AUC | 0.679 | ≥0.70 (in progress) |

Both models are in beta. Detection finds most large rooftop arrays but may miss small installations or produce false positives in dense vegetation. Soiling risk is calibrated and reliable for ranking arrays against each other; absolute scores should be interpreted as risk tiers (low/medium/high) rather than precise probabilities until the holdout gate clears.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `RegistryError: 'production' not found` | Check `models/r2-cameron-20260509.pt` exists |
| `RegistryError: 'soiling_production' not found` | Check `runs/soiling/run_optionb/model.ubj` exists |
| Detection finds 0 arrays | Try lowering `--risk-threshold 0.4` or check the AOI bbox is over a populated area |
| Score stage hangs (>5 min) | MERRA-2 AQ fetch is running; set `NASA_EARTHDATA_TOKEN=` to skip and use cached weather only |
| `AOIValidationError: Unknown named scene` | Use one of the listed scene names or provide a bbox directly |

---

## What's next

- Add more AOIs and compare risk scores across sites
- Set up a weekly cron job to re-score after weather events
- Submit cleaning feedback after each service visit to improve the model
- Contact solarsoil.app@gmail.com with questions or to report detection issues in your area
