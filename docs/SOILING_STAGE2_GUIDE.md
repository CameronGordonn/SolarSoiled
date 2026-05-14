# Stage 2 — Soiling Risk Model (XGBoost) Partner Guide

**Goal:** per-array soiling risk score in `[0, 1]` from weather + location + structural features. Complements Stage 1 detection: Stage 1 tells us *where* arrays are, Stage 2 tells us *how likely each is soiled right now* without needing visual soiling detection (too coarse at 0.6 m GSD).

**Status:** v2 pipeline — real NREL annual-IWSR panel labels (255 stations × up to ~15 years of per-year observations), physics-prior Kimber IWSR as a feature, ESA WorldCover land-cover + OSM distance features pre-computed per station, isotonic-calibrated XGBoost with spatial CV and optional temporal holdout. The Kimber physics-proxy path remains available as a bootstrap/ablation.

**Honest current baseline (run_k_patched_holdout2022, 2026-04-24):** mean spatial-CV AUC **0.630**, holdout-2022 AUC **0.655** on 641 training rows / 76 holdout rows. Below the 0.70 acceptance gate; ships as **beta** with explicit disclosure. Prior runs reported ~0.668 mean CV — that number was inflated by per-fold imputation leakage (medians computed on the full matrix before splitting) and is no longer the production metric. The patched pipeline now learns medians on training data only, persists them to `feature_medians.json` for inference, drops features below 20% coverage, and clamps Open-Meteo AQ fetches to the CAMS start date (2013-01-01).

---

## Quick run (end-to-end)

```bash
conda activate solar-soiling
pip install -r requirements.txt          # requests-cache, pystac-client, planetary-computer, xgboost, pyarrow, joblib

# Unit tests (offline, no API calls)
PYTHONPATH=. pytest tests/test_weather_client.py tests/test_feature_engineering.py -v

# One-time data prep
PYTHONPATH=. python scripts/12_ingest_nrel_soiling_map.py   # NREL JSON → CSV + annual panel CSV
PYTHONPATH=. python scripts/13_build_static_features.py     # elevation + WorldCover + OSM (one-time, ~15–20 min for 255 stations)

# Train (default: nrel_panel labels, Kimber feature on, regression or binary via model.yaml)
PYTHONPATH=. python scripts/10_train_soiling_model.py --run-name run_latest

# Temporal holdout variant (train on pre-2022, test on 2022)
PYTHONPATH=. python scripts/10_train_soiling_model.py --run-name run_holdout2022 --holdout-year 2022

# Compare any saved runs
PYTHONPATH=. python scripts/14_compare_soiling_runs.py

# Ablation: synthetic Kimber-only labels (edit configs/soiling/california.yaml → label_source: kimber_proxy)

# Build features for detected arrays (requires scripts/07)
PYTHONPATH=. python scripts/09_build_soiling_features.py --arrays outputs/array_features.geo.parquet

# Score arrays (automatically applies the saved isotonic calibrator if present)
PYTHONPATH=. python scripts/11_predict_soiling_risk.py --model runs/soiling/<run_name>/model.ubj
```

Output: `outputs/soiling_risk.geojson` — each detected array polygon gets a calibrated `risk_score` column in `[0, 1]`.

---

## Architecture

```
Open-Meteo (ERA5 + CAMS)          array_features.geo.parquet
         │                                     │
         ▼                                     ▼
weather_client.py               location_features.py
         │                                     │
         └──────────────┬──────────────────────┘
                        ▼
              feature_engineering.py   (rolling 7/30/90d aggregates)
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
  labels.py (Kimber OR NREL)   inference matrix (per-array features)
         │                             │
         ▼                             ▼
  risk_model.py (XGBoost + spatial CV) → runs/soiling/<run>/model.ubj
         │                             │
         └──────────────┬──────────────┘
                        ▼
               soiling_risk.geojson
```

---

## Label source

Controlled by `configs/soiling/california.yaml:label_source`:

- **`nrel_panel`** (default): the per-(station, year) NREL panel-label CSV used by the current config. This is the preferred path for Stage 2 because it aligns weather windows to the actual observation year.

- **`nrel_soiling_map`**: ingests the NREL PV Soiling Map (Micheli/Deceglie/Muller, 255 US stations). Request from NREL at https://www.nrel.gov/pv/soiling or extract from the Micheli 2019 *Progress in PV* supplementary. This is the summary CSV path when you do not need the panel-year breakdown.

- **`kimber_proxy`**: generates IWSR labels from historical weather using the Kimber 2007 model (daily dust deposition ∝ PM2.5, reset by any day with precipitation ≥ 1 mm). Use this only as a bootstrap fallback when you want the pipeline to train without external label files. Results are *physically plausible but uncalibrated* — treat the model as a ranking tool, not an absolute loss estimator.

  **Required CSV schema** for the summary NREL CSV at `data/external/nrel_soiling_map.csv` (extra columns ignored):

  | column | type | notes |
  |--------|------|-------|
  | `station_id` | str | unique per row |
  | `latitude` | float | EPSG:4326 decimal degrees |
  | `longitude` | float | EPSG:4326 decimal degrees |
  | `iwsr` | float | insolation-weighted soiling ratio; 1.0 = clean, 0.95 = 5% annual loss |

  Optional: `soiling_rate_pct_per_day`, `start_date`, `end_date`. Once the file is in place, flip `label_source: nrel_soiling_map` in [configs/soiling/california.yaml](../configs/soiling/california.yaml) and re-run script 10.

Switching sources is a one-line config change; nothing else in the pipeline knows or cares which labels were used.

---

## Features

| Group | Columns | Source |
|-------|---------|--------|
| Weather (rolling 7/30/90d) | `precip_*d_mm`, `dry_day_streak`, `days_since_rain_5mm`, `wind_speed_10m_max_*d_mean`, `relative_humidity_2m_mean_*d_mean`, `temperature_2m_max_*d_mean` | Open-Meteo ERA5 archive |
| Air quality (rolling 7/30/90d) | `pm2_5_*d_mean`, `pm10_*d_mean` | Open-Meteo CAMS |
| Physics prior | `kimber_iwsr_proxy`, `kimber_iwsr_7d_mean`, `kimber_iwsr_30d_mean`, `kimber_iwsr_90d_mean` | Kimber 2007 IWSR computed on the same daily weather stream |
| Static location | `elevation_m`, `nlcd_class` (WorldCover), `worldcover_cropland`, `worldcover_built_up`, `worldcover_bare`, `worldcover_tree`, `worldcover_grass`, `distance_to_highway_m`, `distance_to_agriculture_m` | Open-Meteo elevation, ESA WorldCover 2021 (Planetary Computer), OSM Overpass — pre-computed by [scripts/13_build_static_features.py](../scripts/13_build_static_features.py) |
| NREL covariates | `tilt_deg`, `months_in_data_set`, `mounting_Fixed`, `mounting_Tracking`, `measurement_type_PV System`, `measurement_type_Soiling Station` | NREL soiling-map CSV |
| Geometric (inference only) | `area_m2`, `orientation_deg`, `compactness`, `neighbor_count_100m`, `distance_to_nearest_array_m` | [scripts/07_extract_array_features.py](../scripts/07_extract_array_features.py) |

All Open-Meteo calls pass through `requests-cache` — first request for a given (lat, lon, window) hits the API, subsequent requests hit `.cache/soiling/openmeteo.sqlite`. Open-Meteo 429s trigger exponential-backoff retries in `src/soiling/weather_client.py` so large panel runs don't silently drop rows. Overpass queries are cached separately in `.cache/soiling/overpass.sqlite`.

## Panel labels (station × year)

`scripts/12_ingest_nrel_soiling_map.py` emits two CSVs from the NREL JSON:

- `data/external/nrel_soiling_map.csv` — one row per station, with a single summary IWSR. Historical/v1 path.
- `data/external/nrel_soiling_map_annual.csv` — one row per (station, year) from the NREL `Annual IWSR` block. **This is the default training source**; each row is matched to a year-specific weather window (`as_of = Dec 31`, `lookback = 365 d`) instead of a single today-centered snapshot. Yields ~891 rows across 15 years.

Switch between them via `label_source: nrel_panel | nrel_soiling_map | kimber_proxy` in `configs/soiling/california.yaml`.

## Sample weights, calibration, and temporal holdout

- **Sample weights** (`model.yaml:sample_weights.mode`): `iwsr_ci_width` weights each training row by `1 / (iwsr_upper − iwsr_lower)`, clipped to the 5–95th percentile. Stations with tighter NREL confidence intervals pull the model harder.
- **Isotonic calibration**: `train_risk_model` collects out-of-fold predictions during spatial CV, fits `sklearn.isotonic.IsotonicRegression`, and saves the result to `runs/soiling/<run>/calibrator.joblib`. Training also persists `feature_medians.json` so inference uses the same NaN-imputation values learned on train/CV instead of recomputing medians on the scored batch.
- **Temporal holdout**: pass `--holdout-year 2022` (panel mode only). Trains on all other years, evaluates separately on the held-out year. Use to catch year-to-year drift that spatial CV alone misses.
- **Regression head** (`model.yaml:target_mode: regression`): fits `reg:squarederror` on continuous IWSR instead of the thresholded binary label. Reports Spearman rank correlation alongside AUC. Binary is the more robust head at N ≲ 200; regression is preferred once the panel path is fully warm (N ≈ 900).

---

## Spatial cross-validation

Stations are clustered by lat/lon into ~10 km buckets and held out as whole groups (`GroupKFold`). Random KFold would leak neighboring stations between folds and inflate AUC by 5–15 points. Acceptance gate (in `configs/soiling/model.yaml`): mean CV AUC ≥ 0.70.

**Why 10 km, not 50 km:** at 50 km, the LA / Inland Empire region (~40 stations) bucketed into a single cluster that took 37% of the training data into one fold. That fold's near-random AUC dragged down the mean while the other four folds (multi-cluster, multi-region) reported 0.62–0.75 — and conversely the mean was *inflated* relative to a balanced split by the asymmetric averaging. Dropping to 10 km splits LA into ~12 contiguous bins that GroupKFold spreads across all five folds (159 rows each), producing the honest 0.63 mean. See `scripts/research/diag_soiling_folds.py` for the per-fold composition reporter used to diagnose this.

---

## Sanity check on California outputs

After running end-to-end, the GeoJSON should show:
- Higher `risk_score` on arrays in the **Central Valley** (Fresno/Bakersfield — high PM2.5, long dry season, ag dust)
- Lower `risk_score` on arrays in **coastal Santa Cruz / SF Bay** (marine air + frequent rain)
- Highest scores in **the desert southeast** (Palm Springs / Barstow — bone dry + dust events)

If the map doesn't look like that, something is wrong with the feature stream (most likely the AQ fetch silently failing).

---

## Known gaps (deferred to v3)

- **PVDAQ label source**: NREL PVDAQ has ~44 GB of per-system performance time-series that could yield thousands of additional IWSR labels via the SRR extraction method. Path to the largest accuracy win; high engineering cost.
- **Dust-specific CAMS variables + wind direction**: CAMS has a `dust` AOD channel separate from PM; prevailing wind direction vs. nearby source matters for desert stations. Small to medium lift.
- **Explainability**: `predict_risk` returns a scalar. For per-array SHAP attributions, wrap with `shap.TreeExplainer(model)` in script 11 and add a `top_features` column.
- **Live scoring**: pipeline is batch-only. For periodic rescoring, schedule script 09 + 11 via cron — Open-Meteo cache expires every 30 days by default.
- **Multi-region**: `configs/soiling/california.yaml` is the only region config. The NREL panel data already covers 15 states; to train a region-specific model copy the file and set `bbox`.

---

## Troubleshooting

**`requests-cache` not installed:** `pip install requests-cache` (added to requirements.txt).

**Kimber bootstrap labels all zero (or all one):** the risk threshold (`iwsr_risk_threshold: 0.97`) may not match the seed stations' climate. Lower to 0.99 for more positives, raise to 0.95 for fewer. Check `outputs/soiling/training_matrix.parquet` to see the IWSR distribution.

**Single-class CV fold warnings:** means at least one spatial cluster has all-positive or all-negative labels. Either expand the seed station list in [scripts/10_train_soiling_model.py](../scripts/10_train_soiling_model.py) `DEFAULT_CA_STATIONS` or lower `cluster_km` in `configs/soiling/model.yaml`.

**`array_features.geo.parquet` missing:** run the full Stage 1 pipeline (`scripts/04` → `scripts/06` → `scripts/07`) first.

**Open-Meteo rate limits:** free tier is 10k calls/day. California has O(10²) stations + whatever detected arrays you have, well under the limit. If you hit rate limits during bulk backfill, the cache will serve repeats for free.

**AQ features mostly null:** if the training log reports PM feature coverage well below 50%, treat it as an upstream historical-AQ problem, not evidence that PM has no predictive value. The trainer now warns and drops extremely sparse features (<20% coverage by default) so the saved model does not silently depend on a broken feed.
