"""Calibrate SOMOSclean (k, sl_sat) parameters against measured NREL IWSR.

Runs a grid search over k (time constant, days) and sl_sat (saturation ceiling)
to minimize MAE between SOMOSclean-computed IWSR and NREL measured IWSR.
Fits globally and per climate zone. Writes best-fit parameters back to
configs/soiling/features.yaml and reports before/after MAE.

Weather data is pulled from the Open-Meteo cache populated by the validation
run (scripts/15), so this runs quickly even over 800+ station-years.

Usage:
    PYTHONPATH=. python scripts/16_calibrate_somosclean.py
"""

from __future__ import annotations

import argparse
import itertools
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.soiling.labels import somosclean_eqd_trajectory
from src.soiling.weather_client import fetch_combined

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Grid to search — k in days, sl_sat as fraction (0–1).
K_GRID = [5, 10, 20, 30, 45, 60, 90]
SL_SAT_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

# Fixed parameters (not calibrated in this pass).
HEAVY_RAIN_MM = 10.0
RAIN_MIN_MM = 1.0
PM10_DUST_THRESHOLD = 50.0
PM10_DUST_SCALE = 0.02

LOOKBACK_DAYS = 365


def _climate_zone(lat: float, lon: float) -> str:
    if lon < -120.0:
        return "coastal"
    if lon > -116.0 or (lat < 34.5 and lon > -117.5):
        return "desert"
    return "valley"


def _somosclean_iwsr(daily: pd.DataFrame, k: float, sl_sat: float) -> float:
    """Terminal IWSR for one station-year using SOMOSclean."""
    _, sl_series = somosclean_eqd_trajectory(
        daily,
        sl_sat=sl_sat,
        k=k,
        heavy_rain_mm=HEAVY_RAIN_MM,
        rain_min_mm=RAIN_MIN_MM,
        pm10_dust_threshold=PM10_DUST_THRESHOLD,
        pm10_dust_scale=PM10_DUST_SCALE,
    )
    return float(1.0 - sl_series.iloc[-1])


def _grid_search(dailies: list[pd.DataFrame], measured: np.ndarray) -> tuple[float, float, float]:
    """Return (best_k, best_sl_sat, best_mae) over the parameter grid."""
    best_mae = float("inf")
    best_k, best_sl_sat = K_GRID[0], SL_SAT_GRID[0]
    for k, sl_sat in itertools.product(K_GRID, SL_SAT_GRID):
        preds = np.array([_somosclean_iwsr(d, k, sl_sat) for d in dailies])
        mae = float(np.mean(np.abs(preds - measured)))
        if mae < best_mae:
            best_mae, best_k, best_sl_sat = mae, k, sl_sat
    return best_k, best_sl_sat, best_mae


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--nrel-csv", default="data/external/nrel_soiling_map_annual.csv")
    p.add_argument("--config", default="configs/soiling/california.yaml")
    p.add_argument("--features-config", default="configs/soiling/features.yaml")
    p.add_argument("--no-write", action="store_true", help="Report results without updating features.yaml")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]

    nrel_path = repo_root / args.nrel_csv
    if not nrel_path.exists():
        raise FileNotFoundError(f"NREL annual CSV not found at {nrel_path}")

    region_cfg = yaml.safe_load((repo_root / args.config).read_text())
    feat_cfg = yaml.safe_load((repo_root / args.features_config).read_text())
    cache_dir = repo_root / region_cfg.get("cache_dir", ".cache/soiling")

    nrel = pd.read_csv(nrel_path)
    if "iwsr_censored" in nrel.columns:
        nrel = nrel[~nrel["iwsr_censored"].astype(bool)].reset_index(drop=True)
    nrel["climate_zone"] = nrel.apply(
        lambda r: _climate_zone(float(r["latitude"]), float(r["longitude"])), axis=1
    )

    # Fetch daily weather for every station-year (hits cache from validation run).
    logger.info("Fetching daily weather for %d station-years (using cache)...", len(nrel))
    dailies, measured, zones = [], [], []
    for _, s in nrel.iterrows():
        year = int(s["year"])
        as_of = date(year, 12, 31)
        start = as_of - timedelta(days=LOOKBACK_DAYS - 1)
        try:
            daily = fetch_combined(float(s["latitude"]), float(s["longitude"]), start, as_of, cache_dir=cache_dir)
        except Exception as exc:
            logger.warning("Skipping %s (%d): %s", s["station_id"], year, exc)
            continue
        dailies.append(daily)
        measured.append(float(s["iwsr"]))
        zones.append(s["climate_zone"])

    measured_arr = np.array(measured)
    logger.info("Loaded %d station-years for calibration", len(dailies))

    # Baseline MAE with current default parameters.
    current = feat_cfg.get("somosclean", {})
    default_k = float(current.get("k", 30.0))
    default_sl_sat = float(current.get("sl_sat", 0.25))
    default_preds = np.array([_somosclean_iwsr(d, default_k, default_sl_sat) for d in dailies])
    default_mae = float(np.mean(np.abs(default_preds - measured_arr)))
    logger.info("Baseline (k=%.0f, sl_sat=%.2f): MAE=%.4f", default_k, default_sl_sat, default_mae)

    # Global grid search.
    logger.info("Running global grid search (%d combinations)...", len(K_GRID) * len(SL_SAT_GRID))
    best_k, best_sl_sat, best_mae = _grid_search(dailies, measured_arr)
    logger.info(
        "Global best: k=%.0f  sl_sat=%.2f  MAE=%.4f  (improvement: %.4f → %.4f)",
        best_k, best_sl_sat, best_mae, default_mae, best_mae,
    )

    # Per-zone grid search.
    zone_results: dict[str, dict] = {}
    for zone in sorted(set(zones)):
        idx = [i for i, z in enumerate(zones) if z == zone]
        if len(idx) < 10:
            logger.warning("Zone %s has only %d rows — skipping per-zone fit", zone, len(idx))
            continue
        z_dailies = [dailies[i] for i in idx]
        z_measured = measured_arr[idx]
        zk, zsl, zmae = _grid_search(z_dailies, z_measured)
        logger.info("  %s (n=%d): k=%.0f  sl_sat=%.2f  MAE=%.4f", zone, len(idx), zk, zsl, zmae)
        zone_results[zone] = {"k": zk, "sl_sat": zsl, "mae": zmae, "n": len(idx)}

    # Write global best back to features.yaml.
    if not args.no_write:
        feat_cfg.setdefault("somosclean", {})
        feat_cfg["somosclean"]["k"] = float(best_k)
        feat_cfg["somosclean"]["sl_sat"] = float(best_sl_sat)
        out_path = repo_root / args.features_config
        out_path.write_text(yaml.dump(feat_cfg, default_flow_style=False, sort_keys=False))
        logger.info("Updated %s: k=%.0f, sl_sat=%.2f", out_path, best_k, best_sl_sat)

    # Summary table.
    print("\n── Calibration summary ──────────────────────────────────")
    print(f"  Global  k={best_k:5.0f}  sl_sat={best_sl_sat:.2f}  MAE={best_mae:.4f}  (default MAE={default_mae:.4f})")
    for zone, r in zone_results.items():
        print(f"  {zone:<8} k={r['k']:5.0f}  sl_sat={r['sl_sat']:.2f}  MAE={r['mae']:.4f}  (n={r['n']})")
    print()

    # Save zone results for reference.
    zone_df = pd.DataFrame([
        {"zone": z, **v} for z, v in zone_results.items()
    ])
    out_csv = repo_root / "outputs" / "eval" / "somosclean_calibration.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    zone_df.to_csv(out_csv, index=False)
    logger.info("Zone calibration results → %s", out_csv)


if __name__ == "__main__":
    main()
