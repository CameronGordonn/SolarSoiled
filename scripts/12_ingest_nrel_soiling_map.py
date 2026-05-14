"""Ingest the NREL PV Soiling Map JSON into the CSV schema expected by
`src/soiling/labels.py:load_nrel_soiling_map()`.

Source: https://www.nrel.gov/docs/libraries/pv/soiling_data.json
  (the JSON served by the interactive map at https://www.nrel.gov/pv/soiling)

Usage:
    python scripts/12_ingest_nrel_soiling_map.py                   # download + convert
    python scripts/12_ingest_nrel_soiling_map.py --input <path>    # use local JSON
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

SOURCE_URL = "https://www.nrel.gov/docs/libraries/pv/soiling_data.json"

logger = logging.getLogger(__name__)


def fetch_raw(dest: Path) -> None:
    req = Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())
    logger.info("Downloaded %d bytes → %s", dest.stat().st_size, dest)


def _coerce_iwsr(value) -> tuple[float | None, bool]:
    """109/255 NREL records report IWSR as ">0.99" — the Micheli/Deceglie
    extraction method doesn't resolve soiling losses below 1%. Coerce these to
    0.995 (mid of clean range) and flag via iwsr_censored so downstream code
    can drop or re-weight them."""
    if value is None:
        return None, False
    if isinstance(value, (int, float)):
        return float(value), False
    s = str(value).strip()
    if s.startswith(">"):
        try:
            floor = float(s[1:])
        except ValueError:
            return None, True
        return (floor + 1.0) / 2.0, True
    try:
        return float(s), False
    except ValueError:
        return None, False


def convert(raw_path: Path, out_path: Path) -> pd.DataFrame:
    with raw_path.open() as fh:
        raw = json.load(fh)

    rows = []
    for station_id, rec in raw.items():
        iwsr, censored = _coerce_iwsr(rec.get("IWSR"))
        iwsr_lower, _ = _coerce_iwsr(rec.get("IWSR lower"))
        iwsr_upper, _ = _coerce_iwsr(rec.get("IWSR upper"))
        rows.append(
            {
                "station_id": station_id,
                "latitude": rec["Latitude"],
                "longitude": rec["Longitude"],
                "iwsr": iwsr,
                "iwsr_censored": censored,
                "iwsr_lower": iwsr_lower,
                "iwsr_upper": iwsr_upper,
                "measurement_type": rec.get("Measurement type"),
                "state": rec.get("State"),
                "county": rec.get("County"),
                "tilt_deg": rec.get("Tilt"),
                "mounting": rec.get("Mounting"),
                "months_in_data_set": rec.get("Months in data set"),
            }
        )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["latitude", "longitude", "iwsr"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d stations → %s (%d censored)", len(df), out_path, int(df["iwsr_censored"].sum()))
    return df


def convert_panel(raw_path: Path, out_path: Path) -> pd.DataFrame:
    """Per-(station, year) panel from the `Annual IWSR` block in the NREL JSON.

    Multiplies the dataset ~5x and lets the model match each label to the
    corresponding year's weather, instead of comparing one annualized IWSR
    against today-minus-180-days as the summary CSV forces.
    """
    with raw_path.open() as fh:
        raw = json.load(fh)

    rows = []
    for station_id, rec in raw.items():
        annual = rec.get("Annual IWSR")
        if not annual or len(annual) < 2:
            continue
        # First row is a header: ['Year', 'IWSR', 'IWSR lower', 'IWSR upper']
        for entry in annual[1:]:
            if not entry or len(entry) < 2:
                continue
            year = entry[0]
            iwsr_y, censored = _coerce_iwsr(entry[1])
            iwsr_lo_y, _ = _coerce_iwsr(entry[2]) if len(entry) > 2 else (None, False)
            iwsr_hi_y, _ = _coerce_iwsr(entry[3]) if len(entry) > 3 else (None, False)
            if iwsr_y is None:
                continue
            rows.append(
                {
                    "station_id": station_id,
                    "year": int(year),
                    "latitude": rec["Latitude"],
                    "longitude": rec["Longitude"],
                    "iwsr": iwsr_y,
                    "iwsr_censored": censored,
                    "iwsr_lower": iwsr_lo_y,
                    "iwsr_upper": iwsr_hi_y,
                    "measurement_type": rec.get("Measurement type"),
                    "state": rec.get("State"),
                    "county": rec.get("County"),
                    "tilt_deg": rec.get("Tilt"),
                    "mounting": rec.get("Mounting"),
                    "months_in_data_set": rec.get("Months in data set"),
                }
            )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["latitude", "longitude", "iwsr"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(
        "Wrote %d (station, year) rows → %s (%d censored, %d unique stations, years %d–%d)",
        len(df), out_path, int(df["iwsr_censored"].sum()),
        df["station_id"].nunique(),
        df["year"].min(), df["year"].max(),
    )
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    repo_root = Path(__file__).resolve().parent.parent
    default_raw = repo_root / "data/external/nrel_soiling_map_raw.json"
    default_out = repo_root / "data/external/nrel_soiling_map.csv"
    default_panel_out = repo_root / "data/external/nrel_soiling_map_annual.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=default_raw)
    parser.add_argument("--output", type=Path, default=default_out)
    parser.add_argument("--panel-output", type=Path, default=default_panel_out)
    parser.add_argument("--no-download", action="store_true", help="Skip re-download if input exists")
    args = parser.parse_args()

    if not args.input.exists() or not args.no_download:
        if args.input.exists():
            logger.info("Re-downloading; pass --no-download to reuse local copy")
        fetch_raw(args.input)

    df = convert(args.input, args.output)
    panel = convert_panel(args.input, args.panel_output)

    print(df.head())
    print(f"\n{len(df)} stations | IWSR range: {df['iwsr'].min():.3f}–{df['iwsr'].max():.3f}")
    print(f"States covered: {df['state'].nunique()} ({', '.join(sorted(df['state'].dropna().unique()))})")
    print(f"\nAnnual panel: {len(panel)} (station, year) rows across {panel['station_id'].nunique()} stations")
    print(f"  years: {sorted(panel['year'].unique())}")
    print(f"  censored rows: {int(panel['iwsr_censored'].sum())}")


if __name__ == "__main__":
    main()
