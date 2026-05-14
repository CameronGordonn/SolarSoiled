#!/usr/bin/env python3
"""Package the checked-in SolarSoiled landing page into a zip bundle.

This script intentionally treats `solarsoiled-landing/` as the source of truth.
Older versions embedded copies of the HTML/CSS/assets here, which made it easy
for the site and the build output to drift apart silently.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


REQUIRED_FILES = (
    "index.html",
    "styles.css",
    "README.md",
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Zip the SolarSoiled landing page")
    parser.add_argument("--site-dir", type=Path, default=repo_root / "solarsoiled-landing")
    parser.add_argument("--zip-name", type=Path, default=repo_root / "solarsoiled-landing.zip")
    return parser.parse_args()


def collect_site_files(site_dir: Path) -> list[Path]:
    if not site_dir.exists():
        raise FileNotFoundError(f"Site directory not found: {site_dir}")
    if not site_dir.is_dir():
        raise NotADirectoryError(f"Site path is not a directory: {site_dir}")

    missing = [name for name in REQUIRED_FILES if not (site_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required site file(s): {', '.join(missing)}")

    files = sorted(p for p in site_dir.rglob("*") if p.is_file())
    if not files:
        raise RuntimeError(f"No files found under {site_dir}")
    return files


def build_zip(site_dir: Path, zip_name: Path) -> list[Path]:
    files = collect_site_files(site_dir)
    zip_name.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=path.relative_to(site_dir.parent))
    return files


def main() -> None:
    args = parse_args()
    site_dir = args.site_dir.expanduser().resolve()
    zip_name = args.zip_name.expanduser().resolve()

    files = build_zip(site_dir, zip_name)

    print(f"Packaged {len(files)} files from {site_dir}")
    print(f"Zip: {zip_name} ({zip_name.stat().st_size:,} bytes)")
    print(f"Preview: cd {site_dir} && python3 -m http.server 8000")


if __name__ == "__main__":
    main()
