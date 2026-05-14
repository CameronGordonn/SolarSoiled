#!/usr/bin/env python3
"""Validate GeoAI installation and pipeline dependencies."""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_scripts = Path(__file__).parent
_root = _scripts.parent


def _test_import(module: str, severity: str = "error") -> bool:
    try:
        __import__(module)
        logger.info(f"✓ {module}")
        return True
    except ImportError as e:
        log = logger.error if severity == "error" else logger.warning
        log(f"{'✗' if severity == 'error' else '⚠'} {module}: {e}")
        return False


def _test_file(path: Path, label: str) -> bool:
    if path.exists():
        logger.info(f"✓ {label}")
        return True
    logger.error(f"✗ {label} not found: {path}")
    return False


def main() -> int:
    logger.info("GeoAI Setup Validation")

    module_tests = [
        ("geoai",           "error"),
        ("geoai.map_tools", "warn"),
        ("geoai.sam",       "warn"),
        ("geoai.geo_agents","warn"),
        ("geoai.download",  "warn"),
        ("rasterio",        "error"),
        ("geopandas",       "error"),
        ("torch",           "error"),
        ("ultralytics",     "error"),
    ]

    file_tests = [
        (_scripts / "08c_visualize_tiles.py",   "Visualization script"),
        (_scripts / "08b_generate_sam_masks.py", "SAM mask script"),
        (_root / "notebooks" / "geoai_exploration.ipynb", "Exploration notebook"),
    ]

    results = [_test_import(m, sev) for m, sev in module_tests]
    results += [_test_file(path, label) for path, label in file_tests]

    passed, total = sum(results), len(results)
    logger.info(f"{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
