"""End-to-end smoke harness for the ``solarsoiled`` CLI.

This test stays SKIPPED in normal CI until two conditions hold:
1. ``models/registry.yaml`` has a ``smoketest`` entry/alias pointing at a
   real Stage 1 weights file.
2. The env var ``SOLARSOILED_SMOKE_TILES`` points at a directory of pre-tiled
   NAIP PNGs plus a ``tile_index.json``.

Once Josh's next training cut lands and we register it as ``smoketest``,
this test becomes a one-command end-to-end gate:

    SOLARSOILED_SMOKE_TILES=outputs/aoi/<id>/tiles \\
      pytest tests/test_smoke_run.py -v

It exercises: registry resolve → ``solarsoiled detect`` → arrays.geojson +
manifest.json carrying the registered ``model_version``. Output lands under
``outputs/aoi/smoke-test-pytest/`` and is cleaned up afterward.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from solarsoiled.cli import app
from solarsoiled.paths import AoiPaths
from solarsoiled.registry import RegistryError, resolve


SMOKE_ALIAS = "smoketest"
SMOKE_TILES_ENV = "SOLARSOILED_SMOKE_TILES"
SMOKE_PARTNER_ID = "smoke-test-pytest"


def _smoke_weights_available() -> bool:
    try:
        resolve(SMOKE_ALIAS)
    except RegistryError:
        return False
    return True


def _smoke_tiles() -> Path | None:
    raw = os.environ.get(SMOKE_TILES_ENV)
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_dir() and (p / "tile_index.json").is_file() else None


@pytest.mark.skipif(
    not _smoke_weights_available(),
    reason="register a 'smoketest' entry/alias in models/registry.yaml to enable",
)
@pytest.mark.skipif(
    _smoke_tiles() is None,
    reason=f"set {SMOKE_TILES_ENV} to a tiles dir with tile_index.json to enable",
)
def test_smoke_detect_through_registry():
    src_tiles = _smoke_tiles()
    assert src_tiles is not None

    paths = AoiPaths(SMOKE_PARTNER_ID)
    if paths.root.exists():
        shutil.rmtree(paths.root)
    paths.ensure_root()
    paths.tiles_dir.mkdir(parents=True, exist_ok=True)
    for entry in src_tiles.iterdir():
        if entry.is_file():
            shutil.copy2(entry, paths.tiles_dir / entry.name)

    try:
        bbox = "-122.05,36.95,-122.00,37.00"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "detect",
                "--aoi", bbox,
                "--weights", SMOKE_ALIAS,
                "--partner-id", SMOKE_PARTNER_ID,
            ],
        )
        assert result.exit_code == 0, result.output

        manifest_path = paths.detect_dir / "manifest.json"
        assert manifest_path.is_file(), "expected detect/manifest.json after run"
        manifest = json.loads(manifest_path.read_text())
        resolved = resolve(SMOKE_ALIAS)
        assert manifest["model_version"] == resolved.model_version
        assert manifest["stage"] == "stage1_detect"
        assert paths.arrays_geojson.is_file()
    finally:
        if paths.root.exists():
            shutil.rmtree(paths.root)
