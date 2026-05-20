"""Stage 1 → Stage 2 contract test.

Verifies that the output of the detection stage (arrays.geojson) is correctly
consumed by the soiling risk pipeline and that both risk.geojson and
recommendations.json conform to their expected schemas.

Uses a pre-baked fixture arrays.geojson (3 small polygons in Santa Cruz) so
no NAIP download or YOLO inference is required. Skipped unless the soiling
model weights are present (runs/soiling/run_optionb/model.ubj).

Run:
    pytest tests/test_stage2_contract.py -v
Or to run all smoke + contract tests:
    pytest tests/test_smoke_run.py tests/test_stage2_contract.py -v
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import pytest

from solarsoiled.paths import AoiPaths
from solarsoiled.registry import RegistryError, resolve_soiling

FIXTURE_ARRAYS = Path(__file__).parent / "fixtures" / "arrays_santa_cruz.geojson"
PARTNER_ID = "smoke-stage2-contract"
SOILING_ALIAS = "soiling_smoketest"

# Required output schemas
_RISK_REQUIRED_PROPS = {"array_id", "risk_score", "area_m2"}
_RECOMMEND_REQUIRED_KEYS = {"window_start", "confidence", "rule_fired", "inputs", "beta"}


def _soiling_weights_available() -> bool:
    try:
        resolve_soiling(SOILING_ALIAS)
        return True
    except RegistryError:
        return False


@pytest.fixture(autouse=True)
def clean_output():
    paths = AoiPaths(PARTNER_ID)
    if paths.root.exists():
        shutil.rmtree(paths.root)
    yield
    if paths.root.exists():
        shutil.rmtree(paths.root)


@pytest.mark.skipif(
    not _soiling_weights_available(),
    reason=f"register '{SOILING_ALIAS}' in models/registry.yaml and ensure model.ubj is present",
)
def test_score_produces_valid_risk_geojson():
    """score stage: arrays.geojson → risk.geojson with risk_score on every feature."""
    import sys
    import importlib.util

    from solarsoiled.paths import REPO_ROOT

    paths = AoiPaths(PARTNER_ID)
    paths.ensure_root()
    paths.features_dir.mkdir(parents=True, exist_ok=True)

    # Copy fixture arrays into the expected location
    shutil.copy2(FIXTURE_ARRAYS, paths.arrays_geojson)

    resolved = resolve_soiling(SOILING_ALIAS)

    # extract_features
    script_path = REPO_ROOT / "scripts" / "07_extract_array_features.py"
    spec = importlib.util.spec_from_file_location("_extract", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.extract_features(
        input_geojson=paths.arrays_geojson,
        out_table=paths.array_features_parquet,
        out_geo=paths.array_features_geo_parquet,
    )
    assert paths.array_features_geo_parquet.is_file(), "array_features.geo.parquet not produced"

    # build_features
    script_path = REPO_ROOT / "scripts" / "09_build_soiling_features.py"
    spec = importlib.util.spec_from_file_location("_build", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main([
        "--config", str(REPO_ROOT / "configs" / "soiling" / "california.yaml"),
        "--features-config", str(REPO_ROOT / "configs" / "soiling" / "features.yaml"),
        "--arrays", str(paths.array_features_geo_parquet),
        "--out", str(paths.inference_matrix),
    ])
    assert paths.inference_matrix.is_file(), "inference_matrix.parquet not produced"

    # predict_risk
    script_path = REPO_ROOT / "scripts" / "11_predict_soiling_risk.py"
    spec = importlib.util.spec_from_file_location("_predict", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main([
        "--model", str(resolved.path),
        "--features", str(paths.inference_matrix),
        "--arrays", str(paths.array_features_geo_parquet),
        "--out", str(paths.risk_geojson),
    ])

    assert paths.risk_geojson.is_file(), "risk.geojson not produced"
    fc = json.loads(paths.risk_geojson.read_text())
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 3, f"expected 3 features, got {len(fc['features'])}"
    for feat in fc["features"]:
        props = feat["properties"]
        missing = _RISK_REQUIRED_PROPS - props.keys()
        assert not missing, f"risk.geojson feature missing props: {missing}"
        score = props["risk_score"]
        assert 0.0 <= score <= 1.0, f"risk_score {score} out of [0, 1]"


@pytest.mark.skipif(
    not _soiling_weights_available(),
    reason=f"register '{SOILING_ALIAS}' in models/registry.yaml and ensure model.ubj is present",
)
def test_recommend_produces_valid_schema():
    """Full pipeline: fixture arrays → score → recommend; check output schemas."""
    from solarsoiled.recommend import (
        recommend_cleaning,
        recommend_per_array,
        write_array_recommendations,
        write_recommendation,
    )

    # Run score first (reuse logic above via a direct call)
    # Shortcut: use an existing risk.geojson from outputs if available,
    # otherwise run the score stage inline.
    import sys
    import importlib.util
    from solarsoiled.paths import REPO_ROOT

    paths = AoiPaths(PARTNER_ID)
    paths.ensure_root()
    paths.features_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FIXTURE_ARRAYS, paths.arrays_geojson)

    resolved = resolve_soiling(SOILING_ALIAS)

    for script_key, func_name, argv in [
        ("07_extract_array_features.py", "extract_features", None),
        ("09_build_soiling_features.py", "main", [
            "--config", str(REPO_ROOT / "configs" / "soiling" / "california.yaml"),
            "--features-config", str(REPO_ROOT / "configs" / "soiling" / "features.yaml"),
            "--arrays", str(paths.array_features_geo_parquet),
            "--out", str(paths.inference_matrix),
        ]),
        ("11_predict_soiling_risk.py", "main", [
            "--model", str(resolved.path),
            "--features", str(paths.inference_matrix),
            "--arrays", str(paths.array_features_geo_parquet),
            "--out", str(paths.risk_geojson),
        ]),
    ]:
        sp = REPO_ROOT / "scripts" / script_key
        spec = importlib.util.spec_from_file_location(f"_{script_key}", sp)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if argv is None:
            mod.extract_features(
                input_geojson=paths.arrays_geojson,
                out_table=paths.array_features_parquet,
                out_geo=paths.array_features_geo_parquet,
            )
        else:
            mod.main(argv)

    assert paths.risk_geojson.is_file()

    # recommend
    payload = recommend_cleaning(
        risk_geojson=paths.risk_geojson,
        last_cleaned=date(2025, 12, 1),
        aoi_centroid=(36.974, -122.030),
        risk_threshold=0.6,
    )
    write_recommendation(paths.recommendations_json, payload)

    array_rows = recommend_per_array(
        paths.risk_geojson, payload, risk_threshold=0.6
    )
    write_array_recommendations(paths.array_recommendations_json, array_rows)

    # Schema checks
    assert paths.recommendations_json.is_file()
    rec = json.loads(paths.recommendations_json.read_text())
    missing = _RECOMMEND_REQUIRED_KEYS - rec.keys()
    assert not missing, f"recommendations.json missing keys: {missing}"

    assert paths.array_recommendations_json.is_file()
    arr_recs = json.loads(paths.array_recommendations_json.read_text())
    assert isinstance(arr_recs, list)
    assert len(arr_recs) == 3
    for row in arr_recs:
        assert "array_id" in row
        assert "risk_score" in row
