"""Tests for solarsoiled.manifest — the sidecar manifest writer."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from solarsoiled.manifest import (
    KNOWN_STAGES,
    SCHEMA_VERSION,
    build_manifest,
    hash_inputs,
    write_manifest,
)


ISO8601_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def test_unknown_stage_rejected():
    with pytest.raises(ValueError, match="unknown stage"):
        build_manifest(stage="not_a_stage", model_version="x")


def test_known_stages_cover_pipeline():
    for s in ("tile", "stage1_detect", "stage2_score", "recommend", "eval"):
        assert s in KNOWN_STAGES


def test_timestamp_iso8601_utc():
    m = build_manifest(stage="tile", model_version="v0")
    assert ISO8601_Z.match(m["generated_at"]), m["generated_at"]


def test_weights_hash_matches_sha256(tmp_path: Path):
    payload = b"weights-bytes-\x00\x01\x02"
    weights = tmp_path / "fake.pt"
    weights.write_bytes(payload)

    m = build_manifest(
        stage="stage1_detect",
        model_version="stage1-test",
        model_weights=weights,
    )

    expected = hashlib.sha256(payload).hexdigest()
    assert m["model_weights_sha256"] == expected
    assert m["model_weights_path"] == str(weights)


def test_missing_weights_records_none(tmp_path: Path):
    missing = tmp_path / "nope.pt"
    m = build_manifest(
        stage="stage1_detect",
        model_version="stage1-test",
        model_weights=missing,
    )
    assert m["model_weights_path"] == str(missing)
    assert m["model_weights_sha256"] is None


def test_inputs_hash_stable_across_path_order(tmp_path: Path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("alpha")
    b.write_text("beta")

    h1 = hash_inputs([a, b])
    h2 = hash_inputs([b, a])
    assert h1 == h2  # sorted internally


def test_inputs_hash_changes_when_content_changes(tmp_path: Path):
    a = tmp_path / "a.txt"
    a.write_text("alpha")
    h1 = hash_inputs([a])
    a.write_text("alpha-changed")
    h2 = hash_inputs([a])
    assert h1 != h2


def test_inputs_hash_works_for_non_path_shapes():
    h = hash_inputs({"foo": 1, "bar": [1, 2]})
    assert isinstance(h, str) and len(h) == 64


def test_schema_roundtrip_via_json(tmp_path: Path):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"abc")

    m = build_manifest(
        stage="stage2_score",
        model_version="run_x",
        model_weights=weights,
        inputs=[weights],
        beta=True,
        metrics={"cv_auc": 0.63, "holdout_auc": 0.66},
        known_limitations=["AQ features sparse pre-2022"],
    )
    roundtrip = json.loads(json.dumps(m, default=str))
    assert roundtrip["stage"] == "stage2_score"
    assert roundtrip["schema_version"] == SCHEMA_VERSION
    assert roundtrip["metrics"]["cv_auc"] == 0.63
    assert roundtrip["beta"] is True
    assert roundtrip["known_limitations"] == ["AQ features sparse pre-2022"]


def test_write_manifest_creates_output_dir(tmp_path: Path):
    out_dir = tmp_path / "nested" / "run_x"
    written = write_manifest(
        out_dir,
        stage="eval",
        model_version="eval-test",
    )
    assert written == out_dir / "manifest.json"
    payload = json.loads(written.read_text())
    assert payload["stage"] == "eval"
    assert payload["model_version"] == "eval-test"
    assert payload["beta"] is True


def test_extra_fields_round_trip(tmp_path: Path):
    m = build_manifest(
        stage="tile",
        model_version="v0",
        extra={"tile_count": 249, "source_crs": "EPSG:32610"},
    )
    assert m["tile_count"] == 249
    assert m["source_crs"] == "EPSG:32610"


def test_extra_collision_rejected():
    with pytest.raises(ValueError, match="collides"):
        build_manifest(stage="tile", model_version="v0", extra={"stage": "x"})
