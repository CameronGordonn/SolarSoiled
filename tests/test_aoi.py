"""Unit tests for solarsoiled.aoi."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from solarsoiled.aoi import AOIValidationError, parse_aoi, write_aoi_geojson


def test_parse_bbox_basic():
    aoi = parse_aoi("-122.1,36.9,-122.0,37.0")
    assert aoi.source == "bbox"
    assert aoi.polygon.bounds == (-122.1, 36.9, -122.0, 37.0)
    assert len(aoi.aoi_id) == 10


def test_parse_bbox_partner_id_overrides_hash():
    aoi = parse_aoi("-122.1,36.9,-122.0,37.0", partner_id="acme")
    assert aoi.aoi_id == "acme"


def test_parse_bbox_id_is_stable():
    a = parse_aoi("-122.1,36.9,-122.0,37.0")
    b = parse_aoi("-122.1,36.9,-122.0,37.0")
    assert a.aoi_id == b.aoi_id


def test_parse_bbox_rejects_degenerate():
    with pytest.raises(ValueError):
        parse_aoi("0,0,0,0")
    with pytest.raises(ValueError):
        parse_aoi("0,0,1")  # missing component


def test_parse_geojson_polygon(tmp_path: Path):
    feature = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-122, 36], [-121, 36], [-121, 37], [-122, 37], [-122, 36]]],
        },
    }
    p = tmp_path / "aoi.geojson"
    p.write_text(json.dumps(feature))
    aoi = parse_aoi(str(p))
    assert aoi.source == "geojson"
    assert aoi.polygon.bounds == (-122, 36, -121, 37)


def test_write_aoi_geojson_round_trip(tmp_path: Path):
    aoi = parse_aoi("0,0,1,1", partner_id="t")
    out = write_aoi_geojson(aoi, tmp_path / "out.geojson")
    payload = json.loads(out.read_text())
    assert payload["type"] == "Feature"
    assert payload["properties"]["aoi_id"] == "t"
    assert payload["geometry"]["type"] == "Polygon"


def test_parse_bbox_rejects_out_of_range_lon():
    """Web-Mercator coordinates accidentally passed as lon/lat must fail loud."""
    with pytest.raises(AOIValidationError, match="lon"):
        parse_aoi("-13580977,4500000,-13570000,4510000")  # meters, not degrees


def test_parse_bbox_rejects_out_of_range_lat():
    with pytest.raises(AOIValidationError, match="lat"):
        parse_aoi("-122.1,-95.0,-122.0,-94.9")


def test_parse_geojson_rejects_non_wgs84_crs(tmp_path: Path):
    """Explicit non-WGS84 CRS must be rejected so partners reproject upstream."""
    feature = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            }
        ],
    }
    p = tmp_path / "merc.geojson"
    p.write_text(json.dumps(feature))
    with pytest.raises(AOIValidationError, match="non-WGS84"):
        parse_aoi(str(p))


def test_parse_geojson_rejects_self_intersecting_polygon(tmp_path: Path):
    """A bowtie polygon should be rejected before it corrupts downstream tiling."""
    feature = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]],  # bowtie
        },
    }
    p = tmp_path / "bowtie.geojson"
    p.write_text(json.dumps(feature))
    with pytest.raises(AOIValidationError, match="invalid"):
        parse_aoi(str(p))


def test_parse_geojson_accepts_explicit_wgs84_crs(tmp_path: Path):
    """Files that explicitly tag WGS84 should still parse cleanly."""
    feature = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-122, 36], [-121, 36], [-121, 37], [-122, 37], [-122, 36]]],
                },
            }
        ],
    }
    p = tmp_path / "wgs84.geojson"
    p.write_text(json.dumps(feature))
    aoi = parse_aoi(str(p))
    assert aoi.source == "geojson"
