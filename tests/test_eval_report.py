"""Tests for solarsoiled.eval_report — HTML quality report builder."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from solarsoiled.eval_report import (
    Artifacts,
    _html_table,
    _load_artifacts,
    _pr_curve_png_b64,
    _render_worst_offenders,
    build_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_synthetic_artifacts(base: Path, *, include_failure_modes: bool = True, include_sweep: bool = True) -> Path:
    """Write minimal valid eval artifacts under base/run/ and return the run dir."""
    run_dir = base / "run"
    run_dir.mkdir(parents=True)

    # manifest.json
    (run_dir / "manifest.json").write_text(json.dumps({
        "schema_version": "1",
        "stage": "eval",
        "model_version": "test-model-v1",
        "generated_at": "2026-05-05T00:00:00Z",
        "beta": True,
        "model_weights_sha256": "abcdef1234567890",
        "inputs": [],
        "inputs_hash": "aabbcc",
        "known_limitations": ["AUC below GA bar"],
    }), encoding="utf-8")

    # per_detection.csv — 10 rows across 5 tiles
    fields = [
        "tile_id", "domain", "split", "class", "iou", "conf",
        "pred_area_px", "gt_area_px",
        "pred_centroid_x", "pred_centroid_y",
        "gt_centroid_x", "gt_centroid_y",
        "distance_to_image_edge_px", "num_other_panels_in_tile",
        "pred_aspect", "gt_aspect", "weights_run",
    ]
    rows = []
    for i in range(10):
        tile = f"tile_{i // 2:06d}.png"
        cls = "fp" if i % 3 == 0 else ("fn" if i % 3 == 1 else "tp")
        rows.append({
            "tile_id": tile, "domain": "naip", "split": "val",
            "class": cls, "iou": "0.5" if cls == "tp" else "",
            "conf": "0.4" if cls != "fn" else "",
            "pred_area_px": "200", "gt_area_px": "180" if cls != "fp" else "",
            "pred_centroid_x": "100", "pred_centroid_y": "100",
            "gt_centroid_x": "100", "gt_centroid_y": "100",
            "distance_to_image_edge_px": "50",
            "num_other_panels_in_tile": "3",
            "pred_aspect": "1.2", "gt_aspect": "1.1",
            "weights_run": "test-run",
        })
    with (run_dir / "per_detection.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    if include_sweep:
        # sahi_threshold_sweep.csv
        with (run_dir / "sahi_threshold_sweep.csv").open("w", newline="", encoding="utf-8") as fh:
            w2 = csv.DictWriter(fh, fieldnames=["conf", "iou", "tp", "fp", "fn", "precision", "recall", "f1"])
            w2.writeheader()
            for conf in [0.1, 0.2, 0.3]:
                for iou in [0.5, 0.6]:
                    tp = int(40 + conf * 10)
                    fp = int(30 - conf * 10)
                    fn = int(60 - conf * 5)
                    p = tp / (tp + fp) if tp + fp else 0
                    r = tp / (tp + fn) if tp + fn else 0
                    f1 = 2 * p * r / (p + r) if p + r else 0
                    w2.writerow({"conf": conf, "iou": iou, "tp": tp, "fp": fp, "fn": fn,
                                 "precision": p, "recall": r, "f1": f1})

        # sahi_threshold_sweep_best.json
        (run_dir / "sahi_threshold_sweep_best.json").write_text(json.dumps({
            "target_metric": "f1",
            "conf": 0.3,
            "iou": 0.5,
            "tp": 43,
            "fp": 27,
            "fn": 57,
            "precision": 0.614,
            "recall": 0.430,
            "f1": 0.506,
        }), encoding="utf-8")

    if include_failure_modes:
        (run_dir / "failure_modes.json").write_text(json.dumps({
            "n_rows": 10,
            "overall": {"tp": 3, "fp": 4, "fn": 3, "precision": 0.43, "recall": 0.50},
            "by_panel_size": {
                "small_lt800": {"tp": 3, "fp": 4, "fn": 3, "fn_rate": 0.50, "precision": 0.43},
            },
            "by_edge_distance": {
                "near_lt40": {"tp": 2, "fp": 2, "fn": 1, "fn_rate": 0.33, "precision": 0.50},
            },
            "by_density": {
                "alone": {"tp": 0, "fp": 3, "fn": 1, "fn_rate": 1.0, "precision": 0.0},
                "few_1_3": {"tp": 3, "fp": 1, "fn": 2, "fn_rate": 0.40, "precision": 0.75},
            },
        }), encoding="utf-8")

    # label_viz/<run-name>/<bucket>/*.png — 5 tiny 1×1 PNGs
    viz_dir = base.parent / "label_viz" / "run" if base.parent.name != "eval" else base / "label_viz" / "run"
    # keep label_viz as sibling to run's parent so _load_artifacts finds it:
    # outputs/eval/run/ → look at outputs/label_viz/run/
    # Let's write it as <base>/../label_viz/run/
    viz_dir = base / ".." / "label_viz" / "run"
    viz_dir = viz_dir.resolve()
    bucket_dir = viz_dir / "confident_fp"
    bucket_dir.mkdir(parents=True)
    for i in range(5):
        # Minimal valid 1x1 PNG (89 bytes)
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00"
            b"\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx"
            b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00"
            b"\x00IEND\xaeB`\x82"
        )
        (bucket_dir / f"tile_{i:06d}.png").write_bytes(png_bytes)

    return run_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_report_full_artifacts(tmp_path: Path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    run_dir = _write_synthetic_artifacts(eval_dir)

    out = build_report(run_dir, run_dir / "report.html", weights_resolved=None)

    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "<title>" in html
    # best F1 callout present
    assert "F1=0.506" in html
    # PR curve + overlay PNGs (≥6 data:image/png substrings: 1 curve + 5 overlays)
    assert html.count("data:image/png;base64,") >= 6
    # sibling manifest written with stage==eval
    mf = run_dir / "manifest.json"
    assert mf.exists()
    data = json.loads(mf.read_text())
    assert data["stage"] == "eval"
    assert "missing_artifacts" in data


def test_build_report_missing_failure_modes(tmp_path: Path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    run_dir = _write_synthetic_artifacts(eval_dir, include_failure_modes=False)

    out = build_report(run_dir, run_dir / "report.html", weights_resolved=None)
    html = out.read_text(encoding="utf-8")

    # placeholder shown
    assert "not generated yet" in html
    # no exception → manifest still written
    mf = run_dir / "manifest.json"
    assert mf.exists()
    data = json.loads(mf.read_text())
    assert "failure_modes.json" in data["missing_artifacts"]


def test_build_report_missing_sweep(tmp_path: Path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    run_dir = _write_synthetic_artifacts(eval_dir, include_sweep=False)

    out = build_report(run_dir, run_dir / "report.html", weights_resolved=None)
    html = out.read_text(encoding="utf-8")

    assert "not generated yet" in html
    # PR curve absent but document still valid
    assert "<!doctype html>" in html.lower()


def test_pr_curve_png_b64_deterministic_size():
    sweep = [
        {"recall": 0.3, "precision": 0.7, "f1": 0.43},
        {"recall": 0.5, "precision": 0.6, "f1": 0.55},
        {"recall": 0.7, "precision": 0.4, "f1": 0.50},
    ]
    b64 = _pr_curve_png_b64(sweep)
    assert len(b64) > 1024, f"Expected >1KB PNG, got {len(b64)} chars"


def test_worst_offenders_top10():
    # Build 50 rows across 20 tiles, with varying fp+fn counts
    rows = []
    for i in range(50):
        tile = f"tile_{i % 20:06d}.png"
        cls = ["fp", "fn", "tp"][i % 3]
        rows.append({"tile_id": tile, "class": cls, "iou": "", "conf": ""})
    html = _render_worst_offenders(rows)
    # Exactly 10 data rows in the table (header excluded)
    assert html.count("<tr>") == 11  # 1 header + 10 data
    # Ordering: most fp+fn first — tile_000000 (fp) repeats most
    # Just verify the table is present and bounded
    assert "tile_" in html


def test_html_table_color_col():
    rows = [{"f1": 1.0, "label": "best"}, {"f1": 0.1, "label": "worst"}]
    html = _html_table(rows, ["label", "f1"], color_col="f1")
    # F1=1.0 → hsl(120,60%,50%) (lightness 50), F1=0.1 → lightness 86
    assert "hsl(120,60%,50%)" in html
    assert "hsl(120,60%,86%)" in html
