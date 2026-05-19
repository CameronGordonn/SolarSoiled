"""solarsoiled — single CLI for the Stage 1 / Stage 2 / recommend pipeline.

Wraps the existing numbered research scripts in ``scripts/`` as library
calls. The 14 scripts keep working standalone for research; this CLI is the
canonical entrypoint for partner-facing AOI runs.

Run ``solarsoiled --help`` for usage.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import typer

from solarsoiled.aoi import Aoi, parse_aoi, write_aoi_geojson
from solarsoiled.manifest import write_manifest
from solarsoiled.paths import AoiPaths, REPO_ROOT
from solarsoiled.recommend import recommend_cleaning, write_recommendation
from solarsoiled.registry import RegistryError, ResolvedWeights, resolve as resolve_weights, resolve_soiling


app = typer.Typer(
    add_completion=False,
    help="SolarSoiled — detect arrays, score soiling risk, recommend cleaning.",
    no_args_is_help=True,
)


# ---------- script-loader helpers ----------

_SCRIPT_FILES = {
    "tile": REPO_ROOT / "scripts" / "02_tile_naip_image.py",
    "infer": REPO_ROOT / "scripts" / "04_infer_yolov8_seg.py",
    "eval": REPO_ROOT / "scripts" / "05_evaluate_yolov8_seg.py",
    "eval_sweep": REPO_ROOT / "scripts" / "05b_eval_threshold_sweep.py",
    "rca": REPO_ROOT / "scripts" / "05c_per_detection_rca.py",
    "sahi_sweep": REPO_ROOT / "scripts" / "05d_sahi_threshold_sweep.py",
    "bucket_overlays": REPO_ROOT / "scripts" / "labeling" / "18_bucket_overlays.py",
    "export_polygons": REPO_ROOT / "scripts" / "06_export_polygons_geojson.py",
    "extract_features": REPO_ROOT / "scripts" / "07_extract_array_features.py",
    "build_features": REPO_ROOT / "scripts" / "09_build_soiling_features.py",
    "predict_risk": REPO_ROOT / "scripts" / "11_predict_soiling_risk.py",
}


def _load_script(key: str):
    """Import a numbered script (whose filename starts with a digit) by path."""
    path = _SCRIPT_FILES[key]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(f"_solarsoiled_{key}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _aoi_centroid_wgs84(aoi: Aoi) -> tuple[float, float]:
    """Return ``(lat, lon)`` WGS84 centroid for an AOI."""
    c = aoi.polygon.centroid
    return (float(c.y), float(c.x))


def _resolve_aoi(spec: str, partner_id: str | None) -> tuple[Aoi, AoiPaths]:
    aoi = parse_aoi(spec, partner_id=partner_id)
    paths = AoiPaths(aoi.aoi_id)
    paths.ensure_root()
    write_aoi_geojson(aoi, paths.aoi_geojson)
    return aoi, paths


def _resolve_weights(spec: str) -> ResolvedWeights:
    try:
        return resolve_weights(spec)
    except RegistryError as exc:
        raise typer.BadParameter(str(exc), param_hint="--weights") from exc


def _resolve_soiling_model(spec: str) -> ResolvedWeights:
    try:
        return resolve_soiling(spec)
    except RegistryError as exc:
        raise typer.BadParameter(str(exc), param_hint="--soiling-model") from exc


# ---------- subcommands ----------


@app.command()
def tile(
    aoi: str = typer.Option(..., "--aoi", help="bbox 'minx,miny,maxx,maxy' OR path to GeoJSON polygon"),
    partner_id: str | None = typer.Option(None, "--partner-id", help="Override AOI directory name"),
    download: bool = typer.Option(False, "--download", help="Pass AOI to scripts/02 for GeoAI download"),
) -> None:
    """Tile NAIP imagery for an AOI into 640px PNGs + tile_index.json."""
    aoi_obj, paths = _resolve_aoi(aoi, partner_id)
    mod = _load_script("tile")
    download_arg = aoi if download else None
    mod.main(
        download_aoi=download_arg,
        out_tiles_dir=paths.tiles_dir,
        out_tile_index=paths.tile_index,
    )
    typer.echo(f"tile → {paths.tiles_dir}")


@app.command()
def detect(
    aoi: str = typer.Option(..., "--aoi"),
    weights: str = typer.Option(
        ...,
        "--weights",
        help="Registered model name/alias (e.g. 'production', 'stage1-v0.5-baseline') or filesystem path to a .pt",
    ),
    partner_id: str | None = typer.Option(None, "--partner-id"),
    conf: float = typer.Option(0.40, "--conf"),
    iou: float = typer.Option(0.50, "--iou"),
    sahi: bool = typer.Option(True, "--sahi/--no-sahi"),
) -> None:
    """Run YOLOv11 detection over an AOI's tiles → arrays.geojson."""
    resolved = _resolve_weights(weights)
    aoi_obj, paths = _resolve_aoi(aoi, partner_id)
    if not paths.tile_index.is_file():
        raise typer.BadParameter(
            f"missing {paths.tile_index} — run `solarsoiled tile --aoi …` first"
        )

    infer = _load_script("infer")
    infer_argv = [
        "--weights", str(resolved.path),
        "--source", str(paths.tiles_dir),
        "--project", str(paths.root),
        "--name", "detect",
        "--conf", str(conf),
        "--iou", str(iou),
    ]
    if sahi:
        infer_argv.append("--sahi")
    infer.main(infer_argv)

    export = _load_script("export_polygons")
    export.export_polygons(
        labels_dir=paths.detect_labels_dir,
        tile_index_path=paths.tile_index,
        output_geojson=paths.arrays_geojson,
    )

    # Overlay the detect manifest with full registry metadata so `model_version`
    # is the semantic name (e.g. 'stage1-v0.5-baseline') rather than the
    # script-derived run tag, and the catalog metrics + limitations propagate.
    tile_inputs = sorted(str(p) for p in paths.tiles_dir.glob("*.png"))
    write_manifest(
        paths.detect_dir,
        stage="stage1_detect",
        model_version=resolved.model_version,
        model_weights=resolved.path,
        inputs=tile_inputs,
        beta=resolved.beta,
        metrics={
            **dict(resolved.metrics),
            "conf": conf,
            "iou": iou,
            "sahi": int(sahi),
            "n_tiles": len(tile_inputs),
        },
        known_limitations=list(resolved.known_limitations) or None,
        extra={"weights_source": resolved.source},
    )

    typer.echo(f"detect → {paths.arrays_geojson} ({resolved.model_version})")


@app.command()
def score(
    aoi: str = typer.Option(..., "--aoi"),
    soiling_model: str = typer.Option(..., "--soiling-model", help="Registered name/alias (e.g. 'soiling_production') or path to model.ubj"),
    partner_id: str | None = typer.Option(None, "--partner-id"),
    region_config: Path = typer.Option(REPO_ROOT / "configs" / "soiling" / "california.yaml", "--region-config"),
    features_config: Path = typer.Option(REPO_ROOT / "configs" / "soiling" / "features.yaml", "--features-config"),
    as_of: str | None = typer.Option(None, "--as-of", help="YYYY-MM-DD; default = today UTC"),
) -> None:
    """Extract array features → soiling features → risk scores. Writes risk.geojson."""
    resolved_soiling = _resolve_soiling_model(soiling_model)
    aoi_obj, paths = _resolve_aoi(aoi, partner_id)
    if not paths.arrays_geojson.is_file():
        raise typer.BadParameter(
            f"missing {paths.arrays_geojson} — run `solarsoiled detect --aoi …` first"
        )
    paths.features_dir.mkdir(parents=True, exist_ok=True)

    extract = _load_script("extract_features")
    extract.extract_features(
        input_geojson=paths.arrays_geojson,
        out_table=paths.array_features_parquet,
        out_geo=paths.array_features_geo_parquet,
    )

    build = _load_script("build_features")
    build_argv = [
        "--config", str(region_config),
        "--features-config", str(features_config),
        "--arrays", str(paths.array_features_geo_parquet),
        "--out", str(paths.inference_matrix),
    ]
    if as_of:
        build_argv += ["--as-of", as_of]
    build.main(build_argv)

    predict = _load_script("predict_risk")
    predict.main([
        "--model", str(resolved_soiling.path),
        "--features", str(paths.inference_matrix),
        "--arrays", str(paths.array_features_geo_parquet),
        "--out", str(paths.risk_geojson),
    ])
    typer.echo(f"score → {paths.risk_geojson}")


@app.command()
def recommend(
    aoi: str = typer.Option(..., "--aoi"),
    last_cleaned: str = typer.Option(..., "--last-cleaned", help="YYYY-MM-DD"),
    partner_id: str | None = typer.Option(None, "--partner-id"),
    risk_threshold: float = typer.Option(0.6, "--risk-threshold"),
    rain_mm_threshold: float = typer.Option(5.0, "--rain-mm-threshold"),
    min_days_since_clean: int = typer.Option(30, "--min-days-since-clean"),
) -> None:
    """Apply v1 cleaning rule → recommendations.json."""
    aoi_obj, paths = _resolve_aoi(aoi, partner_id)
    if not paths.risk_geojson.is_file():
        raise typer.BadParameter(
            f"missing {paths.risk_geojson} — run `solarsoiled score --aoi …` first"
        )
    payload = recommend_cleaning(
        risk_geojson=paths.risk_geojson,
        last_cleaned=date.fromisoformat(last_cleaned),
        aoi_centroid=_aoi_centroid_wgs84(aoi_obj),
        risk_threshold=risk_threshold,
        rain_mm_threshold=rain_mm_threshold,
        min_days_since_clean=min_days_since_clean,
    )
    write_recommendation(paths.recommendations_json, payload, upstream_manifest=paths.root / "manifest.json")
    typer.echo(f"recommend → {paths.recommendations_json}")
    typer.echo(json.dumps({"rule_fired": payload["rule_fired"], "confidence": payload["confidence"]}))


@app.command()
def run(
    aoi: str = typer.Option(..., "--aoi"),
    weights: str = typer.Option(
        ...,
        "--weights",
        help="Registered model name/alias or filesystem path to a .pt",
    ),
    soiling_model: str = typer.Option(..., "--soiling-model", help="Registered name/alias or path to model.ubj"),
    last_cleaned: str = typer.Option(..., "--last-cleaned"),
    partner_id: str | None = typer.Option(None, "--partner-id"),
    skip_tile: bool = typer.Option(False, "--skip-tile"),
    skip_detect: bool = typer.Option(False, "--skip-detect"),
    skip_score: bool = typer.Option(False, "--skip-score"),
    skip_recommend: bool = typer.Option(False, "--skip-recommend"),
    download: bool = typer.Option(False, "--download"),
) -> None:
    """Chain tile → detect → score → recommend, fail-fast."""
    aoi_obj, paths = _resolve_aoi(aoi, partner_id)
    # Resolve once up-front so we fail fast on a bad name and so the rollup
    # manifest at the bottom can record the resolved version even when
    # --skip-detect was set.
    resolved_weights = _resolve_weights(weights)
    started = datetime.now(timezone.utc).isoformat()

    if not skip_tile:
        tile(aoi=aoi, partner_id=paths.aoi_id, download=download)
    elif not paths.tile_index.is_file():
        raise typer.BadParameter(f"--skip-tile but {paths.tile_index} is missing")

    if not skip_detect:
        detect(aoi=aoi, weights=weights, partner_id=paths.aoi_id, sahi=True, conf=0.40, iou=0.50)
    elif not paths.arrays_geojson.is_file():
        raise typer.BadParameter(f"--skip-detect but {paths.arrays_geojson} is missing")

    if not skip_score:
        score(
            aoi=aoi, soiling_model=soiling_model, partner_id=paths.aoi_id,
            region_config=REPO_ROOT / "configs" / "soiling" / "california.yaml",
            features_config=REPO_ROOT / "configs" / "soiling" / "features.yaml",
            as_of=None,
        )
    elif not paths.risk_geojson.is_file():
        raise typer.BadParameter(f"--skip-score but {paths.risk_geojson} is missing")

    if not skip_recommend:
        recommend(aoi=aoi, last_cleaned=last_cleaned, partner_id=paths.aoi_id)

    # AOI-level rollup manifest indexes each stage's manifest.
    stage_manifests = [
        paths.tiles_dir / "manifest.json",
        paths.detect_dir / "manifest.json",
        paths.root / "manifest.json",  # 06 writes here for arrays.geojson; will be overwritten by rollup
        paths.features_dir / "manifest.json",
        paths.recommendations_json.parent / "manifest.recommend.json",
    ]
    write_manifest(
        paths.root,
        stage="recommend",
        model_version="solarsoiled-run-v1",
        model_weights=None,
        inputs=[str(p) for p in stage_manifests if p.exists()],
        beta=True,
        metrics={
            "skipped_tile": int(skip_tile),
            "skipped_detect": int(skip_detect),
            "skipped_score": int(skip_score),
            "skipped_recommend": int(skip_recommend),
        },
        known_limitations=[
            "Detection below 0.70 mAP50 GA bar",
            "Soiling AUC below 0.70 GA bar",
            "Recommend engine v1 is rule-based; expected_recovery_pct is a placeholder",
        ],
        extra={
            "started_at": started,
            "aoi_root": str(paths.root),
            "stage1_model_version": resolved_weights.model_version,
            "stage1_weights_source": resolved_weights.source,
        },
    )
    typer.echo(f"run → {paths.root}")


def _run_full_eval_pipeline(
    resolved,
    *,
    data: Path | None,
    run_name: str | None,
    report_out: Path | None,
) -> None:
    """Chain 05c → 05c --summarize → 05d → 18_bucket_overlays → build_report."""
    weights_str = str(resolved.path)
    data_args = ["--data", str(data)] if data else []

    # Derive run_name from weights path the same way 05c does when --run-name is omitted,
    # so all artifacts land in the same directory.
    if run_name is None:
        wp = resolved.path
        run_name = wp.parent.parent.name if wp.parent.name == "weights" else wp.stem

    typer.echo(f"[1/5] 05c per-detection RCA (SAHI, conf=0.05, splits=val test) → outputs/eval/{run_name}/")
    rca = _load_script("rca")
    rc = rca.main(
        ["--weights", weights_str, "--sahi", "--conf", "0.05", "--iou", "0.5",
         "--splits", "val", "test", "--run-name", run_name] + data_args
    ) or 0
    if rc:
        raise typer.Exit(code=rc)

    csv_path = REPO_ROOT / "outputs" / "eval" / run_name / "per_detection.csv"

    typer.echo("[2/5] 05c --summarize → failure_modes.json")
    rc = rca.main(["--summarize", "--csv", str(csv_path)]) or 0
    if rc:
        raise typer.Exit(code=rc)

    typer.echo("[3/5] 05d SAHI threshold sweep → sahi_threshold_sweep.csv")
    sahi_sweep = _load_script("sahi_sweep")
    rc = sahi_sweep.main(
        ["--weights", weights_str, "--run-name", run_name] + data_args
    ) or 0
    if rc:
        raise typer.Exit(code=rc)

    typer.echo("[4/5] 18_bucket_overlays (confident_fp + worst_small_fn)")
    overlays = _load_script("bucket_overlays")
    for bucket in ("confident_fp", "worst_small_fn"):
        rc = overlays.main(
            ["--csv", str(csv_path), "--bucket", bucket, "--top", "20"] + data_args
        ) or 0
        if rc:
            raise typer.Exit(code=rc)

    typer.echo("[5/5] Building HTML report")
    from solarsoiled.eval_report import build_report
    eval_dir = REPO_ROOT / "outputs" / "eval" / run_name
    out = build_report(eval_dir, report_out, weights_resolved=resolved)
    typer.echo(f"report → {out}")


@app.command()
def eval(
    weights: str = typer.Option(
        ...,
        "--weights",
        help="Registered model name/alias or filesystem path to a .pt",
    ),
    data: Path | None = typer.Option(None, "--data", help="data.yaml path; defaults handled by script"),
    split: str = typer.Option("val", "--split"),
    threshold_sweep: bool = typer.Option(False, "--threshold-sweep", help="Run scripts/05b instead of 05"),
    metrics_json: Path | None = typer.Option(None, "--metrics-json"),
    report: bool = typer.Option(False, "--report", help="Build single-file HTML quality report from existing eval artifacts in --report-dir."),
    report_dir: Path | None = typer.Option(None, "--report-dir", help="outputs/eval/<run-name>/ to ingest. Defaults to most recent under outputs/eval/."),
    report_out: Path | None = typer.Option(None, "--report-out", help="HTML output path. Defaults to <report-dir>/report.html."),
    full: bool = typer.Option(False, "--full", help="Run the full eval pipeline (05c RCA → 05d SAHI sweep → 18 overlays → HTML report). Slow — runs inference."),
    run_name: str | None = typer.Option(None, "--run-name", help="Artifact directory name under outputs/eval/. Derived from weights if omitted."),
) -> None:
    """Evaluate Stage 1 weights (scripts/05) or run a threshold sweep (scripts/05b)."""
    resolved = _resolve_weights(weights)
    typer.echo(f"eval → resolved weights: {resolved.model_version} ({resolved.path})")

    if full:
        _run_full_eval_pipeline(resolved, data=data, run_name=run_name, report_out=report_out)
        return

    if report:
        from solarsoiled.eval_report import build_report
        out = build_report(report_dir, report_out, weights_resolved=resolved)
        typer.echo(f"report → {out}")
        return

    if threshold_sweep:
        mod = _load_script("eval_sweep")
        argv = ["--weights", str(resolved.path)]
        if data:
            argv += ["--data", str(data)]
        rc = mod.main(argv) or 0
        if rc:
            raise typer.Exit(code=rc)
    else:
        mod = _load_script("eval")
        argv = ["--weights", str(resolved.path), "--split", split]
        if data:
            argv += ["--data", str(data)]
        if metrics_json:
            argv += ["--metrics-json", str(metrics_json)]
        mod.main(argv)


@app.command()
def viz(
    aoi: str = typer.Option(..., "--aoi", help="bbox 'minx,miny,maxx,maxy' OR path to GeoJSON polygon"),
    partner_id: str | None = typer.Option(None, "--partner-id", help="Override AOI directory name"),
    basemap: str = typer.Option("satellite", "--basemap", help="Basemap style: satellite | osm | topo"),
    out: Path | None = typer.Option(None, "--out", help="Override output HTML path"),
) -> None:
    """Render an interactive soiling-risk map from risk.geojson → risk_map.html."""
    from solarsoiled.viz import build_risk_map

    _, paths = _resolve_aoi(aoi, partner_id)

    if not paths.risk_geojson.exists():
        typer.echo(
            f"risk.geojson not found at {paths.risk_geojson}. "
            "Run `solarsoiled score` first.",
            err=True,
        )
        raise typer.Exit(code=1)

    out_path = out or (paths.root / "risk_map.html")
    rec_json = paths.recommendations_json if paths.recommendations_json.exists() else None

    build_risk_map(paths.risk_geojson, out_path, recommendations_json=rec_json, basemap=basemap)

    write_manifest(
        paths.root,
        stage="eval",
        model_version="viz",
        inputs=[str(paths.risk_geojson)],
        metrics={},
        known_limitations=[],
        filename="manifest.viz.json",
    )
    typer.echo(f"viz → {out_path}")


if __name__ == "__main__":
    app()
